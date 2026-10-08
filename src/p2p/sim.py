"""LIBERO side: scene geometry, a pose-tracking servo, episode recording.

The servo turns absolute pinch targets into the 7-D delta actions LIBERO's
operational space controller takes (position in units of 5 cm, rotation in
units of 0.5 rad, gripper -1 open / +1 close). Recording those actions with
the rendered frames gives episodes in exactly the format LIBERO policies are
trained and evaluated on.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from scipy.spatial.transform import Rotation

from p2p.sim_map import Scene, approach_from, rot_z

TASK_SUITE = "libero_goal"
TASK_LANGUAGE = "put the bowl on the plate"
BOWL, PLATE = "akita_black_bowl_1", "plate_1"
POS_SCALE, ROT_SCALE = 0.05, 0.5
SETTLE_STEPS = 10  # same as LeRobot's LiberoEnv


def make_env(resolution: int = 256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    tid = next(i for i in range(suite.n_tasks) if suite.get_task(i).language == TASK_LANGUAGE)
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution,
                             camera_widths=resolution)
    return env, suite.get_task_init_states(tid)


def settle(env):
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(np.r_[np.zeros(6), -1.0])
    return obs


def _body_boxes(sim, body_name):
    """World-space centres and corners of the box collision geoms of a body."""
    m, d = sim.model, sim.data
    bid = m.body(body_name).id
    centres, corners = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != 6:  # 6 = box
            continue
        half = m.geom_size[g]
        R = d.geom_xmat[g].reshape(3, 3)
        centres.append(d.geom_xpos[g].copy())
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(d.geom_xpos[g] + R @ (half * [sx, sy, sz]))
    return np.array(centres), np.array(corners)


def scene_of(env) -> Scene:
    """Bowl rim, plate top and table height measured from collision geometry.

    The bowl's wall is a ring of thin boxes; the rim radius used for the pinch
    is the radius of the upper ring's centres, i.e. the middle of the wall.
    """
    sim = env.env.sim
    bc, b = _body_boxes(sim, f"{BOWL}_main")
    _, p = _body_boxes(sim, f"{PLATE}_main")
    bowl_xy = sim.data.body(f"{BOWL}_main").xpos[:2]
    plate_xy = sim.data.body(f"{PLATE}_main").xpos[:2]
    upper = bc[bc[:, 2] >= np.median(bc[:, 2])]
    bowl_r = float(np.median(np.linalg.norm(upper[:, :2] - bowl_xy, axis=1)))
    table_z = float(min(b[:, 2].min(), p[:, 2].min()))
    others = [n for n in env.env.objects_dict if n not in (BOWL, PLATE)]
    obstacles = np.array([sim.data.body(f"{n}_main").xpos[:2] for n in others]).reshape(-1, 2)
    return Scene(np.r_[bowl_xy, b[:, 2].max()], bowl_r, np.r_[plate_xy, p[:, 2].max()],
                 table_z, obstacles)


def eef(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.env.robots[0]
    return np.array(robot.controller.ee_pos), np.array(robot.controller.ee_ori_mat)


def finger_heading(R: np.ndarray) -> float:
    """Heading on the table of the axis the Panda fingers close along.

    In robosuite's Panda the fingers slide along the end effector's x axis
    (measured from the two fingertip bodies, not assumed)."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


class Servo:
    """Proportional tracking of (position, finger heading, gripper) targets.

    Orientation is kept pointing down; only the heading about the vertical is
    tracked, and modulo pi, because the gripper is symmetric under a half turn.
    """

    def __init__(self, R_down: np.ndarray):
        self.R_down = R_down
        self.h0 = finger_heading(R_down)

    def action(self, env, pos, heading, closed) -> np.ndarray:
        p, R = eef(env)
        dp = np.clip((pos - p) / POS_SCALE, -1, 1)
        cur = finger_heading(R)
        dh = (heading - cur + np.pi / 2) % np.pi - np.pi / 2  # nearest equivalent heading
        R_target = rot_z(cur + dh - self.h0) @ self.R_down
        dr = Rotation.from_matrix(R_target @ R.T).as_rotvec()
        dr = np.clip(dr / ROT_SCALE, -1, 1)
        return np.r_[dp, dr, 1.0 if closed else -1.0]


def play(env, P, heading, grip, hz: float, on_step=None, arrive_tol: float = 0.01,
         max_wait_s: float = 1.0, finger_s: float = 0.5, hold_s: float = 1.0,
         stop_after_success: int | None = None) -> tuple[bool, int]:
    """Execute pinch targets from the current robot pose.

    A straight approach from the start pose comes first. The gripper command
    switches when the fingers get there, not when the human's did: at each
    open/close switch the playback holds the target until the end effector is
    within ``arrive_tol`` of it (at most ``max_wait_s``), then holds it for
    another ``finger_s`` while the fingers move. The servo lags the human
    motion by a few centimetres, and the Panda's fingers take about half a
    second to close; on the human's clock the gripper closed on air.

    ``on_step(obs_before, action)`` sees every step, for recording. Returns
    (success, steps).
    """
    p0, R0 = eef(env)
    servo = Servo(R0)
    pre = approach_from(p0, P[0], hz)
    targets = [(p, heading[0], False) for p in pre] + list(zip(P, heading, grip > 0.5))
    targets += [(P[-1], heading[-1], False)] * int(hold_s * hz)

    obs = env.env._get_observations()
    success, steps, after = False, 0, 0
    prev_closed = False

    def step(p, h, c):
        nonlocal obs, success, steps, after
        a = servo.action(env, p, h, c)
        if on_step is not None:
            on_step(obs, a)
        obs, _, _, _ = env.step(a)
        steps += 1
        success = success or env.check_success()
        after += success
        return stop_after_success is not None and after >= stop_after_success

    for p, h, c in targets:
        if c != prev_closed:
            for _ in range(int(max_wait_s * hz)):
                if np.linalg.norm(eef(env)[0] - p) < arrive_tol:
                    break
                if step(p, h, prev_closed):
                    return True, steps
            for _ in range(int(finger_s * hz)):
                if step(p, h, c):
                    return True, steps
        prev_closed = c
        if step(p, h, c):
            return True, steps
    return success, steps


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """robosuite's conversion (x, y, z, w quaternion), as used by LeRobot's
    LiberoProcessorStep. The gripper points down, a rotation close to pi, where
    this and scipy's as_rotvec choose different but equivalent vectors; the
    state has to match the evaluation pipeline, so this one is copied."""
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-10:
        return np.zeros(3)
    return q[:3] * 2.0 * np.arccos(w) / den


def lerobot_frame(obs, action) -> dict:
    """One frame in the layout of the lerobot/libero dataset.

    LIBERO renders upside down; the dataset (and LeRobot's eval processor)
    rotate both cameras by 180 degrees.
    """
    aa = quat2axisangle(obs["robot0_eef_quat"])
    return {
        "observation.images.image": obs["agentview_image"][::-1, ::-1].copy(),
        "observation.images.image2": obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
        "observation.state": np.r_[obs["robot0_eef_pos"], aa,
                                   obs["robot0_gripper_qpos"]].astype(np.float32),
        "action": np.asarray(action, dtype=np.float32),
    }
