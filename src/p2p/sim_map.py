"""Map a table-frame pinch trajectory onto a simulated scene.

Only two moments of the demonstration are tied to the scene: the grasp (the
pinch is on the bowl rim) and the release (the bowl stands on the plate). The
rest of the motion keeps its shape.

1. Bowl and plate are symmetric about their vertical axes, so the demo can be
   rotated freely: it is turned so that the carry direction (grasp point to
   release point) matches the simulated bowl -> plate direction. Mirroring it
   across the carry axis is just as valid. My table was empty and LIBERO's is
   not, so of the two mirror images the one whose open fingers stay farther
   from the other objects is used.
2. Grasp anchor: the simulated pinch goes on the simulated rim, with the
   fingers closing across the rim wall. The side is the one the hand came
   from, snapped to the nearer of the two sides the Panda reaches without
   turning its wrist (``snap_heading``): the teleoperated LIBERO demos never
   turn the wrist, and replays that did gave a policy that hesitated over the
   bowl (0 of 50 against 42 of 50, see the README).
3. Release anchor: the simulated bowl ends centred on the simulated plate.
4. Between the two, the shift blends with the distance carried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scene:
    """Simulated object geometry. z of a centre = rim height / plate top."""
    bowl_c: np.ndarray   # (3,)
    bowl_r: float        # rim radius
    plate_c: np.ndarray  # (3,)
    table_z: float
    obstacles: np.ndarray = None  # (M,2) centres of the other objects on the table


@dataclass
class Demo:
    pinch: np.ndarray  # (N,3) table frame
    grip: np.ndarray   # (N,) 0 open, 1 closed
    grasp: int
    release: int


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def heading(v: np.ndarray) -> float:
    return float(np.arctan2(v[1], v[0]))


def approach_side(pinch: np.ndarray, g: int, reach: float = 0.03) -> np.ndarray:
    """Unit vector on the table from the grasp point towards where the hand
    came from: the last point before the grasp at least ``reach`` away. The
    hand reaches the rim from outside the bowl, so this points away from the
    bowl's centre."""
    d = np.linalg.norm(pinch[: g + 1, :2] - pinch[g, :2], axis=1)
    far = np.flatnonzero(d > reach)
    if far.size:
        v = pinch[far[-1], :2] - pinch[g, :2]
    else:  # the clip starts at the bowl: take the side away from the carry
        v = pinch[g, :2] - pinch[-1, :2]
    return v / np.linalg.norm(v)


FINGER_REACH = 0.04  # open Panda finger, from the grasp point outwards


def trim(demo: Demo, reach: float = 0.10) -> Demo:
    """Keep the demo from the last time the hand came within ``reach`` of the
    grasp point until it first moves ``reach`` away from the release point.

    Before that the path says where I happened to be standing, not how to do
    the task, and the robot starts from its own home pose instead. Replaying
    my whole path taught the policy my detours: it circled the bowl and never
    grasped (3 of 50).
    """
    P, g, r = demo.pinch, demo.grasp, demo.release
    near_g = np.linalg.norm(P[: g + 1, :2] - P[g, :2], axis=1) < reach
    start = g - int(np.argmin(near_g[::-1])) + 1 if not near_g.all() else 0
    far_r = np.linalg.norm(P[r:, :2] - P[r, :2], axis=1) >= reach
    end = r + int(np.argmax(far_r)) if far_r.any() else len(P) - 1
    keep = slice(start, end + 1)
    return Demo(P[keep], demo.grip[keep], g - start, r - start)


def map_demo(demo: Demo, sim: Scene, pinch_below_rim: float, allow_mirror: bool = True,
             snap_heading: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pinch targets (N,3), finger heading (N,)) in the sim frame.

    The heading is the radial direction of the grasp, held for the whole demo:
    the gripper closes across the rim wall and the bowl does not turn. With
    ``snap_heading`` (the gripper's closing axis at rest) the grasp side is
    snapped to the nearer end of that axis, so the wrist never turns.
    """
    options = [_map(demo, sim, pinch_below_rim, mirror, snap_heading)
               for mirror in (False, True)[: 1 + allow_mirror]]
    P, h, _ = max(options, key=lambda o: o[2])
    return P, h


def _map(demo: Demo, sim: Scene, pinch_below_rim: float, mirror: bool, snap_heading):
    g, r = demo.grasp, demo.release
    theta = heading(sim.plate_c - sim.bowl_c) - heading(demo.pinch[r] - demo.pinch[g])
    P = (rot_z(theta) @ (demo.pinch - demo.pinch[g]).T).T
    if mirror:  # reflect across the carry axis, which now passes through the origin
        u = (sim.plate_c - sim.bowl_c)[:2]
        u = u / np.linalg.norm(u)
        P[:, :2] = 2 * (P[:, :2] @ u)[:, None] * u - P[:, :2]

    side = approach_side(P, g)
    if snap_heading is not None:
        axis = np.array([np.cos(snap_heading), np.sin(snap_heading)])
        side = axis if side @ axis >= 0 else -axis
    rim_z = sim.bowl_c[2] - pinch_below_rim
    grasp_target = np.r_[sim.bowl_c[:2] + sim.bowl_r * side, rim_z]

    on_plate = sim.plate_c[2] - sim.table_z  # the bowl is lifted by the plate's height
    release_target = np.r_[sim.plate_c[:2] + sim.bowl_r * side, rim_z + on_plate]

    alpha = carry_progress(P, g, r)
    shift = (1 - alpha)[:, None] * (grasp_target - P[g]) + alpha[:, None] * (release_target - P[r])

    finger = sim.bowl_c[:2] + (sim.bowl_r + FINGER_REACH) * side
    if sim.obstacles is None or not len(sim.obstacles):
        clearance = np.inf
    else:
        clearance = float(np.min(np.linalg.norm(sim.obstacles - finger, axis=1)))
    return P + shift, np.full(len(P), heading(side)), clearance


def carry_progress(P: np.ndarray, g: int, r: int) -> np.ndarray:
    """0 up to the grasp, 1 from the release, in between the fraction of the
    horizontal distance covered so far.

    Blending on distance instead of time keeps the shift still while the
    fingers close and while the bowl is lifted or lowered; blending on time
    slid the gripper sideways during the grasp and pulled the rim out of it.
    """
    step = np.r_[0.0, np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)]
    step[: g + 1] = 0.0
    step[r + 1:] = 0.0
    s = np.cumsum(step)
    return s / s[-1] if s[-1] > 1e-9 else (np.arange(len(P)) >= r).astype(float)


def approach_from(home: np.ndarray, first: np.ndarray, hz: float,
                  speed: float = 0.15) -> np.ndarray:
    """Straight line from the robot's start pose to the first demo target."""
    n = max(2, int(np.ceil(np.linalg.norm(first - home) / speed * hz)))
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))  # ease in and out
    return home + s[:, None] * (first - home)
