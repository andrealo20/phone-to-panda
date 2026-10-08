"""Render one phone demo replayed in LIBERO, for the README.

Writes docs/media/replay_<demo>.gif and outputs/figures/<demo>_sim.png (the
frame at the grasp, used as the third panel of the overview figure).

Usage: python scripts/render_replay.py data/raw/<session> demo_020 [--state 0]
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from p2p import retarget as rt
from p2p import sim
from p2p.camera import Intrinsics
from p2p.sim_map import Demo, map_demo, trim

ROOT = Path(__file__).resolve().parents[1]
HZ = 20.0
PINCH_BELOW = 0.016  # as in generate.py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("demo")
    ap.add_argument("--state", type=int, default=0)
    args = ap.parse_args()

    cfg = rt.Session(**json.loads((args.session / "session.json").read_text()))
    intr = Intrinsics.load(args.session / "intrinsics.json")
    d = np.load(ROOT / "data" / "cache" / args.session.name / f"{args.demo}.npz")
    c = rt.clip_trajectory(d["t"], d["hand_px"], d["T_cam_table"], intr.K, intr.dist, cfg, HZ)
    demo = trim(Demo(c["pinch"], c["grip"], c["grasp"], c["release"]))

    env, init_states = sim.make_env(resolution=384)
    env.reset()
    env.set_init_state(init_states[args.state])
    sim.settle(env)
    P, heading = map_demo(demo, sim.scene_of(env), PINCH_BELOW,
                          snap_heading=sim.finger_heading(sim.eef(env)[1]))
    frames, grip_cmd = [], []

    def rec(obs, a):
        frames.append(obs["agentview_image"][::-1, ::-1].copy())
        grip_cmd.append(a[6])

    ok, _ = sim.play(env, P, heading, demo.grip, HZ, on_step=rec)
    print(f"{args.demo} in state {args.state}: success={ok}, {len(frames)} steps")

    closed = int(np.argmax(np.array(grip_cmd) > 0))
    out_png = ROOT / "outputs" / "figures" / f"{args.demo}_sim.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png, frames[min(closed + 8, len(frames) - 1)])
    gif = ROOT / "docs" / "media" / f"replay_{args.demo}.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    small = [f[::2, ::2] for f in frames[::3]]  # 192 px, every third step: keeps the GIF under 2 MB
    imageio.mimsave(gif, small, duration=0.15, loop=0)
    print("wrote", out_png, gif)


if __name__ == "__main__":
    main()
