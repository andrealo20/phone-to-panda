"""Runs a synthetic demonstration through the simulator side.

A hand-made pinch trajectory (reach a rim, lift, carry 25 cm, set down,
retract), in a layout unlike LIBERO's, is mapped with p2p.sim_map and played
on several of the benchmark's initial states. It checks the mapping, the
servo and the success detection without any phone data.

Usage: python scripts/sim_smoke.py [n_states] [--video out.mp4]
"""

import argparse

import imageio.v2 as imageio
import numpy as np

from p2p import sim
from p2p.sim_map import Demo, map_demo

HZ = 20.0
PINCH_BELOW = 0.016


def synthetic_demo() -> Demo:
    grasp = np.array([0.20, 0.10, 0.045])
    release = grasp + np.array([0.22, -0.12, 0.015])
    side = np.array([-1.0, -0.3, 0.0]) / np.hypot(1.0, 0.3)
    up = np.array([0, 0, 0.10])

    def seg(a, b, secs):
        s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, int(secs * HZ)))
        return a + s[:, None] * (b - a)

    parts = [seg(grasp + up + 0.08 * side, grasp, 1.0), np.repeat(grasp[None], 12, 0),
             seg(grasp, grasp + up, 0.8), seg(grasp + up, release + up, 1.5),
             seg(release + up, release, 0.8), np.repeat(release[None], 12, 0),
             seg(release, release + up, 0.8)]
    pinch = np.vstack(parts)
    lens = np.cumsum([len(p) for p in parts])
    g, r = lens[0] + 2, lens[4] + 6
    grip = np.zeros(len(pinch))
    grip[g:r] = 1
    return Demo(pinch, grip, g, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_states", type=int, nargs="?", default=5)
    ap.add_argument("--video")
    args = ap.parse_args()

    demo = synthetic_demo()
    env, init_states = sim.make_env()
    frames = []
    wins = 0
    for k in range(args.n_states):
        env.reset()
        env.set_init_state(init_states[k])
        sim.settle(env)
        P, heading = map_demo(demo, sim.scene_of(env), PINCH_BELOW)
        rec = (lambda obs, a: frames.append(obs["agentview_image"][::-1, ::-1])) if k == 0 else None
        ok, _ = sim.play(env, P, heading, demo.grip, HZ, on_step=rec)
        wins += ok
        print(f"state {k}: success={ok}", flush=True)
    print(f"{wins}/{args.n_states} successful")
    if args.video:
        imageio.mimsave(args.video, frames, fps=HZ)


if __name__ == "__main__":
    main()
