"""Replay variants screened on fresh scenes, before any of them is trained on.

Usage: python scripts/screen_servo.py [--scenes 4]
Every usable clip is replayed in the same freshly sampled scenes (seeds far
from the ones generate.py uses, and none of the 50 fixed evaluation states)
under each variant. Besides LIBERO's success, it records what the policy would
learn from: steps per phase, rotation commands during the carry, commands at
their limit and still steps, to compare with the teleoperated demos.
Writes outputs/screen_servo.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from p2p import sim
from p2p.sim_map import map_demo, retime_carry

from generate import HZ, SIM_PINCH_BELOW, STEPS_AFTER_SUCCESS, load_demo

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 7_000_000
CARRY_STEPS = 40  # teleop carry: median 48 recorded steps, 5 of them still

# name -> (servo options, carry steps or None)
VARIANTS = {
    "current servo": ({}, None),
    "yaw only": ({"rot": "yaw"}, None),
    "carry retimed": ({}, CARRY_STEPS),
}


def stats(actions: np.ndarray, eef: np.ndarray) -> dict:
    closed = actions[:, 6] > 0
    g = int(np.argmax(closed)) if closed.any() else len(actions)
    opened = np.flatnonzero(~closed[g:])
    r = g + int(opened[0]) if opened.size else len(actions)
    still = np.r_[np.linalg.norm(np.diff(eef, axis=0), axis=1) < 0.001, False]
    carry = slice(g, r)
    return {
        "steps": len(actions), "approach": g, "carry": r - g,
        "still_carry": int(still[carry].sum()), "still": int(still.sum()),
        "rot_carry": np.abs(actions[carry, 3:6]).mean(0).round(3).tolist() if r > g else None,
        "rot_all": np.abs(actions[:, 3:6]).mean(0).round(3).tolist(),
        "at_limit": float((np.abs(actions[:, :3]) >= 1 - 1e-6).any(1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--session", default="s01")
    ap.add_argument("--clips", type=int, help="only the first n clips (smoke test)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "screen_servo.json")
    args = ap.parse_args()

    files = sorted((ROOT / "data" / "cache" / args.session).glob("demo_*_traj.npz"))[: args.clips]
    env, _ = sim.make_env()
    runs = []
    for f in files:
        demo0 = load_demo(f)
        for k in range(args.scenes):
            for name, (skw, carry) in VARIANTS.items():
                demo = retime_carry(demo0, carry) if carry else demo0
                env.seed(SEED0 + 1000 * files.index(f) + k)
                env.reset()
                sim.settle(env)
                P, heading = map_demo(demo, sim.scene_of(env), SIM_PINCH_BELOW,
                                      snap_heading=sim.finger_heading(sim.eef(env)[1]))
                A, E = [], []

                def rec(obs, a):
                    A.append(a)
                    E.append(obs["robot0_eef_pos"].copy())

                ok, _ = sim.play(env, P, heading, demo.grip, HZ, on_step=rec,
                                 stop_after_success=STEPS_AFTER_SUCCESS, servo=skw)
                row = {"clip": f.stem.removesuffix("_traj"), "scene": k, "variant": name, "success": bool(ok),
                       **stats(np.array(A), np.array(E))}
                runs.append(row)
                print(f"{row['clip']} scene {k} | {name}: {'ok' if ok else 'fail'} {row['steps']} steps", flush=True)
        args.out.write_text(json.dumps(runs, indent=1))

    print()
    for name in VARIANTS:
        rs = [r for r in runs if r["variant"] == name]
        ok = [r for r in rs if r["success"]]
        med = lambda k: np.median([r[k] for r in ok]) if ok else float("nan")
        rot = np.mean([r["rot_carry"] for r in ok if r["rot_carry"]], axis=0).round(3) if ok else None
        print(f"{name:40s} {len(ok):3d}/{len(rs)}  steps {med('steps'):.0f}  approach {med('approach'):.0f}  "
              f"carry {med('carry'):.0f} (still {med('still_carry'):.0f})  rot carry {rot}  "
              f"at limit {np.mean([r['at_limit'] for r in ok]):.0%}")


if __name__ == "__main__":
    main()
