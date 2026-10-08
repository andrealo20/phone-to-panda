"""Replay ablation: what each design decision is worth, without any learning.

Every usable phone demo is replayed open loop in the same LIBERO scenes (the
first fixed benchmark states), once with the full method and once with each
decision switched off. Success is LIBERO's own check (bowl on plate).

Usage: python scripts/ablate.py data/raw/<session> [--states 4]
Writes outputs/ablation.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from p2p import retarget as rt
from p2p import sim
from p2p.camera import Intrinsics
from p2p.sim_map import Demo, map_demo, trim

ROOT = Path(__file__).resolve().parents[1]
HZ = 20.0
PINCH_BELOW = 0.016  # as in generate.py

# name -> (retarget switches, mapping switches, playback switches)
VARIANTS = {
    "full method": ({}, {}, {}),
    "descent at 45 degrees": ({"slope": 1.0}, {}, {}),
    "rays cut at the rim plane": ({"replay_height_rays": False}, {}, {}),
    "measured carry path": ({"straight": False}, {}, {}),
    "no mirror image": ({}, {"allow_mirror": False}, {}),
    "wrist turned to my side": ({}, {"snap": False}, {}),
    "my whole approach path": ({}, {"trim": False}, {}),
    "no wait for arrival": ({}, {}, {"max_wait_s": 0.0}),
    "no pause for the fingers": ({}, {}, {"finger_s": 0.0}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--states", type=int, default=4)
    args = ap.parse_args()

    cfg = rt.Session(**json.loads((args.session / "session.json").read_text()))
    intr = Intrinsics.load(args.session / "intrinsics.json")
    cache = ROOT / "data" / "cache" / args.session.name
    clips = sorted(f for f in cache.glob("demo_???.npz") if (cache / f"{f.stem}_traj.npz").exists())

    env, init_states = sim.make_env()
    results = {name: {} for name in VARIANTS}
    for f in clips:
        d = np.load(f)
        for name, (rkw, mkw, pkw) in VARIANTS.items():
            try:
                c = rt.clip_trajectory(d["t"], d["hand_px"], d["T_cam_table"], intr.K, intr.dist,
                                       cfg, HZ, **rkw)
            except ValueError:
                results[name][f.stem] = [False] * args.states
                continue
            demo = Demo(c["pinch"], c["grip"], c["grasp"], c["release"])
            if mkw.get("trim", True):
                demo = trim(demo)
            wins = []
            for k in range(args.states):
                env.reset()
                env.set_init_state(init_states[k])
                sim.settle(env)
                mkw2 = {key: val for key, val in mkw.items() if key not in ("snap", "trim")}
                if mkw.get("snap", True):
                    mkw2["snap_heading"] = sim.finger_heading(sim.eef(env)[1])
                P, heading = map_demo(demo, sim.scene_of(env), PINCH_BELOW, **mkw2)
                ok, _ = sim.play(env, P, heading, demo.grip, HZ, stop_after_success=1, **pkw)
                wins.append(bool(ok))
            results[name][f.stem] = wins
            print(f"{f.stem} | {name}: {sum(wins)}/{len(wins)}", flush=True)

    table = {}
    for name, per_demo in results.items():
        flat = [w for ws in per_demo.values() for w in ws]
        table[name] = {"success": sum(flat), "runs": len(flat), "rate": round(sum(flat) / len(flat), 3)}
    out = ROOT / "outputs" / "ablation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"states": args.states, "demos": [f.stem for f in clips],
                               "table": table, "per_demo": results}, indent=1))
    for name, row in table.items():
        print(f"{name:28s} {row['success']:3d}/{row['runs']}  {100 * row['rate']:5.1f}%")


if __name__ == "__main__":
    main()
