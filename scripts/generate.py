"""Phone trajectories -> LIBERO episodes in LeRobot format.

Every phone demonstration is replayed in several freshly sampled LIBERO scenes
(the bddl placement sampler, not the 50 fixed benchmark states that evaluation
uses). Only episodes where LIBERO reports success are kept. The kept fraction
is itself a result: it measures how often an open-loop replay of my hand works.

Usage:
  python scripts/generate.py data/raw/<session> --per-demo 8 --out outputs/datasets/phone

The feature schema is copied from the lerobot/libero dataset so that the
episodes are interchangeable with the official teleoperated ones.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from p2p import sim
from p2p.sim_map import Demo, map_demo, trim

ROOT = Path(__file__).resolve().parents[1]
HZ = 20
STEPS_AFTER_SUCCESS = 10
# Grasp depth below the simulated rim. The teleoperators of the official demos
# close the gripper a median 1.6 cm below it (49 demos of this task).
SIM_PINCH_BELOW = 0.016


def libero_features() -> dict:
    from huggingface_hub import snapshot_download
    from libero_episodes import REVISION
    path = Path(snapshot_download("lerobot/libero", repo_type="dataset", revision=REVISION,
                                  allow_patterns=["meta/info.json"]))
    feats = json.loads((path / "meta" / "info.json").read_text())["features"]
    default = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return {k: v for k, v in feats.items() if k not in default}


def load_demo(f: Path) -> Demo:
    d = np.load(f)
    return trim(Demo(d["pinch"], d["grip"], int(d["grasp"]), int(d["release"])))


def rollout(env, demo: Demo):
    """Execute one mapped demo. Returns (frames, success)."""
    P, heading = map_demo(demo, sim.scene_of(env), SIM_PINCH_BELOW,
                          snap_heading=sim.finger_heading(sim.eef(env)[1]))
    frames = []
    ok, _ = sim.play(env, P, heading, demo.grip, HZ, stop_after_success=STEPS_AFTER_SUCCESS,
                     on_step=lambda obs, a: frames.append(sim.lerobot_frame(obs, a)))
    return frames, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--per-demo", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "datasets" / "phone")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cache = ROOT / "data" / "cache" / args.session.name
    files = sorted(cache.glob("demo_*_traj.npz"))
    ds = LeRobotDataset.create(repo_id=f"local/{args.out.name}", fps=HZ,
                               features=libero_features(), root=args.out,
                               robot_type="panda", use_videos=True)
    env, _ = sim.make_env()
    log = []
    for f in files:
        demo = load_demo(f)
        for k in range(args.per_demo):
            env.seed(args.seed * 100_000 + len(log))
            env.reset()
            sim.settle(env)
            frames, ok = rollout(env, demo)
            log.append({"demo": f.stem.removesuffix("_traj"), "try": k, "success": bool(ok),
                        "steps": len(frames)})
            if ok:
                for fr in frames:
                    ds.add_frame({**fr, "task": sim.TASK_LANGUAGE})
                ds.save_episode()
            print(f"{f.stem} try {k}: {'ok' if ok else 'fail'} ({len(frames)} steps)", flush=True)
    ds.finalize()
    (args.out / "generation_log.json").write_text(json.dumps(log, indent=1))
    rate = np.mean([r["success"] for r in log])
    print(f"kept {sum(r['success'] for r in log)}/{len(log)} episodes ({rate:.0%})")


if __name__ == "__main__":
    main()
