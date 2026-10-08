"""Episode indices of one task in lerobot/libero, for the teleoperation baseline.

Usage: python scripts/libero_episodes.py [n] [--seed S]
Prints a list usable as --dataset.episodes: n teleoperated demos of
"put the bowl on the plate", drawn at random (the dataset has 49 for this task).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from p2p.sim import TASK_LANGUAGE

REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def episodes_of_task(task: str = TASK_LANGUAGE) -> list[int]:
    root = Path(snapshot_download("lerobot/libero", repo_type="dataset", revision=REVISION,
                                  allow_patterns=["meta/*", "data/*"]))
    tasks = pd.read_parquet(next((root / "meta").glob("tasks*.parquet")))
    ti = int(tasks.loc[task, "task_index"])
    eps = set()
    for f in sorted((root / "data").glob("*/*.parquet")):
        d = pd.read_parquet(f, columns=["episode_index", "task_index"])
        eps.update(d.loc[d.task_index == ti, "episode_index"].unique().tolist())
    return sorted(int(e) for e in eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    idx = episodes_of_task()
    if args.n is not None:
        idx = sorted(np.random.default_rng(args.seed).choice(idx, args.n, replace=False).tolist())
    print(f"[{','.join(map(str, idx))}]")


if __name__ == "__main__":
    main()
