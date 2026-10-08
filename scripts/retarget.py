"""Tracked frames -> pinch trajectories in the table frame, at the sim rate.

Usage: python scripts/retarget.py data/raw/<session>

Writes data/cache/<session>/demo_XXX_traj.npz, retarget_report.json and
retarget.png (top view of every demo with its grasp and release points).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p2p import retarget as rt
from p2p.camera import Intrinsics

ROOT = Path(__file__).resolve().parents[1]
SIM_HZ = 20.0


def main(session: Path) -> None:
    cfg = rt.Session(**json.loads((session / "session.json").read_text()))
    intr = Intrinsics.load(session / "intrinsics.json")
    cache = ROOT / "data" / "cache" / session.name

    report, files = {}, sorted(cache.glob("demo_???.npz"))
    cols = min(6, len(files))
    rows = -(-len(files) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat[len(files):]:
        ax.axis("off")
    for f, ax in zip(files, axes.flat):
        d = np.load(f)
        ax.set_title(f.stem, fontsize=9)
        try:
            c = rt.clip_trajectory(d["t"], d["hand_px"], d["T_cam_table"], intr.K, intr.dist, cfg, SIM_HZ)
        except ValueError as e:  # no clear pause at the bowl or at the plate
            (cache / f"{f.stem}_traj.npz").unlink(missing_ok=True)
            report[f.stem] = {"skipped": str(e)}
            ax.set_title(f"{f.stem}: skipped", fontsize=9)
            continue
        np.savez(cache / f"{f.stem}_traj.npz", **{k: c[k] for k in ("t", "pinch", "grip", "grasp", "release")})

        tu, xy, g, r = c["t"], c["pinch"][:, :2], c["grasp"], c["release"]
        report[f.stem] = {
            "duration_s": round(float(tu[-1] - tu[0]), 2),
            "grasp_s": round(float(tu[g] - tu[0]), 2),
            "release_s": round(float(tu[r] - tu[0]), 2),
            "carried_cm": round(100 * float(np.linalg.norm(xy[r] - xy[g])), 1),
            "longest_gap_s": round(c["gap"], 2),
            "board_visible": round(float(np.isfinite(d["board_err"]).mean()), 2),
            "hand_visible": round(float(np.isfinite(d["hand_px"][:, 0, 0]).mean()), 2),
        }
        ax.plot(100 * c["measured"][:, 0], 100 * c["measured"][:, 1], lw=0.8, color="0.65", label="measured")
        ax.plot(100 * xy[:, 0], 100 * xy[:, 1], lw=1.2, label="replayed")
        ax.plot(*(100 * xy[g]), "o", label="grasp")
        ax.plot(*(100 * xy[r]), "s", label="release")
        ax.set_aspect("equal")
        ax.set_xlabel("x (cm)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cache / "retarget.png", dpi=110)
    (cache / "retarget_report.json").write_text(json.dumps(report, indent=2))
    for name, row in report.items():
        print(name, row)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
