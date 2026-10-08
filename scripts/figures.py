"""The README's overview figure for one demo: the phone frame with my pinch
path, the same path on the table in centimetres, and (optionally) the LIBERO
frame of its replay.

Usage: python scripts/figures.py data/raw/<session> demo_020 [--sim outputs/figures/demo_020_sim.png]
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p2p import retarget as rt
from p2p.camera import Intrinsics
from p2p.hand import pinch_px
from p2p.sim_map import Demo, trim
from p2p.video import frames

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("demo")
    ap.add_argument("--sim", type=Path)
    args = ap.parse_args()

    cfg = rt.Session(**json.loads((args.session / "session.json").read_text()))
    intr = Intrinsics.load(args.session / "intrinsics.json")
    d = np.load(ROOT / "data" / "cache" / args.session.name / f"{args.demo}.npz")
    c = rt.clip_trajectory(d["t"], d["hand_px"], d["T_cam_table"], intr.K, intr.dist, cfg, 20.0)

    # phone frame at the grasp, with the pinch pixels of the whole clip
    t_grasp = c["t"][c["grasp"]]
    k = int(np.argmin(np.abs(d["t"] - t_grasp)))
    frame = next(f for i, (_, f) in enumerate(frames(args.session / f"{args.demo}.mp4")) if i == k)
    px = pinch_px(d["hand_px"])
    ok = np.isfinite(px).all(1)
    for a, b in zip(px[ok][:-1].astype(int), px[ok][1:].astype(int)):
        cv2.line(frame, tuple(a), tuple(b), (31, 103, 217), 4, cv2.LINE_AA)
    T = d["T_cam_table"][k]
    if np.isfinite(T).all():
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        cv2.drawFrameAxes(frame, intr.K, intr.dist, rvec, T[:3, 3], 0.06, 4)

    panels = 3 if args.sim else 2
    fig, ax = plt.subplots(1, panels, figsize=(5.2 * panels, 3.6),
                           gridspec_kw={"width_ratios": [1.6, 1, 1][:panels]})
    ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax[0].set_title("phone, at the grasp: board axes and my pinch path", fontsize=9)
    ax[0].axis("off")

    m, xy, g, r = 100 * c["measured"], 100 * c["pinch"][:, :2], c["grasp"], c["release"]
    kept = trim(Demo(c["pinch"], c["grip"], g, r))
    kx = 100 * kept.pinch[:, :2]
    ax[1].plot(m[:, 0], m[:, 1], color="0.65", lw=1, label="measured on the rim plane")
    ax[1].plot(xy[:, 0], xy[:, 1], color="C0", lw=1, ls=":", label="not replayed (far from the contacts)")
    ax[1].plot(kx[:, 0], kx[:, 1], color="C0", lw=1.8, label="replayed")
    ax[1].plot(*xy[g], "o", color="C1", label="grasp (pause)")
    ax[1].plot(*xy[r], "s", color="C2", label="release (pause)")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x on the table (cm)")
    ax[1].set_ylabel("y (cm)")
    ax[1].legend(fontsize=7, loc="lower left")
    ax[1].set_title("table frame, from the board", fontsize=9)

    if args.sim:
        ax[2].imshow(plt.imread(args.sim))
        ax[2].axis("off")
        ax[2].set_title("LIBERO replay", fontsize=9)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "overview.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
