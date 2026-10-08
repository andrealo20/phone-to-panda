"""Per-frame camera pose (from the board) and 2D hand landmarks for every demo.

Usage: python scripts/track.py data/raw/<session> [--overlay]
Writes data/cache/<session>/demo_XXX.npz and, with --overlay, an MP4 with the
board axes and the hand drawn on top, for checking by eye.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from p2p.camera import BoardDetector, Intrinsics, table_to_camera
from p2p.hand import HandTracker
from p2p.video import frames

ROOT = Path(__file__).resolve().parents[1]
MAX_BOARD_ERR_PX = 2.0


def track(clip: Path, det: BoardDetector, intr: Intrinsics, overlay: Path | None):
    tracker = HandTracker()
    out = {k: [] for k in ("t", "T_cam_table", "board_err", "hand_px")}
    prev, writer = None, None
    for t, bgr in frames(clip):
        T, berr = np.full((4, 4), np.nan), np.nan
        hit = det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if hit is not None:
            T, berr = table_to_camera(*hit, intr, prev)
            if berr > MAX_BOARD_ERR_PX:
                T, berr = table_to_camera(*hit, intr)  # retry without the seed
            if berr > MAX_BOARD_ERR_PX:
                T = np.full((4, 4), np.nan)
        prev = None if np.isnan(T).any() else T

        px = tracker(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(round(t * 1000)))
        if px is None:
            px = np.full((21, 2), np.nan)
        for k, v in zip(out, (t, T, berr, px)):
            out[k].append(v)

        if overlay is not None:
            if writer is None:
                h, w = bgr.shape[:2]
                writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            if prev is not None:
                rvec, _ = cv2.Rodrigues(T[:3, :3])
                cv2.drawFrameAxes(bgr, intr.K, intr.dist, rvec, T[:3, 3], 0.05, 3)
            for p in px[np.isfinite(px[:, 0])].astype(int):
                cv2.circle(bgr, tuple(p), 4, (0, 255, 255), -1)
            writer.write(bgr)
    tracker.close()
    if writer is not None:
        writer.release()
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((args.session / "session.json").read_text())
    intr = Intrinsics.load(args.session / "intrinsics.json")
    det = BoardDetector(cfg["square_mm"] / 1000.0)
    cache = ROOT / "data" / "cache" / args.session.name
    cache.mkdir(parents=True, exist_ok=True)

    for clip in sorted(args.session.glob("demo_*.mp4")):
        overlay = cache / f"{clip.stem}_overlay.mp4" if args.overlay else None
        d = track(clip, det, intr, overlay)
        np.savez_compressed(cache / f"{clip.stem}.npz", **d)
        print(f"{clip.stem}: {len(d['t'])} frames, board {np.isfinite(d['board_err']).mean():.0%}, "
              f"hand {np.isfinite(d['hand_px'][:, 0, 0]).mean():.0%}, "
              f"board err {np.nanmedian(d['board_err']):.2f} px")


if __name__ == "__main__":
    main()
