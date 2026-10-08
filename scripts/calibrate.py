"""Phone intrinsics from the session's calibration clip.

Usage: python scripts/calibrate.py data/raw/<session>
"""

import json
import sys
from pathlib import Path

import cv2

from p2p.camera import calibrate
from p2p.video import frames


def main(session: Path) -> None:
    cfg = json.loads((session / "session.json").read_text())
    clip = session / "calib.mp4"
    n = int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n // 200)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in frames(clip, step)]
    intr = calibrate(gray, cfg["square_mm"] / 1000.0)
    intr.save(session / "intrinsics.json")
    print(f"{len(gray)} frames, rms {intr.rms_px:.3f} px")
    print("K =\n", intr.K.round(1))
    print("dist =", intr.dist.ravel().round(4))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
