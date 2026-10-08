from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def frames(path: Path, step: int = 1) -> Iterator[tuple[float, np.ndarray]]:
    """Yields (timestamp_s, BGR frame). Phone videos often have a variable
    frame rate, so timestamps come from the container, not from an index."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, frame
            i += 1
    finally:
        cap.release()


def first_frame(path: Path) -> np.ndarray:
    return next(frames(path))[1]
