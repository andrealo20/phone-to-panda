"""2D hand landmarks from MediaPipe.

Only the image positions are used. MediaPipe also predicts 3D landmarks, and
solving PnP on them gives a hand depth, but on these clips that depth is
unusable: the camera sees the back of the hand, the fingertips are inside the
bowl, and when the hand turns sideways to set the bowl down the palm's
apparent width halves, so the estimated depth doubles (the README shows
the plots). The pinch height comes from the table instead, see p2p.retarget.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

THUMB_TIP, INDEX_TIP = 4, 8

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "hand_landmarker.task"


def _model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    # A low threshold keeps the hand while the fingers are hidden in the bowl:
    # 0.5 found it in 33-59% of frames, 0.2 in 55-69%.
    def __init__(self, min_conf: float = 0.2):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray, t_ms: int) -> np.ndarray | None:
        """21x2 landmark pixels, or None."""
        import mediapipe as mp
        res = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
        if not res.hand_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])

    def close(self) -> None:
        self.landmarker.close()


def pinch_px(px: np.ndarray) -> np.ndarray:
    """Midpoint of thumb and index fingertips, (..., 2)."""
    return 0.5 * (px[..., THUMB_TIP, :] + px[..., INDEX_TIP, :])

