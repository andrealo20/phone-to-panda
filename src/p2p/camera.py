"""Phone intrinsics and per-frame camera pose from the ChArUco board."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from p2p.board import R_BOARD_TO_TABLE, make_board

MIN_CORNERS = 6


@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]  # (width, height)
    rms_px: float = float("nan")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "K": self.K.tolist(), "dist": self.dist.ravel().tolist(),
            "size": list(self.size), "rms_px": self.rms_px,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Intrinsics":
        d = json.loads(path.read_text())
        return cls(np.array(d["K"]), np.array(d["dist"]), tuple(d["size"]), d["rms_px"])


class BoardDetector:
    def __init__(self, square_m: float):
        self.board = make_board(square_m)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect(self, gray: np.ndarray):
        """Returns (object_points Nx3, image_points Nx2) or None."""
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < MIN_CORNERS:
            return None
        obj, img = self.board.matchImagePoints(corners, ids)
        return obj.reshape(-1, 3), img.reshape(-1, 2)


def calibrate(frames, square_m: float, max_views: int = 60) -> Intrinsics:
    """Intrinsics from a sequence of grayscale frames showing the board."""
    det = BoardDetector(square_m)
    views = [v for v in (det.detect(f) for f in frames) if v is not None]
    if len(views) < 8:
        raise RuntimeError(f"only {len(views)} usable calibration views")
    # Spread the chosen views over the clip instead of taking the first ones.
    idx = np.linspace(0, len(views) - 1, min(max_views, len(views))).astype(int)
    obj = [views[i][0].astype(np.float32) for i in idx]
    img = [views[i][1].astype(np.float32) for i in idx]
    h, w = frames[0].shape[:2]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, (w, h), None, None)
    return Intrinsics(K, dist, (w, h), float(rms))


def table_to_camera(obj: np.ndarray, img: np.ndarray, intr: Intrinsics,
                    prev: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """4x4 transform mapping table-frame points to camera-frame points.

    Returns the transform and the RMS reprojection error in pixels. ``prev``
    (the previous frame's result) seeds the iterative solver, which avoids the
    planar pose flip when the board is seen nearly face on.
    """
    if prev is not None:
        Rb = prev[:3, :3] @ R_BOARD_TO_TABLE.T
        rvec0, _ = cv2.Rodrigues(Rb)
        ok, rvec, tvec = cv2.solvePnP(obj, img, intr.K, intr.dist, rvec0,
                                      prev[:3, 3].copy(), useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, intr.K, intr.dist,
                                      flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, intr.K, intr.dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    R_cam_board, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    # board point p_b = R_BOARD_TO_TABLE.T @ p_table (the rotation is its own inverse)
    T[:3, :3] = R_cam_board @ R_BOARD_TO_TABLE.T
    T[:3, 3] = tvec.ravel()
    return T, err
