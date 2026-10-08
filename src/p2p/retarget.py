"""From 2D pinch pixels to a metric pinch trajectory in the table frame.

The phone gives no reliable depth for the hand, so the pinch is placed on a
known plane instead: the camera ray through the pinch pixel is cut with the
horizontal plane at the bowl rim's height. That is exact whenever the fingers
are on the rim, which is when it matters (grasp and release), and only shifts
the path in between.

Grasp and release come from the motion, not from the fingers: the bowl moves
only while it is held, so the carry is the stretch between the two pauses
that are farthest apart.

The height is not measured. The gripper follows a cone around the two contact
points: it rises with horizontal distance from them, up to a ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import savgol_filter

from p2p.hand import pinch_px


@dataclass
class Session:
    square_mm: float
    board_height_mm: float  # board surface above the table (tablet thickness)
    bowl_height_mm: float
    plate_height_mm: float
    pinch_below_rim_mm: float = 10.0


def rays_to_plane(px: np.ndarray, T_cam_table: np.ndarray, K: np.ndarray,
                  dist: np.ndarray, z_board: np.ndarray) -> np.ndarray:
    """Cut the camera ray of each pixel with the plane z = z_board.

    px (N,2), T_cam_table (N,4,4), z_board (N,) in the board frame. Frames
    without a pixel or a camera pose give NaN.
    """
    out = np.full((len(px), 3), np.nan)
    ok = np.isfinite(px).all(1) & np.isfinite(T_cam_table).all((1, 2))
    if not ok.any():
        return out
    und = cv2.undistortPoints(px[ok].reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    d_cam = np.column_stack([und, np.ones(len(und))])
    R = T_cam_table[ok, :3, :3]
    t = T_cam_table[ok, :3, 3]
    o = -np.einsum("nji,nj->ni", R, t)       # camera centre in the board frame
    d = np.einsum("nji,nj->ni", R, d_cam)    # ray direction in the board frame
    lam = (z_board[ok] - o[:, 2]) / d[:, 2]
    out[ok] = o + lam[:, None] * d
    return out


def resample(t: np.ndarray, x: np.ndarray, hz: float,
             smooth_s: float = 0.25) -> tuple[np.ndarray, np.ndarray, float]:
    """Uniform resampling, linear across gaps, then Savitzky-Golay smoothing.

    Returns (times, samples, longest gap in seconds) so that a clip where the
    hand was lost for long can be flagged.
    """
    ok = np.isfinite(x).all(1)
    tu = np.arange(t[ok][0], t[ok][-1], 1.0 / hz)
    xu = np.column_stack([np.interp(tu, t[ok], x[ok, j]) for j in range(x.shape[1])])
    k = max(5, int(round(smooth_s * hz)) | 1)
    if len(tu) > k:
        xu = savgol_filter(xu, k, 2, axis=0)
    return tu, xu, float(np.max(np.diff(t[ok])))


def find_pauses(xy: np.ndarray, hz: float, v_max: float = 0.08,
                min_s: float = 0.25) -> list[tuple[int, int]]:
    """Stretches of at least ``min_s`` where the horizontal speed is below
    ``v_max`` (m/s). Returns (first, last) index pairs."""
    v = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1) * hz]
    pauses, start = [], None
    for i, slow in enumerate(np.r_[v < v_max, False]):
        if slow and start is None:
            start = i
        elif not slow and start is not None:
            if i - start >= min_s * hz:
                pauses.append((start, i - 1))
            start = None
    return pauses


def contacts(xy: np.ndarray, pauses: list[tuple[int, int]],
             same_place: float = 0.04) -> tuple[int, int]:
    """Grasp and release indices.

    Consecutive pauses closer than ``same_place`` are one place (tracking
    noise splits a real pause into pieces, and fingers adjust on the rim).
    The release place is the last one, the grasp place the one before it, so
    a pause before the reach does not count. The grasp is the end of the
    grasp place, the release the start of the release place.
    """
    places: list[list[int]] = []
    for a, b in pauses:
        if places and np.linalg.norm(xy[a:b + 1].mean(0) - xy[places[-1][0]:places[-1][1] + 1].mean(0)) < same_place:
            places[-1][1] = b
        else:
            places.append([a, b])
    if len(places) < 2:
        raise ValueError(f"need a pause at the bowl and one at the plate, found {len(places)} places")
    return places[-2][1], places[-1][0]


def cone_height(xy: np.ndarray, g: int, r: int, slope: float = 3.0,
                ceiling: float = 0.10) -> np.ndarray:
    """Height above the contact planes: ``slope`` times the horizontal distance
    to the nearest contact point (the grasp point before the grasp, the
    release point after the release, either during the carry), capped.

    The slope sets how steeply the gripper comes down onto the rim. At 1
    (45 degrees) the inner finger hit the rim before passing over it; at 3
    the last centimetres are close to vertical.
    """
    dg = np.linalg.norm(xy - xy[g], axis=1)
    dr = np.linalg.norm(xy - xy[r], axis=1)
    d = np.where(np.arange(len(xy)) <= g, dg, np.where(np.arange(len(xy)) >= r, dr, np.minimum(dg, dr)))
    return np.minimum(slope * d, ceiling)


def straight_carry(xy: np.ndarray, g: int, r: int) -> np.ndarray:
    """Replace the measured carry path by a straight line from the grasp point
    to the release point, with a minimum-jerk time profile over the same
    duration as the human carry.

    During the carry the hand holds the bowl well above the rim plane, at a
    height the phone cannot measure, so its rays land beyond the true
    position: on one clip the measured path passed the plate by 25 cm before
    coming back. The two end points are measured at the rim, where the plane
    is right; the path between them is not.
    """
    out = xy.copy()
    s = np.linspace(0.0, 1.0, r - g + 1)
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    out[g:r + 1] = xy[g] + s[:, None] * (xy[r] - xy[g])
    return out


def contact_planes(n: int, g: int, r: int, z_grasp: float, z_release: float) -> np.ndarray:
    """Pinch plane height per sample: the rim before the grasp, the rim of a
    bowl standing on the plate after the release, linear in between."""
    z = np.full(n, z_grasp)
    z[g:r + 1] = np.linspace(z_grasp, z_release, r - g + 1)
    z[r:] = z_release
    return z


# Bowl and plate are never closer than ~20 cm centre to centre in the protocol,
# so a shorter carry means the contacts were read wrong (hand lost at a pause).
MIN_CARRY = 0.12


def clip_trajectory(t, hand_px, T_cam_table, K, dist, cfg: Session, hz: float,
                    slope: float = 3.0, straight: bool = True, replay_height_rays: bool = True) -> dict:
    """The whole per-clip pipeline. Raises ValueError when the contacts cannot
    be read. The keyword switches exist for the ablation in scripts/ablate.py;
    the defaults are the method.

    Pass 1 cuts the rays at the rim plane and finds the contacts. Pass 2 cuts
    each ray at the height the robot will replay (contact plane plus cone): a
    hand lifted 10 cm but cut at rim height lands ~10 cm too far along its ray.
    """
    mm = 1e-3
    board = cfg.board_height_mm * mm
    z_grasp = (cfg.bowl_height_mm - cfg.pinch_below_rim_mm) * mm
    z_release = z_grasp + cfg.plate_height_mm * mm
    px = pinch_px(hand_px)

    def pinch_on(z_table_per_frame):
        P = rays_to_plane(px, T_cam_table, K, dist, z_table_per_frame - board)
        return resample(t, P[:, :2], hz)

    tu, xy, gap = pinch_on(np.full(len(t), z_grasp))
    g, r = contacts(xy, find_pauses(xy, hz))
    if np.linalg.norm(xy[r] - xy[g]) < MIN_CARRY:
        raise ValueError(f"carry of {100 * np.linalg.norm(xy[r] - xy[g]):.0f} cm is too short")
    planes = contact_planes(len(tu), g, r, z_grasp, z_release)
    if replay_height_rays:
        tu, xy, gap = pinch_on(np.interp(t, tu, planes + cone_height(xy, g, r, slope)))
    measured = xy
    if straight:
        xy = straight_carry(xy, g, r)
    z = planes + cone_height(xy, g, r, slope)
    grip = np.zeros(len(tu))
    grip[g:r] = 1.0
    return dict(t=tu, measured=measured, pinch=np.column_stack([xy, z]), grip=grip,
                grasp=g, release=r, gap=gap)
