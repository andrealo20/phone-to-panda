"""Calibration and board pose on synthetic views with known ground truth."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from p2p.board import R_BOARD_TO_TABLE, SQUARES_X, SQUARES_Y, render_board_image
from p2p.camera import BoardDetector, Intrinsics, calibrate, table_to_camera

SQUARE = 0.035
PPS, MARGIN = 120, 60
W, H = 1280, 720
K_TRUE = np.array([[950.0, 0, 645.0], [0, 950.0, 355.0], [0, 0, 1]])
BOARD_IMG = render_board_image(PPS, MARGIN)


def render_view(R_cam_board, t_cam_board):
    """Image of the board seen by a pinhole camera at the given board pose."""
    # board image pixel -> board metres -> camera image
    S = np.array([[SQUARE / PPS, 0, -MARGIN * SQUARE / PPS],
                  [0, SQUARE / PPS, -MARGIN * SQUARE / PPS],
                  [0, 0, 1]])
    Hm = K_TRUE @ np.column_stack([R_cam_board[:, 0], R_cam_board[:, 1], t_cam_board]) @ S
    return cv2.warpPerspective(BOARD_IMG, Hm, (W, H), borderValue=255)


def random_pose(rng):
    centre = np.array([SQUARES_X, SQUARES_Y, 0]) * SQUARE / 2
    tilt = Rotation.from_euler("xyz", [rng.uniform(-40, 40), rng.uniform(-40, 40),
                                       rng.uniform(-30, 30)], degrees=True).as_matrix()
    R = tilt
    dist = rng.uniform(0.35, 0.6)
    t = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03), dist]) - R @ centre
    return R, t


def test_calibration_recovers_focal_length():
    rng = np.random.default_rng(0)
    frames = [render_view(*random_pose(rng)) for _ in range(20)]
    intr = calibrate(frames, SQUARE)
    assert intr.rms_px < 0.5
    np.testing.assert_allclose(intr.K[0, 0], K_TRUE[0, 0], rtol=0.01)
    np.testing.assert_allclose(intr.K[:2, 2], K_TRUE[:2, 2], atol=5)


def test_board_pose_matches_ground_truth():
    rng = np.random.default_rng(1)
    intr = Intrinsics(K_TRUE, np.zeros(5), (W, H))
    det = BoardDetector(SQUARE)
    for _ in range(5):
        R, t = random_pose(rng)
        obj, img = det.detect(render_view(R, t))
        T, err = table_to_camera(obj, img, intr)
        assert err < 0.5
        np.testing.assert_allclose(T[:3, :3], R @ R_BOARD_TO_TABLE, atol=2e-3)
        np.testing.assert_allclose(T[:3, 3], t, atol=1e-3)
        # The seeded iterative solver must agree with the truth just as well.
        T2, _ = table_to_camera(obj, img, intr, prev=T)
        np.testing.assert_allclose(T2[:3, :3], R @ R_BOARD_TO_TABLE, atol=2e-3)
        np.testing.assert_allclose(T2[:3, 3], t, atol=1e-3)


def test_table_z_points_up():
    """A camera above the table looking down sees table +z pointing at it."""
    intr = Intrinsics(K_TRUE, np.zeros(5), (W, H))
    R = np.eye(3)  # camera looking straight at the board (board z away from camera)
    t = np.array([-0.1, -0.07, 0.5])
    obj, img = BoardDetector(SQUARE).detect(render_view(R, t))
    T, _ = table_to_camera(obj, img, intr)
    cam_in_table = -T[:3, :3].T @ T[:3, 3]
    assert cam_in_table[2] > 0.4
