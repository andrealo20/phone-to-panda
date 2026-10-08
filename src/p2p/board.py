"""The ChArUco board that defines the table frame.

The board is shown on a screen lying flat on the table. Its physical size depends
on the screen, so the square side is measured with a ruler once and stored in
the recording's config (``square_mm``). Everything else about the board is fixed
here so that calibration and tracking always agree on the layout.

Table frame: origin at the board's top-left inner corner as seen on screen, x
along the board width, y along the board height (pointing "down" the screen),
z = x cross y, i.e. into the table. ``table_frame_up`` flips z so that it
points up, which is the convention used everywhere after detection.
"""

import cv2
import numpy as np

SQUARES_X = 7
SQUARES_Y = 5
MARKER_RATIO = 0.72
DICTIONARY = cv2.aruco.DICT_5X5_100


def make_board(square_m: float = 1.0) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), square_m, square_m * MARKER_RATIO, dictionary
    )


def render_board_image(px_per_square: int = 200, margin_px: int = 40) -> np.ndarray:
    board = make_board()
    w = SQUARES_X * px_per_square + 2 * margin_px
    h = SQUARES_Y * px_per_square + 2 * margin_px
    return board.generateImage((w, h), marginSize=margin_px, borderBits=1)


# Rotation from the OpenCV board frame (z into the table) to the table frame
# used downstream (x right, y away from the person, z up). Rotating 180 deg
# about x keeps x, flips y and z.
R_BOARD_TO_TABLE = np.diag([1.0, -1.0, -1.0])
