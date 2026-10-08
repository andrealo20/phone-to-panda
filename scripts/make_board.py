"""Writes the ChArUco board as a PNG and as a full-screen HTML page.

Open board.html on a tablet (or any screen that can lie flat), go full screen,
then measure the side of one checker square with a ruler. That number goes in
the recording config as square_mm.
"""

import base64
from pathlib import Path

import cv2

from p2p.board import SQUARES_X, SQUARES_Y, render_board_image

OUT = Path(__file__).resolve().parents[1] / "docs" / "board"

HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChArUco board</title>
<style>
html,body{{margin:0;height:100%;background:#fff}}
img{{display:block;width:100vw;height:100vh;object-fit:contain;image-rendering:pixelated}}
</style></head>
<body><img src="data:image/png;base64,{b64}" alt="ChArUco board {sx}x{sy}"></body></html>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = render_board_image()
    cv2.imwrite(str(OUT / "board.png"), img)
    ok, png = cv2.imencode(".png", img)
    assert ok
    b64 = base64.b64encode(png.tobytes()).decode()
    (OUT / "board.html").write_text(HTML.format(b64=b64, sx=SQUARES_X, sy=SQUARES_Y))
    print(f"wrote {OUT / 'board.png'} and {OUT / 'board.html'}")


if __name__ == "__main__":
    main()
