"""README GIFs from LeRobot evaluation videos.

Usage: python scripts/make_gifs.py
Reads ~/runs/<run>/eval_last/videos/... and writes docs/media/*.gif.

Frames are kept at real speed (every second or third 20 Hz control step, 50 ms
per step), at full resolution unless a long clip would make the GIF too heavy
for a README. The evaluation stops recording the moment LIBERO
reports success, so the last frame is held for a while instead of the GIF
cutting off mid-motion.
"""

from pathlib import Path

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path.home() / "runs"

CLIPS = [  # (run, evaluation episode, output name, every, size)
    ("phone13", 2, "policy_phone_success", 2, 360),
    ("phone13", 0, "policy_phone_failure", 3, 300),
    ("teleop13", 0, "policy_teleop_success", 2, 360),
]
STEP_MS = 50  # LIBERO runs at 20 Hz


def save_gif(frames, path: Path, every: int = 2, size: int | None = None, hold_ms: int = 1500) -> None:
    """Adaptive palette per frame, no lossy frame merging, last frame held."""
    imgs = []
    for f in frames[::every]:
        im = Image.fromarray(np.asarray(f))
        if size and im.width != size:
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        imgs.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
    durations = [STEP_MS * every] * (len(imgs) - 1) + [hold_ms]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durations,
                 loop=0, optimize=False, disposal=1)


def main() -> None:
    out = ROOT / "docs" / "media"
    out.mkdir(parents=True, exist_ok=True)
    for run, ep, name, every, size in CLIPS:
        src = RUNS / run / "eval_last" / "videos" / "libero_goal_8" / f"eval_episode_{ep}.mp4"
        with av.open(str(src)) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        save_gif(frames, out / f"{name}.gif", every, size)
        print(f"{name}: {len(frames)} steps, {(out / f'{name}.gif').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
