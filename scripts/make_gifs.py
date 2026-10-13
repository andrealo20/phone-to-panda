"""README GIFs from LeRobot evaluation videos and from one phone clip.

Usage: python scripts/make_gifs.py
Reads ~/runs/<run>/<eval>/videos/... and, for the phone GIF, the clip of
demo_020 with its tracking (scripts/track.py); writes docs/media/*.gif.

Frames are kept at real speed (every second or third 20 Hz control step, 50 ms
per step), at full resolution unless a long clip would make the GIF too heavy
for a README. The evaluation stops recording the moment LIBERO
reports success, so the last frame is held for a while instead of the GIF
cutting off mid-motion.
"""

from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from p2p.camera import Intrinsics
from p2p.hand import INDEX_TIP, THUMB_TIP, pinch_px
from p2p.video import frames as clip_frames

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path.home() / "runs"

CLIPS = [  # (evaluation folder under ~/runs, episode, output name, every, size)
    ("phone_bal16_20k/eval_008000_a10", 0, "policy_phone_success", 2, 360),
    ("phone_bal16_20k/eval_008000_a10", 1, "policy_phone_failure", 3, 300),
    ("teleop13/eval_last", 0, "policy_teleop_success", 2, 360),
]
STEP_MS = 50  # LIBERO runs at 20 Hz
SESSION = ROOT / "data" / "raw" / "s01"
PHONE = ("demo_020", 3, 480)  # clip, every, width; the phone films at 30 fps
PHONE_MS = 1000 / 30
TRAIL = 15  # frames of pinch path drawn; the phone moves, so older pixels are elsewhere on the table


def save_gif(frames, path: Path, every: int = 2, size: int | None = None, hold_ms: int = 1500,
             frame_ms: float = STEP_MS) -> None:
    """Adaptive palette per frame, no lossy frame merging, last frame held."""
    imgs = []
    for f in frames[::every]:
        im = Image.fromarray(np.asarray(f))
        if size and im.width != size:
            im = im.resize((size, round(im.height * size / im.width)), Image.Resampling.LANCZOS)
        imgs.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
    durations = [round(frame_ms * every)] * (len(imgs) - 1) + [hold_ms]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durations,
                 loop=0, optimize=False, disposal=1)


def read(path: Path) -> list:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def phone_frames(clip: str, width: int) -> list:
    """The clip with the board axes, thumb and index tips, and the last half second of pinch path."""
    intr = Intrinsics.load(SESSION / "intrinsics.json")
    d = np.load(ROOT / "data" / "cache" / SESSION.name / f"{clip}.npz")
    pinch = pinch_px(d["hand_px"])
    out = []
    for k, (_, bgr) in enumerate(clip_frames(SESSION / f"{clip}.mp4")):
        T = d["T_cam_table"][k]
        if np.isfinite(T).all():
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            cv2.drawFrameAxes(bgr, intr.K, intr.dist, rvec, T[:3, 3], 0.06, 6)
        recent = pinch[max(0, k - TRAIL): k + 1]
        path = recent[np.isfinite(recent).all(1)].astype(np.int32)
        if len(path) > 1:
            cv2.polylines(bgr, [path.reshape(-1, 1, 2)], False, (31, 103, 217), 8, cv2.LINE_AA)
        for p in d["hand_px"][k][[THUMB_TIP, INDEX_TIP]]:
            if np.isfinite(p).all():
                cv2.circle(bgr, tuple(p.astype(int)), 14, (0, 220, 255), -1, cv2.LINE_AA)
        h = round(bgr.shape[0] * width / bgr.shape[1])
        small = cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    return out


def main() -> None:
    out = ROOT / "docs" / "media"
    out.mkdir(parents=True, exist_ok=True)
    clip, every, width = PHONE
    save_gif(phone_frames(clip, width), out / f"phone_{clip}.gif", every, frame_ms=PHONE_MS)
    print(f"phone_{clip}: {(out / f'phone_{clip}.gif').stat().st_size // 1024} KB")
    for run, ep, name, every, size in CLIPS:
        src = RUNS / run / "videos" / "libero_goal_8" / f"eval_episode_{ep}.mp4"
        frames = read(src)
        save_gif(frames, out / f"{name}.gif", every, size)
        print(f"{name}: {len(frames)} steps, {(out / f'{name}.gif').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
