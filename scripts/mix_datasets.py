"""The teleoperated demos and my replays in one dataset, for the mixed run.

Usage: python scripts/mix_datasets.py [n] [--phone outputs/datasets/phone_all] [--out outputs/datasets/mixed]
Takes the same n teleoperated episodes as `experiments/train.sh teleop` (seed 0)
and every episode of the phone dataset. Frames are decoded with the backend
used in training and stored as lossless images rather than re-encoded video,
so the policy sees exactly the pixels it saw in the two separate runs.
"""

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from libero_episodes import REVISION, pick

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["observation.images.image", "observation.images.image2", "observation.state", "action"]


def image_features(src: Path) -> dict:
    feats = json.loads((src / "meta" / "info.json").read_text())["features"]
    out = {}
    for k in KEYS:
        f = dict(feats[k])
        if f["dtype"] == "video":
            f = {"dtype": "image", "shape": f["shape"], "names": ["height", "width", "channel"]}
        out[k] = f
    return out


def copy(src: LeRobotDataset, dst: LeRobotDataset) -> int:
    """Append every episode of src to dst; returns the number of episodes."""
    n, current = 0, None
    for i in range(len(src)):
        item = src[i]
        ep = int(item["episode_index"])
        if current is not None and ep != current:
            dst.save_episode()
            n += 1
        current = ep
        frame = {k: item[k].numpy() for k in KEYS}
        for k in KEYS[:2]:
            frame[k] = frame[k].transpose(1, 2, 0)  # CHW uint8 -> HWC
        dst.add_frame({**frame, "task": item["task"]})
    dst.save_episode()
    return n + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=13)
    ap.add_argument("--phone", type=Path, default=ROOT / "outputs" / "datasets" / "phone_all")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "datasets" / "mixed")
    args = ap.parse_args()

    teleop = LeRobotDataset("lerobot/libero", revision=REVISION, episodes=pick(args.n),
                            video_backend="pyav", return_uint8=True)
    phone = LeRobotDataset(f"local/{args.phone.name}", root=args.phone,
                           video_backend="pyav", return_uint8=True)
    out = LeRobotDataset.create(repo_id=f"local/{args.out.name}", fps=phone.meta.fps,
                                features=image_features(args.phone), root=args.out,
                                robot_type="panda", use_videos=False)
    n_t = copy(teleop, out)
    n_p = copy(phone, out)
    out.finalize()
    print(f"{n_t} teleoperated + {n_p} phone episodes, {len(teleop)} + {len(phone)} frames -> {args.out}")


if __name__ == "__main__":
    main()
