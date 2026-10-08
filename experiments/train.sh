#!/usr/bin/env bash
# Fine-tune SmolVLA (from lerobot/smolvla_base) on one arm of the comparison.
# Usage:
#   bash experiments/train.sh teleop   # N official teleoperated demos of the task
#   bash experiments/train.sh phone    # the dataset built from my phone demos
# Both arms use the same model, steps, batch size and seed.
set -euo pipefail
ARM="$1"
N="${N:-13}"
STEPS="${STEPS:-4000}"
OUT="${OUT:-$HOME/runs}/${ARM}${N}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
REV=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4

case "$ARM" in
  teleop)
    EPS=$(python "$REPO/scripts/libero_episodes.py" "$N" --seed 0 | tail -1)
    DATA=(--dataset.repo_id=lerobot/libero --dataset.revision=$REV --dataset.episodes="$EPS") ;;
  phone)
    DATA=(--dataset.repo_id=local/phone_all --dataset.root="$REPO/outputs/datasets/phone_all") ;;
  *) echo "arm must be teleop or phone"; exit 1 ;;
esac

rm -rf "$OUT"
lerobot-train --policy.path=lerobot/smolvla_base --policy.push_to_hub=false \
  "${DATA[@]}" --dataset.video_backend=pyav \
  --rename_map="$RENAME" --policy.empty_cameras=1 \
  --output_dir="$OUT" --steps="$STEPS" --batch_size=32 --save_freq=1000 --log_freq=100 \
  --seed=0 --num_workers=4 --wandb.enable=false
