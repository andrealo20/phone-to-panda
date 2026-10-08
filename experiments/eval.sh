#!/usr/bin/env bash
# Evaluate a trained policy on "put the bowl on the plate" (libero_goal, task 8)
# over LIBERO's 50 fixed initial states, the same states for every policy.
# Usage: bash experiments/eval.sh <run_dir> [checkpoint]   e.g. ~/runs/teleop13 last
set -euo pipefail
RUN="$1"
CKPT="${2:-last}"
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
export MUJOCO_GL=egl

lerobot-eval --policy.path="$RUN/checkpoints/$CKPT/pretrained_model" \
  --env.type=libero --env.task=libero_goal --env.task_ids='[8]' \
  --eval.n_episodes=50 --eval.batch_size=5 --seed=1000 \
  --rename_map="$RENAME" --output_dir="$RUN/eval_$CKPT"
