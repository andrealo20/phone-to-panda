#!/usr/bin/env bash
# Evaluate a trained policy on "put the bowl on the plate" (libero_goal, task 8)
# over LIBERO's 50 fixed initial states, the same states for every policy.
# Usage: bash experiments/eval.sh <run_dir> [checkpoint] [n_action_steps]
#   e.g. ~/runs/teleop13 last 10
# By default the policy executes its whole 50-step chunk before predicting
# again; a smaller n_action_steps replans more often (results go to
# eval_<checkpoint>_a<n>). SIZE=256 renders the cameras at the training
# data's 256x256 instead of LeRobot's default 360x360 (suffix _s256).
# DEV_SEED=<n> evaluates on 50 freshly sampled scenes from that seed instead of
# the 50 fixed states (suffix _dev<n>), to choose a checkpoint without looking
# at the final test states.
set -euo pipefail
RUN="$1"
CKPT="${2:-last}"
N="${3:-}"
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
export MUJOCO_GL=egl

OUT="$RUN/eval_$CKPT"
EXTRA=()
SEED=1000
if [ -n "${DEV_SEED:-}" ]; then
  OUT="${OUT}_dev$DEV_SEED"
  SEED="$DEV_SEED"
  EXTRA+=(--env.init_states=false)
fi
if [ -n "$N" ]; then
  OUT="${OUT}_a$N"
  EXTRA+=(--policy.n_action_steps="$N")
fi
if [ -n "${SIZE:-}" ]; then
  OUT="${OUT}_s$SIZE"
  EXTRA+=(--env.observation_height="$SIZE" --env.observation_width="$SIZE")
fi

lerobot-eval --policy.path="$RUN/checkpoints/$CKPT/pretrained_model" "${EXTRA[@]}" \
  --env.type=libero --env.task=libero_goal --env.task_ids='[8]' \
  --eval.n_episodes=50 --eval.batch_size=5 --seed="$SEED" \
  --rename_map="$RENAME" --output_dir="$OUT"
