#!/usr/bin/env bash
# Builds the Python environment (Linux only: LIBERO does not run on Windows).
# Usage: bash scripts/setup_env.sh [venv_dir]
# System packages: libegl1 (headless MuJoCo rendering) and libgles2 (MediaPipe),
#   sudo apt install libegl1 libgles2
set -euo pipefail

VENV="${1:-$HOME/venvs/phone-to-panda}"
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/src/lerobot}"
LEROBOT_REF="${LEROBOT_REF:-main}"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip uv

if [ ! -d "$LEROBOT_DIR" ]; then
    git clone -q https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
git -C "$LEROBOT_DIR" fetch -q origin
git -C "$LEROBOT_DIR" checkout -q "$LEROBOT_REF"
echo "lerobot at $(git -C "$LEROBOT_DIR" rev-parse HEAD)"

uv pip install -e "$LEROBOT_DIR[smolvla,libero]"
uv pip install av "mediapipe==1.0.1" scipy matplotlib pytest
# lerobot and mediapipe pull different OpenCV builds into the same cv2 folder;
# keep exactly one version of each.
uv pip uninstall opencv-contrib-python opencv-python opencv-python-headless
uv pip install opencv-python-headless==4.13.0.92
uv pip install opencv-python==4.13.0.92
uv pip install "numpy>=2,<2.3"  # the OpenCV wheels pull a newer numpy than lerobot accepts
uv pip check

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
uv pip install -e "$REPO_DIR"

python - <<'EOF'
import torch, mujoco, cv2, mediapipe
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mujoco", mujoco.__version__, "opencv", cv2.__version__, "mediapipe", mediapipe.__version__)
EOF
