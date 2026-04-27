#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a Vast PyTorch image for IntersectionQA experiments.
#
# This intentionally uses the image's existing Python environment. Do not run
# `uv sync` or create a second project virtualenv on these images unless the
# base image is known to be broken.

REPO_URL="${REPO_URL:-https://github.com/MRiabov/IntersectionQA.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-/root/IntersectionQA}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git
fi

if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" fetch origin "$BRANCH"
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi

cd "$WORKDIR"

"$PYTHON_BIN" -m pip install --upgrade pip wheel packaging
"$PYTHON_BIN" -m pip install --upgrade \
  hf_transfer \
  huggingface_hub \
  datasets \
  "transformers" \
  "trl>=0.22.0" \
  peft \
  bitsandbytes \
  accelerate \
  python-dotenv

# Install the repo package into the existing image env. This adds the project
# package and lightweight runtime deps while preserving the image's Torch stack.
"$PYTHON_BIN" -m pip install --no-build-isolation -e .

# Unsloth is installed after the project deps because it may pin auxiliary
# training packages. It should not force a full environment clone.
"$PYTHON_BIN" -m pip install --upgrade unsloth

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python", sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "Bootstrap complete: $WORKDIR"

