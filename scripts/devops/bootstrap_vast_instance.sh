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
  python-dotenv \
  pydantic \
  PyYAML \
  psutil

# Qwen3.5 uses gated-delta/linear-attention kernels. Without these packages,
# Unsloth falls back to a very slow Torch implementation that under-utilizes an
# A100 badly. Install them without dependency resolution so pip cannot replace
# the Vast image's working Torch/CUDA stack.
"$PYTHON_BIN" -m pip install --upgrade --no-deps flash-linear-attention einops
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" MAX_JOBS="${MAX_JOBS:-4}" \
  "$PYTHON_BIN" -m pip install --upgrade --no-build-isolation --no-deps causal-conv1d

# Install the repo package into the existing image env. Vast's PyTorch images
# commonly use Python 3.11 even though local development targets Python 3.12.
# Training/eval scripts used on these boxes are Python 3.11-compatible, so avoid
# creating a second Torch env and avoid pulling heavyweight CAD deps.
"$PYTHON_BIN" -m pip install \
  --no-build-isolation \
  --no-deps \
  --ignore-requires-python \
  -e .

# Unsloth is installed after the project deps. Constrain Torch/Torchvision to
# the image's existing major/minor stack so pip does not replace a working CUDA
# runtime while resolving optional Unsloth accelerators.
TORCH_MM="$("$PYTHON_BIN" - <<'PY'
import torch

major, minor, *_ = torch.__version__.split("+", 1)[0].split(".")
print(f"{major}.{minor}")
PY
)"
TORCHVISION_MM="$("$PYTHON_BIN" - <<'PY'
try:
    import torchvision
except Exception:
    print("")
else:
    major, minor, *_ = torchvision.__version__.split("+", 1)[0].split(".")
    print(f"{major}.{minor}")
PY
)"
if [ -n "$TORCHVISION_MM" ]; then
  "$PYTHON_BIN" -m pip install --upgrade \
    "torch~=$TORCH_MM.0" \
    "torchvision~=$TORCHVISION_MM.0" \
    unsloth
else
  "$PYTHON_BIN" -m pip install --upgrade "torch~=$TORCH_MM.0" unsloth
fi

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python", sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "Bootstrap complete: $WORKDIR"
