#!/usr/bin/env bash
# Cloud GPU pipeline: clone repo → 40-word HF dataset → landmarks → train/eval
# every arch.md baseline on the 8 highest-count glosses → write comparison report.
#
# Linux VM (RunPod / Lambda / Lightning / Colab terminal):
#   git clone https://github.com/Vidit-01/isl-lowdata-slr.git
#   cd isl-lowdata-slr
#   bash scripts/run_pipeline_baselines.sh
#
# Optional env:
#   REPO_URL   HF_DATASET  HF_TOKEN  ISL_WORKDIR  MODELS  EPOCHS  DRAWS  TEST_PER_CLASS  N_WORDS  SMOKE=1
#   SKIP_CLONE=1  SKIP_HF_DOWNLOAD=1  SKIP_LANDMARKS=1  SKIP_TRAIN=1
#
# Do not set WORKDIR to the notebook folder (Lightning does that). This script
# always runs from the directory that contains islr/fewshot/train.py.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Vidit-01/isl-lowdata-slr.git}"
HF_DATASET="${HF_DATASET:-vidit031/isl-isolated-40words}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
NUM_FRAMES="${NUM_FRAMES:-30}"
GIT_BRANCH="${GIT_BRANCH:-main}"
DATA_DIR="${ISL_DATA_DIR:-ISL_DATASET_40WORDS}"

echo "=== ISL arch.md baselines pipeline ==="

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found (CPU training will be slow)"
fi

if command -v apt-get >/dev/null 2>&1 && [[ "${SKIP_APT:-0}" != "1" ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update -y
    apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0
  else
    sudo apt-get update -y || true
    sudo apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 || true
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer this checkout. Lightning sets WORKDIR=/home/zeus/content — ignore it.
if [[ -f "$REPO_ROOT/islr/fewshot/train.py" ]]; then
  SKIP_CLONE="${SKIP_CLONE:-1}"
fi

PY_ARGS=(--repo-url "$REPO_URL" --hf-dataset "$HF_DATASET" --num-frames "$NUM_FRAMES" --data-dir "$DATA_DIR")

if [[ -n "${ISL_WORKDIR:-}" ]]; then
  PY_ARGS+=(--workdir "$ISL_WORKDIR")
fi
if [[ -n "${GIT_BRANCH:-}" ]]; then
  PY_ARGS+=(--branch "$GIT_BRANCH")
fi
if [[ "${SKIP_CLONE:-0}" == "1" ]]; then
  PY_ARGS+=(--skip-clone)
fi
if [[ "${SKIP_HF_DOWNLOAD:-0}" == "1" ]]; then
  PY_ARGS+=(--skip-download)
fi
if [[ "${SKIP_LANDMARKS:-0}" == "1" ]]; then
  PY_ARGS+=(--skip-landmarks)
fi
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  PY_ARGS+=(--skip-train)
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  PY_ARGS+=(--smoke)
fi
if [[ -n "${EPOCHS:-}" ]]; then
  PY_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODEL_ARR=($MODELS)
  PY_ARGS+=(--models "${MODEL_ARR[@]}")
fi
if [[ -n "${DRAWS:-}" ]]; then
  PY_ARGS+=(--draws "$DRAWS")
fi
if [[ -n "${TEST_PER_CLASS:-}" ]]; then
  PY_ARGS+=(--test-per-class "$TEST_PER_CLASS")
fi
if [[ -n "${N_WORDS:-}" ]]; then
  PY_ARGS+=(--n-words "$N_WORDS")
fi
if [[ -n "${WORDS:-}" ]]; then
  # shellcheck disable=SC2206
  WORD_ARR=($WORDS)
  PY_ARGS+=(--words "${WORD_ARR[@]}")
fi
if [[ "${STRICT_PROTOCOL:-0}" == "1" ]]; then
  PY_ARGS+=(--strict-protocol)
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_pipeline_baselines.py" "${PY_ARGS[@]}"
