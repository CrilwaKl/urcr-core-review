#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <physical_gpu_id>" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
REPO_ROOT="/data2/kongmu/agentic-RL/EviSD-URCR"
EVAL_ENV="/data2/kongmu/miniforge3/envs/evaluation"
PYTHON="$EVAL_ENV/bin/python"
MODEL="/data1/kongmu/model_checkpoints/base_grpo_3b_s150_hf"
OUTPUT="$REPO_ROOT/artifacts/urcr_v4/s150_gpu_scorer_exactness.json"
LOG_DIR="$REPO_ROOT/logs/urcr_v4"
LOG_FILE="$LOG_DIR/s150_gpu_scorer_exactness.log"
TASK_TMPDIR="/data2/kongmu/agentic-RL/.tmp/urcr_v4_s150_gpu_exactness"

for path in "$PYTHON" "$MODEL" "$REPO_ROOT/scripts/run_urcr_v4_gpu_scorer_exactness.py"; do
  if [[ ! -e "$path" ]]; then
    echo "missing URCR-V4 exactness input: $path" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite URCR-V4 exactness artifact: $OUTPUT" >&2
  exit 1
fi
if [[ -e "$TASK_TMPDIR" ]]; then
  echo "refusing to reuse stale URCR-V4 task temporary directory: $TASK_TMPDIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$TASK_TMPDIR"
cleanup() {
  rm -rf -- "$TASK_TMPDIR"
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export CONDA_PREFIX="$EVAL_ENV"
export CUDA_HOME="$EVAL_ENV"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$TASK_TMPDIR"
export TMP="$TASK_TMPDIR"
export TEMP="$TASK_TMPDIR"

cd "$MODEL"
sha256sum -c --quiet SHA256SUMS
cd "$REPO_ROOT"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1'
"$PYTHON" scripts/run_urcr_v4_gpu_scorer_exactness.py \
  --model "$MODEL" \
  --scorer-micro-batch-size 1 \
  --output "$OUTPUT" 2>&1 | tee "$LOG_FILE"
