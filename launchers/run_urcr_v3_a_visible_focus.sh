#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-}"
case "$PHASE" in
  smoke|resume_check) RUN_NAME="${URCR_V3_RUN_NAME:-${PHASE}_001}" ;;
  formal) RUN_NAME=formal ;;
  *) echo "usage: $0 {smoke|resume_check|formal}" >&2; exit 2 ;;
esac
[[ "$RUN_NAME" =~ ^(smoke_[0-9]+|resume_check_[0-9]+|formal)$ ]] || exit 2
REPO_ROOT=/data2/kongmu/agentic-RL/EviSD-URCR
FAMILY=/data0/kongmu/agentic-RL/runtime/urcr_v3_a_visible_focus
RUN_DIR="$FAMILY/$RUN_NAME"
SEARCH_URL=http://127.0.0.1:18268/retrieve
PHYSICAL_GPUS=0,1,6,7
ACTOR_MICRO="${URCR_V3_ACTOR_MICRO:-16}"
TEACHER_MICRO="${URCR_V3_TEACHER_MICRO:-32}"
SMOKE_NAME="${URCR_V3_SMOKE_NAME:-smoke_001}"
RAY_TMPDIR="/data0/kongmu/r_v3_${RUN_NAME}"
TASK_TMPDIR="/data0/kongmu/t_v3_${RUN_NAME}"
CACHE_DIR=/data0/kongmu/agentic-RL/cache/verl_rlhf

[[ ! -e "$RUN_DIR" && ! -e "$RAY_TMPDIR" && ! -e "$TASK_TMPDIR" ]] || {
  echo "refusing to overwrite an existing A1 run or task temporary directory" >&2; exit 1;
}
set +u
source /data2/kongmu/miniforge3/etc/profile.d/conda.sh
conda activate evaluation
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy RAY_ADDRESS
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="$NO_PROXY"
export RAY_TMPDIR TMPDIR="$TASK_TMPDIR" TMP="$TASK_TMPDIR" TEMP="$TASK_TMPDIR"
export XDG_CACHE_HOME="$CACHE_DIR/xdg" VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor" TRITON_CACHE_DIR="$CACHE_DIR/triton"

# Reuse the V2 launcher resource floors. No batch/sequence/offload changes here.
nvidia-smi --query-gpu=memory.free --id="$PHYSICAL_GPUS" --format=csv,noheader,nounits |
  awk '{count++; if ($1 < 50000) low=1} END {exit(count != 4 || low)}'
awk '/MemAvailable:/ {exit($2 < 155 * 1024 * 1024)}' /proc/meminfo
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==4; print(torch.__version__, torch.version.cuda); print([torch.ones(1,device=f"cuda:{i}").item() for i in range(4)])'
env -u LD_LIBRARY_PATH /usr/bin/curl --fail --silent --max-time 15 \
  -H 'Content-Type: application/json' \
  -d '{"queries":["URCR V3 retriever health check"],"topk":1,"return_scores":false}' \
  "$SEARCH_URL" >/dev/null

cd "$REPO_ROOT"
python scripts/urcr_v3_a_verify.py prepare --phase "$PHASE" --run-dir "$RUN_DIR" \
  --actor-micro "$ACTOR_MICRO" --teacher-micro "$TEACHER_MICRO" --smoke-name "$SMOKE_NAME"
mkdir -p "$RAY_TMPDIR" "$TASK_TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv >"$RUN_DIR/gpu_inventory.txt"
printf 'env URCR_V3_RUN_NAME=%q URCR_V3_ACTOR_MICRO=%q URCR_V3_TEACHER_MICRO=%q URCR_V3_SMOKE_NAME=%q bash %q %q\n' \
  "$RUN_NAME" "$ACTOR_MICRO" "$TEACHER_MICRO" "$SMOKE_NAME" "$REPO_ROOT/examples/urcr_online/run_urcr_v3_a_visible_focus.sh" "$PHASE" >"$RUN_DIR/launch_command.txt"
printf 'launcher_pid=%s\ntmux_pane=%s\nstarted=%s\n' "$$" "${TMUX_PANE:-none}" "$(date -Iseconds)" >"$RUN_DIR/launch_receipt.txt"
python -m verl.trainer.main_evisd --config-path "$RUN_DIR" --config-name launch_config \
  hydra.run.dir="$RUN_DIR/hydra" 2>&1 | tee -a "$RUN_DIR/train.log"

# User update 2026-09-06: rolling every 25 only; no S200 milestone or auto-export.
# The same four GPUs evaluate the final S300 after training ends.
if [[ "$PHASE" == formal ]]; then
  CUDA_VISIBLE_DEVICES= python scripts/urcr_v3_a_verify.py export --run-dir "$RUN_DIR" --step 300 \
    2>&1 | tee -a "$RUN_DIR/post_training.log"
  URCR_EVAL_GPUS="$PHYSICAL_GPUS" bash examples/urcr_online/run_plan06_evaluation_3b.sh urcr_v3_s300 "$SEARCH_URL"
  CUDA_VISIBLE_DEVICES= python scripts/urcr_v3_a_verify.py evaluate_summary --run-dir "$RUN_DIR" --step 300 \
    2>&1 | tee -a "$RUN_DIR/post_training.log"
fi
