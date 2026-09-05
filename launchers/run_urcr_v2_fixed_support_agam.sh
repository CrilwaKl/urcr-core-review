#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-}"
SEARCH_URL="${2:-http://127.0.0.1:18268/retrieve}"
PHYSICAL_GPUS="${URCR_V2_GPUS:-0,1,6,7}"
RUN_INSTANCE="${URCR_V2_RUN_INSTANCE:-segment_000}"
MIN_FREE_MB="${URCR_V2_MIN_FREE_MB:-50000}"
BASE_QUERY_REWARD="${URCR_V2_BQ:-0.5}"
RESUME_FROM="${URCR_V2_RESUME_FROM:-}"
WORKSPACE_ROOT="/data2/kongmu/agentic-RL"
REPO_ROOT="$WORKSPACE_ROOT/EviSD-URCR"
CONFIG_ROOT="$REPO_ROOT/examples/urcr_online/configs"
CONFIG_NAME="urcr_v2_fixed_support_agam"
MODEL_PATH="/data0/kongmu/agentic-RL/models/Qwen2.5-3B-Instruct"
OUTPUT_ROOT="/data2/kongmu/agentic-RL-runtime/urcr_v2_fixed_support_agam/formal_4gpu"
ROLLING_ROOT="/data0/kongmu/agentic-RL/runtime/urcr_v2_fixed_support_agam/rolling_checkpoints"
CACHE_DIR="/data0/kongmu/agentic-RL/cache/verl_rlhf"
IFS=',' read -r -a gpu_ids <<<"$PHYSICAL_GPUS"

case "$PHASE" in
  one_step)
    case "$BASE_QUERY_REWARD" in
      0.5) BQ_TAG="bq0p5" ;;
      0.25) BQ_TAG="bq0p25" ;;
      *) echo "one-step gate supports URCR_V2_BQ=0.5 or 0.25" >&2; exit 2 ;;
    esac
    OUTPUT_DIR="$OUTPUT_ROOT/engineering_gate_${#gpu_ids[@]}gpu_$BQ_TAG"
    RAY_TMPDIR="/data0/kongmu/r_v2g"
    TASK_TMPDIR="/data0/kongmu/t_v2g"
    RUN_NAME="urcr_v2_fixed_support_agam_3b_${BQ_TAG}_one_step_gate"
    PHASE_ARGS=(
      trainer.stop_after_steps=1
      trainer.save_freq=-1
      trainer.checkpoint_rotation.enable=False
    )
    ;;
  formal)
    OUTPUT_DIR="$OUTPUT_ROOT/$RUN_INSTANCE"
    RAY_TMPDIR="/data2/kongmu/r_v2f"
    TASK_TMPDIR="/data2/kongmu/t_v2f"
    if [[ -n "$RESUME_FROM" ]]; then
      RUN_NAME="urcr_v2_fixed_support_agam_3b_resume_$(basename "$RESUME_FROM")"
      PHASE_ARGS=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="$RESUME_FROM"
        trainer.require_complete_checkpoint=True
      )
    else
      RUN_NAME="urcr_v2_fixed_support_agam_3b_from_base_300step"
      PHASE_ARGS=(
        trainer.resume_mode=disable
        trainer.resume_from_path=null
      )
    fi
    ;;
  *)
    echo "usage: $0 {one_step|formal} [retrieval_url]" >&2
    exit 2
    ;;
esac

if [[ ! "$RUN_INSTANCE" =~ ^segment_[a-z0-9_]+$ ]]; then
  echo "URCR_V2_RUN_INSTANCE must match segment_[a-z0-9_]+" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "incomplete 3B base model: $MODEL_PATH" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite formal output: $OUTPUT_DIR" >&2
  exit 1
fi
if [[ -e "$RAY_TMPDIR" || -e "$TASK_TMPDIR" ]]; then
  echo "refusing to reuse task temp directories" >&2
  exit 1
fi
if [[ "$PHASE" == formal && -d "$ROLLING_ROOT" ]]; then
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    /data2/kongmu/miniforge3/envs/evaluation/bin/python - "$ROLLING_ROOT" <<'PY'
from pathlib import Path
import sys
from verl.trainer.ppo.plan06_checkpointing import require_complete_checkpoint

root = Path(sys.argv[1])
checkpoints = sorted(root.glob("global_step_*"))
if len(checkpoints) > 1:
    raise RuntimeError(f"v2 placeholder pool has multiple checkpoints: {checkpoints}")
if checkpoints:
    checkpoint = checkpoints[0]
    step = int(checkpoint.name.removeprefix("global_step_"))
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text().strip() != str(step):
        raise RuntimeError("v2 placeholder and tracker do not match")
    require_complete_checkpoint(checkpoint, global_step=step, expected_world_size=4)
print("URCR_V2_PLACEHOLDER_POOL_EXACT")
PY
fi

set +u
source /data2/kongmu/miniforge3/etc/profile.d/conda.sh
conda activate evaluation
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - "$CONFIG_ROOT" "$CONFIG_NAME" "$BASE_QUERY_REWARD" <<'PY'
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.urcr_sources import validate_plan05_config

with initialize_config_dir(config_dir=sys.argv[1], version_base=None):
    cfg = compose(
        config_name=sys.argv[2],
        overrides=[f"algorithm.urcr.support_reward.base_query_reward={sys.argv[3]}"],
    )
c = OmegaConf.to_container(cfg, resolve=True)
u = validate_plan05_config(c["algorithm"]["urcr"])
actor = c["actor_rollout_ref"]["actor"]
rollout = c["actor_rollout_ref"]["rollout"]
assert u.uses_v2_fixed_support_reward
assert u.support_reward.utility_mode == "binary_hierarchical"
assert u.support_reward.base_query_reward == float(sys.argv[3])
assert u.support_reward.doc_only_utility == 0.5
assert u.local_objective.think_length_ref == 22.0
assert u.local_objective.local_max == 0.01
assert u.local_objective.warmup_steps == 30
assert c["data"]["train_batch_size"] == 128
assert c["data"]["shuffle"] is True and c["data"]["seed"] == 1
assert c["env"]["rollout"]["n"] == 8 and c["env"]["max_steps"] == 4
assert actor["ppo_mini_batch_size"] == 256
assert actor["ppo_micro_batch_size_per_gpu"] == 16
assert actor["optim"]["lr"] == 1e-6
assert actor["optim"]["lr_warmup_steps_ratio"] == 0.1
assert actor["use_kl_loss"] and actor["kl_loss_coef"] == 0.001
assert rollout["log_prob_micro_batch_size_per_gpu"] == 32
assert rollout["tensor_model_parallel_size"] == 1
assert "seed" not in rollout
assert c["algorithm"]["urcr"]["answer_agam"] == {"enable": True, "lambda": 0.1}
assert not c["algorithm"]["evisd"]["enable"]
assert not c["algorithm"]["evisd"]["answer_enable"]
assert c["trainer"]["total_training_steps"] == 300
assert c["trainer"]["save_freq"] == 25
print("URCR_V2_CONFIG_EXACT")
PY

source "$REPO_ROOT/examples/urcr_online/worktree_guard.sh"
urcr_assert_worktree_safe \
  "URCR-v2" \
  "$WORKSPACE_ROOT" \
  1 \
  "EviSD-URCR/examples/urcr_online/run_urcr_v2_fixed_support_agam.sh" \
  "EviSD-URCR/examples/urcr_online/run_urcr_v2_fixed_support_agam.sh" \
  "EviSD-URCR/examples/urcr_online/configs/urcr_v2_fixed_support_agam.yaml" \
  "EviSD-URCR/verl/trainer/ppo/evisd_ray_trainer.py" \
  "EviSD-URCR/verl/trainer/ppo/urcr_local_objective.py" \
  "EviSD-URCR/verl/trainer/ppo/urcr_sources.py" \
  "EviSD-URCR/data_reference.md"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
if ! env -u LD_LIBRARY_PATH /usr/bin/curl --fail --silent --max-time 15 \
  -H 'Content-Type: application/json' \
  -d '{"queries":["URCR v2 retriever health check"],"topk":1,"return_scores":false}' \
  "$SEARCH_URL" >/dev/null; then
  echo "retriever health check failed: $SEARCH_URL" >&2
  exit 1
fi

if [[ "$PHASE" == formal && ${#gpu_ids[@]} -ne 4 ]]; then
  echo "URCR v2 formal run requires exactly four GPU IDs" >&2
  exit 2
fi
if [[ "$PHASE" == one_step && ${#gpu_ids[@]} -ne 2 && ${#gpu_ids[@]} -ne 4 ]]; then
  echo "URCR v2 one-step gate requires two or four GPU IDs" >&2
  exit 2
fi
if [[ "$PHASE" == formal ]]; then
  required_available_kb=$((155 * 1024 * 1024))
  stable_memory_checks=0
  echo "waiting for MemAvailable >= 155 GiB for 60 continuous seconds"
  while (( stable_memory_checks < 20 )); do
    available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( available_kb >= required_available_kb )); then
      stable_memory_checks=$((stable_memory_checks + 1))
    else
      stable_memory_checks=0
    fi
    if (( stable_memory_checks < 20 )); then
      sleep 3
    fi
  done
  echo "host-memory stability gate passed"
fi
if ! nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
  --id="$PHYSICAL_GPUS" \
  | awk -v minimum="$MIN_FREE_MB" -v expected="${#gpu_ids[@]}" \
      '{count += 1; if ($1 < minimum) low = 1} END {exit(count != expected || low)}'; then
  echo "URCR v2 requires at least ${MIN_FREE_MB} MiB free on each selected GPU" >&2
  exit 1
fi

if [[ "$PHASE" == formal ]]; then
  rolling_parent="$ROLLING_ROOT"
  while [[ ! -e "$rolling_parent" ]]; do
    rolling_parent="$(dirname "$rolling_parent")"
  done
  rolling_free_kb="$(df -Pk "$rolling_parent" | awk 'NR==2 {print $4}')"
  rolling_reclaimable_kb=0
  for checkpoint in "$ROLLING_ROOT"/global_step_*; do
    if [[ -d "$checkpoint" ]]; then
      rolling_reclaimable_kb="$(du -sk "$checkpoint" | awk '{print $1}')"
    fi
  done
  if (( rolling_free_kb + rolling_reclaimable_kb < 42 * 1024 * 1024 )); then
    echo "URCR v2 checkpoint filesystem needs at least 42 GiB free plus replaceable checkpoint capacity" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$RAY_TMPDIR" "$TASK_TMPDIR"
if [[ "$PHASE" == formal ]]; then
  mkdir -p "$ROLLING_ROOT"
fi
cleanup_task_tmp() {
  rm -rf -- "$RAY_TMPDIR" "$TASK_TMPDIR"
}
trap cleanup_task_tmp EXIT

git -C "$WORKSPACE_ROOT" rev-parse HEAD >"$OUTPUT_DIR/launch_git_head.txt"
git -C "$WORKSPACE_ROOT" status --short --untracked-files=all >"$OUTPUT_DIR/launch_git_status.txt"
git -C "$WORKSPACE_ROOT" diff --no-ext-diff >"$OUTPUT_DIR/launch_git_diff.patch"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS"
export RAY_TMPDIR
export TMPDIR="$TASK_TMPDIR"
export TMP="$TASK_TMPDIR"
export TEMP="$TASK_TMPDIR"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$REPO_ROOT"
python -m verl.trainer.main_evisd \
  --config-path "$CONFIG_ROOT" \
  --config-name "$CONFIG_NAME" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.tokenizer_path="$MODEL_PATH" \
  reward_model.model.input_tokenizer="$MODEL_PATH" \
  env.search.search_url="$SEARCH_URL" \
  trainer.n_gpus_per_node="${#gpu_ids[@]}" \
  trainer.project_name=urcr_v2_fixed_support_agam \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$ROLLING_ROOT" \
  algorithm.urcr.support_reward.base_query_reward="$BASE_QUERY_REWARD" \
  algorithm.urcr.audit.output_dir="$OUTPUT_DIR" \
  "${PHASE_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
