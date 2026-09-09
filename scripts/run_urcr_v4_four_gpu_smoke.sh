#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 {scorer|pipeline|calibration|population|gradient|base-shadow}" >&2
  exit 2
fi

SMOKE_KIND="$1"

REPO_ROOT="/data2/kongmu/agentic-RL/EviSD-URCR"
WORKSPACE_ROOT="/data2/kongmu/agentic-RL"
CONFIG_ROOT="$REPO_ROOT/examples/urcr_online/configs"
CONFIG_NAME="plan07_answer_agam_core"
MODEL="/data1/kongmu/model_checkpoints/base_grpo_3b_s150_hf"
TRAIN_DATA="/data0/kongmu/agentic-RL/data/urcr_v4/s150_diagnostic_seed20260908_128.parquet"
VAL_DATA="$TRAIN_DATA"
DATA_SHUFFLE=True
STOP_AFTER_STEPS=null
POPULATION_MANIFEST=null
POPULATION_OUTPUT=null
POPULATION_START_BATCH=2
POPULATION_END_BATCH=10
POOLED_CALIBRATION=null
GRADIENT_CALIBRATION=null
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
ROLLOUT_ENABLE_CHUNKED_PREFILL=True
ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
SEARCH_URL="http://127.0.0.1:18268/retrieve"
PHYSICAL_GPUS="0,1,6,7"
case "$SMOKE_KIND" in
  scorer)
    V4_MODE="scorer_smoke"
    OUTPUT_DIR="$REPO_ROOT/logs/urcr_v4/four_gpu_fsdp_scorer_smoke"
    ARTIFACT="$REPO_ROOT/artifacts/urcr_v4/s150_four_gpu_fsdp_scorer_smoke.json"
    EXPERIMENT_NAME="urcr_v4_s150_four_gpu_fsdp_scorer_smoke"
    TMP_SUFFIX="v4s150"
    TRAIN_BATCH_SIZE=4
    ;;
  pipeline)
    V4_MODE="pipeline_smoke"
    OUTPUT_DIR="$REPO_ROOT/logs/urcr_v4/four_gpu_pipeline_smoke"
    ARTIFACT="$REPO_ROOT/artifacts/urcr_v4/s150_four_gpu_pipeline_smoke.json"
    EXPERIMENT_NAME="urcr_v4_s150_four_gpu_pipeline_smoke"
    TMP_SUFFIX="v4p150"
    TRAIN_BATCH_SIZE=4
    ;;
  calibration)
    V4_MODE="calibration"
    OUTPUT_DIR="$REPO_ROOT/logs/urcr_v4/s150_calibration"
    ARTIFACT="$REPO_ROOT/artifacts/urcr_v4/s150_calibration.json"
    EXPERIMENT_NAME="urcr_v4_s150_calibration"
    TMP_SUFFIX="v4c150"
    TRAIN_BATCH_SIZE=128
    ;;
  population)
    V4_MODE="calibration_population"
    OUTPUT_DIR="$REPO_ROOT/logs/urcr_v4/s150_calibration_population"
    ARTIFACT="$REPO_ROOT/artifacts/urcr_v4/calibration_population"
    POPULATION_MANIFEST="$ARTIFACT/population_manifest.json"
    POPULATION_OUTPUT="$ARTIFACT/pooled_calibration_10_batches.json"
    EXPERIMENT_NAME="urcr_v4_s150_calibration_population"
    TMP_SUFFIX="v4cpop"
    TRAIN_BATCH_SIZE=128
    DATA_SHUFFLE=False
    while (( POPULATION_START_BATCH <= POPULATION_END_BATCH )); do
      completed_artifact="$ARTIFACT/calibration_batch_$(printf '%02d' "$POPULATION_START_BATCH").json"
      [[ -f "$completed_artifact" ]] || break
      ((POPULATION_START_BATCH += 1))
    done
    if (( POPULATION_START_BATCH > POPULATION_END_BATCH )); then
      echo "all calibration population batches already exist" >&2
      exit 1
    fi
    STOP_AFTER_STEPS=$((POPULATION_END_BATCH - POPULATION_START_BATCH + 1))
    TRAIN_DATA="["
    for batch_index in $(seq "$POPULATION_START_BATCH" "$POPULATION_END_BATCH"); do
      batch_path="/data0/kongmu/agentic-RL/data/urcr_v4/calibration_population/calibration_batch_$(printf '%02d' "$batch_index").parquet"
      if [[ "$batch_index" -gt "$POPULATION_START_BATCH" ]]; then
        TRAIN_DATA+=","
      fi
      TRAIN_DATA+="$batch_path"
    done
    TRAIN_DATA+="]"
    VAL_DATA="/data0/kongmu/agentic-RL/data/urcr_v4/calibration_population/calibration_batch_$(printf '%02d' "$POPULATION_START_BATCH").parquet"
    ;;
  gradient)
    V4_MODE="gradient_calibration"
    OUTPUT_DIR="$REPO_ROOT/logs/urcr_v4/s150_gradient_calibration"
    ARTIFACT="$REPO_ROOT/artifacts/urcr_v4/s150_gradient_calibration.json"
    POOLED_CALIBRATION="$REPO_ROOT/artifacts/urcr_v4/calibration_population/pooled_calibration_10_batches.json"
    EXPERIMENT_NAME="urcr_v4_s150_gradient_calibration"
    TMP_SUFFIX="v4g150"
    TRAIN_BATCH_SIZE=128
    STOP_AFTER_STEPS=1
    ACTOR_PARAM_OFFLOAD=False
    ACTOR_OPTIMIZER_OFFLOAD=False
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.5
    ROLLOUT_ENABLE_CHUNKED_PREFILL=False
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
    ;;
  base-shadow)
    V4_MODE="base_shadow"
    MODEL="/data0/kongmu/agentic-RL/models/Qwen2.5-3B-Instruct"
    OUTPUT_DIR="$REPO_ROOT/logs/urcr_v4/base_shadow_gradient_calibration"
    ARTIFACT="$REPO_ROOT/artifacts/urcr_v4/base_shadow_gradient_calibration.json"
    POOLED_CALIBRATION="$REPO_ROOT/artifacts/urcr_v4/calibration_population/pooled_calibration_10_batches.json"
    GRADIENT_CALIBRATION="$REPO_ROOT/artifacts/urcr_v4/s150_gradient_calibration.json"
    EXPERIMENT_NAME="urcr_v4_base_shadow_gradient_calibration"
    TMP_SUFFIX="v4gbase"
    TRAIN_BATCH_SIZE=128
    STOP_AFTER_STEPS=1
    ACTOR_PARAM_OFFLOAD=False
    ACTOR_OPTIMIZER_OFFLOAD=False
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.5
    ROLLOUT_ENABLE_CHUNKED_PREFILL=False
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
    ;;
  *)
    echo "usage: $0 {scorer|pipeline|calibration|population|gradient|base-shadow}" >&2
    exit 2
    ;;
esac
LOG_FILE="$OUTPUT_DIR/run.log"
RAY_TMPDIR="/data0/kongmu/r_$TMP_SUFFIX"
TASK_TMPDIR="/data0/kongmu/t_$TMP_SUFFIX"
CACHE_DIR="/data0/kongmu/agentic-RL/cache/verl_rlhf"

for path in "$MODEL" "$VAL_DATA" "$CONFIG_ROOT/$CONFIG_NAME.yaml"; do
  if [[ ! -e "$path" ]]; then
    echo "missing URCR-V4 $SMOKE_KIND smoke input: $path" >&2
    exit 1
  fi
done
if [[ "$SMOKE_KIND" == population ]]; then
  for batch_index in $(seq 2 "$POPULATION_END_BATCH"); do
    batch_path="/data0/kongmu/agentic-RL/data/urcr_v4/calibration_population/calibration_batch_$(printf '%02d' "$batch_index").parquet"
    if [[ ! -f "$batch_path" ]]; then
      echo "missing URCR-V4 calibration population input: $batch_path" >&2
      exit 1
    fi
    batch_artifact="$ARTIFACT/calibration_batch_$(printf '%02d' "$batch_index").json"
    if (( batch_index >= POPULATION_START_BATCH )) && [[ -e "$batch_artifact" ]]; then
      echo "refusing to overwrite calibration population artifact: $batch_artifact" >&2
      exit 1
    fi
  done
  if [[ ! -f "$POPULATION_MANIFEST" ]]; then
    echo "missing URCR-V4 calibration population manifest: $POPULATION_MANIFEST" >&2
    exit 1
  fi
  if [[ -e "$POPULATION_OUTPUT" ]]; then
    echo "refusing to overwrite pooled calibration: $POPULATION_OUTPUT" >&2
    exit 1
  fi
elif [[ "$SMOKE_KIND" == gradient || "$SMOKE_KIND" == base-shadow ]]; then
  if [[ ! -f "$POOLED_CALIBRATION" ]]; then
    echo "missing pooled calibration: $POOLED_CALIBRATION" >&2
    exit 1
  fi
  if [[ "$SMOKE_KIND" == base-shadow && ! -f "$GRADIENT_CALIBRATION" ]]; then
    echo "missing S150 gradient calibration: $GRADIENT_CALIBRATION" >&2
    exit 1
  fi
  if [[ -e "$ARTIFACT" ]]; then
    echo "refusing to overwrite URCR-V4 gradient artifact: $ARTIFACT" >&2
    exit 1
  fi
elif [[ -e "$ARTIFACT" ]]; then
  echo "refusing to overwrite URCR-V4 $SMOKE_KIND smoke artifact: $ARTIFACT" >&2
  exit 1
fi
for path in "$RAY_TMPDIR" "$TASK_TMPDIR"; do
  if [[ -e "$path" ]]; then
    echo "refusing to reuse stale URCR-V4 $SMOKE_KIND temporary path: $path" >&2
    exit 1
  fi
done

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy RAY_ADDRESS
export RAY_memory_usage_threshold=0.99

gpu_free="$({ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i "$PHYSICAL_GPUS"; } | awk -F, '{gsub(/ /,"",$2); if ($2 < 40000) bad=1} END {print bad+0}')"
if [[ "$gpu_free" != 0 ]]; then
  echo "URCR-V4 $SMOKE_KIND smoke requires at least 40000 MiB free on GPUs $PHYSICAL_GPUS" >&2
  exit 1
fi
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kib < 86 * 1024 * 1024 )); then
  echo "URCR-V4 $SMOKE_KIND smoke requires at least 86 GiB available host memory" >&2
  exit 1
fi
curl --max-time 5 -fsS -o /dev/null \
  -H 'Content-Type: application/json' \
  -d "{\"queries\":[\"URCR V4 $SMOKE_KIND smoke health check\"],\"topk\":1,\"return_scores\":false}" \
  "$SEARCH_URL"

set +u
source /data2/kongmu/miniforge3/etc/profile.d/conda.sh
conda activate evaluation
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS"
export RAY_TMPDIR
export TMPDIR="$TASK_TMPDIR"
export TMP="$TASK_TMPDIR"
export TEMP="$TASK_TMPDIR"
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"

mkdir -p "$OUTPUT_DIR" "$RAY_TMPDIR" "$TASK_TMPDIR" \
  "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR"
cleanup() {
  rm -rf -- "$RAY_TMPDIR" "$TASK_TMPDIR"
}
trap cleanup EXIT INT TERM

git -C "$WORKSPACE_ROOT" rev-parse HEAD >"$OUTPUT_DIR/git_head.txt"
git -C "$WORKSPACE_ROOT" status --short --untracked-files=all >"$OUTPUT_DIR/git_status.txt"
git -C "$WORKSPACE_ROOT" diff --no-ext-diff | sha256sum >"$OUTPUT_DIR/git_diff_sha256.txt"
if [[ -f "$MODEL/SHA256SUMS" ]]; then
  (
    cd "$MODEL"
    sha256sum -c --quiet SHA256SUMS
  )
else
  sha256sum "$MODEL"/config.json "$MODEL"/model.safetensors.index.json \
    "$MODEL"/model-*.safetensors >"$OUTPUT_DIR/model_sha256.txt"
fi
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4'

cd "$REPO_ROOT"
python -m verl.trainer.main_evisd \
  --config-path "$CONFIG_ROOT" \
  --config-name "$CONFIG_NAME" \
  actor_rollout_ref.model.path="$MODEL" \
  critic.model.tokenizer_path="$MODEL" \
  reward_model.model.input_tokenizer="$MODEL" \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.shuffle="$DATA_SHUFFLE" \
  data.seed=1 \
  env.rollout.n=8 \
  env.seed=0 \
  env.max_steps=4 \
  env.search.topk=3 \
  env.search.timeout=60 \
  env.search.search_url="$SEARCH_URL" \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enable_chunked_prefill="$ROLLOUT_ENABLE_CHUNKED_PREFILL" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  algorithm.evisd.enable=False \
  algorithm.evisd.answer_enable=False \
  algorithm.evisd.search_enable_for_main_loss=False \
  algorithm.evisd.search_shadow_enable=False \
  algorithm.urcr.enable=False \
  algorithm.urcr.source.enable=False \
  algorithm.urcr.source.lambda_a=0.0 \
  algorithm.urcr.routing.lambda_r=0.0 \
  algorithm.urcr.audit.compute_shadow_modes=False \
  algorithm.urcr.audit.compute_q0_shadow=False \
  algorithm.urcr.audit.save_surrogate_coefficients=False \
  algorithm.urcr.audit.save_turn_components=False \
  algorithm.urcr.audit.capture_update_summary=False \
  algorithm.urcr.audit.capture_update_vectors=False \
  algorithm.urcr.audit.save_update_reference=False \
  algorithm.urcr.audit.compare_update_reference=False \
  algorithm.urcr.answer_agam.enable=False \
  algorithm.urcr.answer_agam.lambda=0.0 \
  algorithm.urcr_diagnostics.enable=False \
  ++algorithm.urcr_v4.enable=True \
  ++algorithm.urcr_v4.mode="$V4_MODE" \
  ++algorithm.urcr_v4.diagnostic_output="$ARTIFACT" \
  ++algorithm.urcr_v4.numeric_reference_artifact="$REPO_ROOT/artifacts/urcr_v4/s150_four_gpu_fsdp_scorer_smoke.json" \
  ++algorithm.urcr_v4.pipeline_smoke_search_anchors=16 \
  ++algorithm.urcr_v4.pipeline_smoke_query_responsibility_anchors=8 \
  ++algorithm.urcr_v4.pipeline_smoke_answer_responsibility_anchors=8 \
  ++algorithm.urcr_v4.calibration_search_anchors=256 \
  ++algorithm.urcr_v4.calibration_query_responsibility_anchors=64 \
  ++algorithm.urcr_v4.calibration_answer_responsibility_anchors=64 \
  ++algorithm.urcr_v4.calibration_population_manifest="$POPULATION_MANIFEST" \
  ++algorithm.urcr_v4.calibration_population_start_batch="$POPULATION_START_BATCH" \
  ++algorithm.urcr_v4.calibration_population_end_batch="$POPULATION_END_BATCH" \
  ++algorithm.urcr_v4.pooled_calibration_artifact="$POOLED_CALIBRATION" \
  ++algorithm.urcr_v4.gradient_calibration_artifact="$GRADIENT_CALIBRATION" \
  ++algorithm.urcr_v4.scorer.max_total_tokens=8192 \
  ++algorithm.urcr_v4.scorer.micro_batch_size_per_gpu=32 \
  ++algorithm.urcr_v4.scorer.token_budget_per_gpu=262144 \
  ++algorithm.urcr_v4.scorer.smoke_micro_batch_sizes_per_gpu='[8,16,32]' \
  trainer.project_name=urcr_v4_validation \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$TASK_TMPDIR/checkpoints" \
  trainer.checkpoint_rotation.enable=False \
  trainer.resume_mode=disable \
  trainer.resume_from_path=null \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_training_steps=300 \
  trainer.stop_after_steps="$STOP_AFTER_STEPS" \
  actor_rollout_ref.actor.optim.total_training_steps=300 \
  critic.optim.total_training_steps=300 \
  'trainer.logger=[console]' \
  hydra.run.dir="$OUTPUT_DIR/hydra" \
  2>&1 | tee -a "$LOG_FILE"

if [[ "$SMOKE_KIND" == population ]]; then
  python scripts/pool_urcr_v4_calibration_population.py \
    --population-manifest "$POPULATION_MANIFEST" \
    --output "$POPULATION_OUTPUT"
fi
