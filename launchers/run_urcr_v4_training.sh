#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-}"
SEARCH_URL="${2:-http://127.0.0.1:18268/retrieve}"
PHYSICAL_GPUS="0,1,6,7"
EXPECTED_WORLD_SIZE=4

REPO_ROOT="/data2/kongmu/agentic-RL/EviSD-URCR"
WORKSPACE_ROOT="/data2/kongmu/agentic-RL"
PYTHON="/data2/kongmu/miniforge3/envs/evaluation/bin/python"
MODEL="/data0/kongmu/agentic-RL/models/Qwen2.5-3B-Instruct"
TRAIN_DATA="/data0/kongmu/agentic-RL/data/searchR1_processed_direct/train.parquet"
VALIDATION_DATA="/data0/kongmu/agentic-RL/data/searchR1_processed_direct/test.parquet"
RESOLVED_METHOD="$REPO_ROOT/artifacts/urcr_v4/resolved_method.json"
RESOLVED_SIDECAR="$REPO_ROOT/artifacts/urcr_v4/resolved_method.sha256"
OUTPUT_ROOT="$REPO_ROOT/logs/urcr_v4/training"
SMOKE_CHECKPOINT_ROOT="/data0/kongmu/agentic-RL/runtime/urcr_v4/smoke_checkpoints"
FORMAL_CHECKPOINT_ROOT="/data0/kongmu/agentic-RL/runtime/urcr_v4/rolling_checkpoints"
CACHE_DIR="/data0/kongmu/agentic-RL/cache/verl_rlhf"
RAY_TMPDIR="/data0/kongmu/r_v4train"
TASK_TMPDIR="/data0/kongmu/t_v4train"
TOTAL_STEPS=300

resume_step=""
case "$PHASE" in
  config)
    RUN_ID="config_preview"
    OUTPUT_DIR="$OUTPUT_ROOT/config_preview"
    CHECKPOINT_ROOT="$SMOKE_CHECKPOINT_ROOT"
    TURN_COMPONENT_DIR="$OUTPUT_ROOT/config_preview_components"
    RESUME_MODE=disable
    RESUME_FROM_PATH=null
    REQUIRE_COMPLETE=False
    SAVE_FREQ=2
    STOP_AFTER_STEPS=2
    ;;
  smoke-start)
    RUN_ID="${URCR_V4_RUN_ID:-smoke_s0_s3}"
    OUTPUT_DIR="$OUTPUT_ROOT/smoke/segment_000"
    CHECKPOINT_ROOT="$SMOKE_CHECKPOINT_ROOT"
    TURN_COMPONENT_DIR="$OUTPUT_ROOT/smoke/turn_components"
    RESUME_MODE=disable
    RESUME_FROM_PATH=null
    REQUIRE_COMPLETE=False
    SAVE_FREQ=-1
    STOP_AFTER_STEPS=3
    ;;
  formal)
    RUN_ID="${URCR_V4_RUN_ID:-formal_segment_000}"
    OUTPUT_DIR="$OUTPUT_ROOT/formal/$RUN_ID"
    CHECKPOINT_ROOT="$FORMAL_CHECKPOINT_ROOT"
    TURN_COMPONENT_DIR="$OUTPUT_ROOT/formal/turn_components"
    RESUME_MODE=disable
    RESUME_FROM_PATH=null
    REQUIRE_COMPLETE=False
    SAVE_FREQ=25
    STOP_AFTER_STEPS=null
    ;;
  resume)
    tracker="$FORMAL_CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
    if [[ ! -s "$tracker" ]]; then
      echo "no URCR-V4 formal rolling checkpoint is available" >&2
      exit 1
    fi
    resume_step="$(<"$tracker")"
    if [[ ! "$resume_step" =~ ^[0-9]+$ ]]; then
      echo "invalid URCR-V4 checkpoint tracker: $resume_step" >&2
      exit 1
    fi
    RUN_ID="${URCR_V4_RUN_ID:-formal_resume_s${resume_step}}"
    OUTPUT_DIR="$OUTPUT_ROOT/formal/$RUN_ID"
    CHECKPOINT_ROOT="$FORMAL_CHECKPOINT_ROOT"
    TURN_COMPONENT_DIR="$OUTPUT_ROOT/formal/turn_components"
    RESUME_MODE=resume_path
    RESUME_FROM_PATH="$FORMAL_CHECKPOINT_ROOT/global_step_$resume_step"
    REQUIRE_COMPLETE=True
    SAVE_FREQ=25
    STOP_AFTER_STEPS=null
    ;;
  *)
    echo "usage: $0 {config|smoke-start|formal|resume} [retrieval_url]" >&2
    exit 2
    ;;
esac

if [[ ! "$RUN_ID" =~ ^[a-z0-9_]+$ ]]; then
  echo "URCR_V4_RUN_ID must contain only lowercase letters, digits, and underscores" >&2
  exit 2
fi
for path in "$PYTHON" "$MODEL/config.json" "$MODEL/model.safetensors.index.json" \
  "$TRAIN_DATA" "$VALIDATION_DATA" "$RESOLVED_METHOD" "$RESOLVED_SIDECAR"; do
  if [[ ! -e "$path" ]]; then
    echo "missing URCR-V4 training input: $path" >&2
    exit 1
  fi
done
RESOLVED_SHA256="$(awk 'NR == 1 {print $1}' "$RESOLVED_SIDECAR")"
PYTHONPATH="$REPO_ROOT" "$PYTHON" - "$RESOLVED_METHOD" "$RESOLVED_SHA256" <<'PY'
import sys
from verl.trainer.ppo.urcr_v4_resolution import load_resolved_method

load_resolved_method(sys.argv[1], expected_sha256=sys.argv[2])
print("URCR_V4_RESOLVED_METHOD_VERIFIED")
PY

validate_checkpoint() {
  PYTHONPATH="$REPO_ROOT" "$PYTHON" - \
    "$RESUME_FROM_PATH" "$resume_step" "$RESOLVED_METHOD" "$RESOLVED_SHA256" <<'PY'
import json
from pathlib import Path
import sys

from verl.trainer.ppo.plan06_checkpointing import require_complete_checkpoint

checkpoint = Path(sys.argv[1])
step = int(sys.argv[2])
method_path = str(Path(sys.argv[3]).resolve())
method_sha = sys.argv[4]
require_complete_checkpoint(checkpoint, global_step=step, expected_world_size=4)
config = json.loads((checkpoint / "resolved_config.json").read_text(encoding="utf-8"))
v4 = config["algorithm"]["urcr_v4"]
if v4["mode"] != "train" or not v4["enable"]:
    raise RuntimeError("checkpoint is not a URCR-V4 train checkpoint")
if str(Path(v4["resolved_method_artifact"]).resolve()) != method_path:
    raise RuntimeError("checkpoint uses another URCR-V4 method artifact")
if v4["resolved_method_sha256"] != method_sha:
    raise RuntimeError("checkpoint uses another URCR-V4 method hash")
if int(config["trainer"]["total_training_steps"]) != 300:
    raise RuntimeError("checkpoint scheduler horizon is not 300")
print(f"URCR_V4_CHECKPOINT_VERIFIED step={step}")
PY
}

validate_formal_placeholder() {
  PYTHONPATH="$REPO_ROOT" "$PYTHON" - \
    "$FORMAL_CHECKPOINT_ROOT" "$MODEL" "$TRAIN_DATA" <<'PY'
import json
from pathlib import Path
import sys

from verl.trainer.ppo.plan06_checkpointing import require_complete_checkpoint

root = Path(sys.argv[1])
model = sys.argv[2]
train_data = sys.argv[3]
tracker = root / "latest_checkpointed_iteration.txt"
if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != "2":
    raise RuntimeError("formal Base restart requires the single V4 S2 placeholder")
step_dirs = sorted(path.name for path in root.glob("global_step_*") if path.is_dir())
if step_dirs != ["global_step_2"]:
    raise RuntimeError(f"unexpected rolling checkpoint population: {step_dirs}")
checkpoint = root / "global_step_2"
require_complete_checkpoint(checkpoint, global_step=2, expected_world_size=4)
config = json.loads((checkpoint / "resolved_config.json").read_text(encoding="utf-8"))
v4 = config["algorithm"]["urcr_v4"]
if v4["mode"] != "train" or not v4["enable"]:
    raise RuntimeError("rolling placeholder is not the verified V4 S2 smoke")
if config["actor_rollout_ref"]["model"]["path"] != model:
    raise RuntimeError("rolling placeholder was not initialized from the Base model")
if config["data"]["train_files"] != train_data:
    raise RuntimeError("rolling placeholder uses another train dataset")
if int(config["trainer"]["total_training_steps"]) != 300:
    raise RuntimeError("rolling placeholder scheduler horizon is not 300")
print("URCR_V4_S2_PLACEHOLDER_VERIFIED")
PY
}

if [[ "$PHASE" == smoke-start ]]; then
  if [[ -e "$SMOKE_CHECKPOINT_ROOT" || -e "$OUTPUT_ROOT/smoke" ]]; then
    echo "refusing to overwrite an existing URCR-V4 smoke" >&2
    exit 1
  fi
elif [[ "$PHASE" == formal ]]; then
  validate_formal_placeholder
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "refusing to overwrite an existing URCR-V4 formal run" >&2
    exit 1
  fi
elif [[ "$PHASE" == resume ]]; then
  validate_checkpoint
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "refusing to overwrite URCR-V4 resume output: $OUTPUT_DIR" >&2
    exit 1
  fi
fi

set +u
source /data2/kongmu/miniforge3/etc/profile.d/conda.sh
conda activate evaluation
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT"

hydra_args=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=True
  algorithm.use_kl_in_reward=False
  algorithm.evisd.enable=False
  algorithm.evisd.answer_enable=False
  algorithm.evisd.search_enable_for_main_loss=False
  algorithm.evisd.search_shadow_enable=False
  algorithm.evisd.save_case_study=False
  algorithm.urcr.enable=False
  algorithm.urcr.source.enable=False
  algorithm.urcr.source.lambda_a=0.0
  algorithm.urcr.routing.lambda_r=0.0
  algorithm.urcr.audit.compute_shadow_modes=False
  algorithm.urcr.audit.compute_q0_shadow=False
  algorithm.urcr.audit.save_surrogate_coefficients=False
  algorithm.urcr.audit.save_turn_components=False
  algorithm.urcr.audit.capture_update_summary=False
  algorithm.urcr.audit.capture_update_vectors=False
  algorithm.urcr.audit.save_update_reference=False
  algorithm.urcr.audit.compare_update_reference=False
  ++algorithm.urcr.answer_agam.enable=False
  ++algorithm.urcr.answer_agam.lambda=0.0
  algorithm.urcr_diagnostics.enable=False
  algorithm.urcr_v4.enable=True
  algorithm.urcr_v4.mode=train
  algorithm.urcr_v4.resolved_method_artifact="$RESOLVED_METHOD"
  algorithm.urcr_v4.resolved_method_sha256="$RESOLVED_SHA256"
  algorithm.urcr_v4.run_id="$RUN_ID"
  algorithm.urcr_v4.turn_component_dir="$TURN_COMPONENT_DIR"
  algorithm.urcr_v4.scorer.max_total_tokens=8192
  algorithm.urcr_v4.scorer.micro_batch_size_per_gpu=32
  algorithm.urcr_v4.scorer.token_budget_per_gpu=262144
  reward_model.enable=False
  data.train_files="$TRAIN_DATA"
  data.val_files="$VALIDATION_DATA"
  +data.cache_dir="$CACHE_DIR"
  +data.seed=1
  data.shuffle=True
  data.train_batch_size=128
  data.val_batch_size=512
  data.max_prompt_length=4096
  data.max_response_length=512
  data.filter_overlong_prompts=True
  data.truncation=left
  data.return_raw_chat=True
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
  actor_rollout_ref.actor.optim.total_training_steps="$TOTAL_STEPS"
  actor_rollout_ref.actor.ppo_mini_batch_size=256
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.shuffle=False
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.entropy_coeff=0.001
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  +actor_rollout_ref.actor.use_sdl_loss=False
  +actor_rollout_ref.actor.use_sdar_loss=False
  actor_rollout_ref.actor.use_invalid_action_penalty=True
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.01
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  critic.model.tokenizer_path="$MODEL"
  critic.optim.total_training_steps="$TOTAL_STEPS"
  reward_model.model.input_tokenizer="$MODEL"
  env.env_name=search
  env.seed=0
  env.max_steps=4
  env.rollout.n=8
  env.history_length=4
  env.search.topk=3
  env.search.search_url="$SEARCH_URL"
  env.search.log_requests=False
  trainer.balance_batch=True
  trainer.critic_warmup=0
  "trainer.logger=[console]"
  trainer.project_name=urcr_v4
  trainer.experiment_name=urcr_v4_qwen2p5_3b_4gpu_300step
  trainer.n_gpus_per_node="$EXPECTED_WORLD_SIZE"
  trainer.nnodes=1
  trainer.ray_wait_register_center_timeout=600
  trainer.default_local_dir="$CHECKPOINT_ROOT"
  trainer.default_hdfs_dir=null
  trainer.del_local_ckpt_after_load=False
  trainer.max_actor_ckpt_to_keep=null
  trainer.max_critic_ckpt_to_keep=null
  trainer.rollout_data_dir=null
  trainer.validation_data_dir=null
  trainer.log_val_generations=0
  trainer.save_freq="$SAVE_FREQ"
  trainer.resume_mode="$RESUME_MODE"
  trainer.resume_from_path="$RESUME_FROM_PATH"
  trainer.resume_load_dataloader_state=True
  trainer.require_complete_checkpoint="$REQUIRE_COMPLETE"
  trainer.exit_after_load=False
  trainer.checkpoint_rotation.enable=True
  trainer.checkpoint_rotation.milestone_step=null
  trainer.checkpoint_rotation.milestone_local_dir=null
  trainer.checkpoint_rotation.rolling_minimum_free_gib=38
  trainer.checkpoint_rotation.milestone_minimum_free_gib=0.0
  trainer.checkpoint_rotation.verify_after_save=True
  trainer.test_freq=-1
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.stop_after_steps="$STOP_AFTER_STEPS"
  trainer.val_before_train=False
  +ray_init.address=null
  ray_init.num_cpus=64
  hydra.run.dir="$OUTPUT_DIR/hydra"
)

mkdir -p "$OUTPUT_DIR"
RESOLVED_CONFIG="$OUTPUT_DIR/resolved_config.yaml"
"$PYTHON" - "$REPO_ROOT/verl/trainer/config" "$RESOLVED_CONFIG" \
  "${hydra_args[@]}" <<'PY'
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(config_dir=sys.argv[1], version_base=None):
    config = compose(config_name="ppo_trainer", overrides=sys.argv[3:])
Path(sys.argv[2]).write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
PY
"$PYTHON" - "$RESOLVED_CONFIG" "$MODEL" "$TRAIN_DATA" "$RESOLVED_METHOD" \
  "$RESOLVED_SHA256" "$CHECKPOINT_ROOT" "$SAVE_FREQ" "$STOP_AFTER_STEPS" \
  "$PHASE" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

c = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
model, train_data, method, method_sha, checkpoint_root = sys.argv[2:7]
save_freq = int(sys.argv[7])
stop_after = None if sys.argv[8] == "null" else int(sys.argv[8])
phase = sys.argv[9]
a = c["actor_rollout_ref"]["actor"]
r = c["actor_rollout_ref"]["rollout"]
v4 = c["algorithm"]["urcr_v4"]
legacy = c["algorithm"]["urcr"]
assert c["algorithm"]["adv_estimator"] == "grpo"
assert c["algorithm"]["norm_adv_by_std_in_grpo"] is True
assert c["algorithm"]["use_kl_in_reward"] is False
assert c["algorithm"]["evisd"]["enable"] is False
assert c["algorithm"]["evisd"]["answer_enable"] is False
assert c["algorithm"]["evisd"]["search_enable_for_main_loss"] is False
assert c["algorithm"]["evisd"]["search_shadow_enable"] is False
assert legacy["enable"] is False and legacy["source"]["enable"] is False
assert float(legacy["source"]["lambda_a"]) == 0.0
assert float(legacy["routing"]["lambda_r"]) == 0.0
assert legacy["answer_agam"] == {"enable": False, "lambda": 0.0}
assert c["algorithm"]["urcr_diagnostics"]["enable"] is False
assert v4["enable"] is True and v4["mode"] == "train"
assert str(Path(v4["resolved_method_artifact"]).resolve()) == str(Path(method).resolve())
assert v4["resolved_method_sha256"] == method_sha
assert int(v4["scorer"]["micro_batch_size_per_gpu"]) == 32
assert int(v4["scorer"]["token_budget_per_gpu"]) == 262144
assert c["actor_rollout_ref"]["model"]["path"] == model
assert c["data"]["train_files"] == train_data
assert c["data"]["shuffle"] is True and int(c["data"]["seed"]) == 1
assert int(c["data"]["train_batch_size"]) == 128
assert int(c["env"]["rollout"]["n"]) == 8 and int(c["env"]["seed"]) == 0
assert int(c["env"]["max_steps"]) == 4 and int(c["env"]["search"]["topk"]) == 3
assert int(r["n"]) == 1 and "seed" not in r and float(r["temperature"]) == 1.0
assert float(r["gpu_memory_utilization"]) == 0.5
assert r["enable_chunked_prefill"] is False and int(r["max_num_batched_tokens"]) == 8192
assert int(a["ppo_mini_batch_size"]) == 256
assert int(a["ppo_micro_batch_size_per_gpu"]) == 16 and int(a["ppo_epochs"]) == 1
assert a["loss_agg_mode"] == "token-mean" and a["shuffle"] is False
assert a["use_kl_loss"] is True and float(a["kl_loss_coef"]) == 0.001
assert float(a["entropy_coeff"]) == 0.001
assert a["use_sdl_loss"] is False and a["use_sdar_loss"] is False
assert a["fsdp_config"]["param_offload"] is False
assert a["fsdp_config"]["optimizer_offload"] is False
assert c["trainer"]["default_local_dir"] == checkpoint_root
assert int(c["trainer"]["total_training_steps"]) == 300
assert int(c["trainer"]["save_freq"]) == save_freq
assert c["trainer"]["stop_after_steps"] == stop_after
assert "seed" not in c["trainer"]
if phase == "formal":
    assert c["trainer"]["resume_mode"] == "disable"
    assert c["trainer"]["resume_from_path"] is None
print("URCR_V4_CONFIG_EXACT")
PY

if [[ "$PHASE" == config ]]; then
  exit 0
fi

for path in "$RAY_TMPDIR" "$TASK_TMPDIR"; do
  if [[ -e "$path" ]]; then
    echo "refusing to reuse stale URCR-V4 temporary path: $path" >&2
    exit 1
  fi
done
gpu_free="$({ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i "$PHYSICAL_GPUS"; } | awk -F, '{gsub(/ /,"",$2); if ($2 < 40000) bad=1} END {print bad+0}')"
if [[ "$gpu_free" != 0 ]]; then
  echo "URCR-V4 training requires at least 40000 MiB free on GPUs $PHYSICAL_GPUS" >&2
  exit 1
fi
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kib < 86 * 1024 * 1024 )); then
  echo "URCR-V4 training requires at least 86 GiB available host memory" >&2
  exit 1
fi
env -u LD_LIBRARY_PATH /usr/bin/curl --max-time 5 -fsS -o /dev/null \
  -H 'Content-Type: application/json' \
  -d '{"queries":["URCR V4 training health check"],"topk":1,"return_scores":false}' \
  "$SEARCH_URL"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy RAY_ADDRESS
export RAY_memory_usage_threshold=0.99
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS"
export RAY_TMPDIR
export TMPDIR="$TASK_TMPDIR"
export TMP="$TASK_TMPDIR"
export TEMP="$TASK_TMPDIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
mkdir -p "$RAY_TMPDIR" "$TASK_TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR"
cleanup() {
  rm -rf -- "$RAY_TMPDIR" "$TASK_TMPDIR"
}
trap cleanup EXIT INT TERM

git -C "$WORKSPACE_ROOT" rev-parse HEAD >"$OUTPUT_DIR/git_head.txt"
git -C "$WORKSPACE_ROOT" status --short --untracked-files=all >"$OUTPUT_DIR/git_status.txt"
git -C "$WORKSPACE_ROOT" diff --no-ext-diff | sha256sum >"$OUTPUT_DIR/git_diff_sha256.txt"

cd "$REPO_ROOT"
"$PYTHON" -m verl.trainer.main_evisd \
  --config-path "$REPO_ROOT/verl/trainer/config" \
  --config-name ppo_trainer \
  "${hydra_args[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
