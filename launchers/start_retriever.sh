#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_GPUS="${1:?usage: $0 <physical_gpu_ids> [port]}"
PORT="${2:-18248}"
MIN_FREE_MB="${URCR_RETRIEVER_MIN_FREE_MB:-25000}"
HF_DATASETS_CACHE="${URCR_HF_DATASETS_CACHE:-/data0/kongmu/agentic-RL/cache/huggingface/datasets}"

if ! nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$PHYSICAL_GPUS" \
  | awk -v minimum="$MIN_FREE_MB" '{ if ($1 < minimum) unavailable = 1 } END { exit unavailable }'; then
  echo "GPU guard failed: each of $PHYSICAL_GPUS needs ${MIN_FREE_MB} MiB free" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS"
export HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
exec /data2/kongmu/miniforge3/envs/retriever/bin/python \
  search_r1/search/retrieval_server.py \
  --index_path /data2/kongmu/pretrain/wiki/e5_Flat.index \
  --corpus_path /data2/kongmu/pretrain/wiki/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model /data2/kongmu/pretrain/e5-base-v2 \
  --faiss_gpu \
  --host 127.0.0.1 \
  --port "$PORT"
