#!/bin/sh

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

DEVICE=${1:-0}
echo "Using CUDA device: ${DEVICE}"

for MODEL in \
  "meta-llama/Llama-3.1-8B" \
  "meta-llama/Llama-2-7b-hf" \
  "meta-llama/Llama-2-13b-hf" \
  "mistralai/Mistral-7B-v0.1" \
  "microsoft/Orca-2-7b" \
  "Qwen/Qwen3-8B-Base"
do
  echo "==> load_path: ${MODEL}"
  CUDA_VISIBLE_DEVICES=${DEVICE} python benchmark/original_model.py \
      --load_path "${MODEL}" \
      --device cuda \
      --max_length 4096 \
      --num_test_steps 128 \
      --seed 42 \
      > "logs_$(echo "${MODEL}" | tr '/:' '__')_benchmark.txt"

  echo "Saved: logs_$(echo "${MODEL}" | tr '/:' '__')_benchmark.txt"
done