#!/bin/sh
DEVICE=${1:-0}
echo "Using CUDA device: ${DEVICE}"


# original models
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


BASE_SAVE="/data/lucid/curing/cur_decomposed_models"

# list of
# model_name, energy, min_rank
while read -r MODEL ENERGY MIN_RANK; do
  [ -z "$MODEL" ] && continue
  MODEL_TAG=$(printf "%s" "$MODEL" | tr '/:' '__')
  SAVE_PATH="${BASE_SAVE}/${MODEL_TAG}"
  mkdir -p "${SAVE_PATH}"

  echo "=== Model: ${MODEL} | energy=${ENERGY} | min_rank=${MIN_RANK}"
#   for N in $(seq 1 20); do
#   for N in $(seq 1 2 19); do
  for N in $(seq 2 2 20); do
    echo "==> num_curing_layers=${N}"
    CUDA_VISIBLE_DEVICES=${DEVICE} python CURing/curing.py \
      --model_name "${MODEL}" \
      --cur_metric "cov_fast" \
      --cur_mode "deim" \
      --min_rank ${MIN_RANK} \
      --energy ${ENERGY} \
      --num_curing_layers ${N} \
      --save_path "${SAVE_PATH}" \
      --device cuda \
      --dataset "c4" \
      --dataset_category "en" \
      --batch_size 1 \
      --num_calibration_steps 256 \
      --max_length 4096 \
      --ffn_module_names gate_proj \
      --attn_module_names q_proj k_proj \
      --seed 42 \
      --model_save

    LATEST=$(tr -d '[:space:]' < "${SAVE_PATH}/latest.txt")
    LOAD_PATH="${SAVE_PATH}/${LATEST}/"
    echo "Load path: ${LOAD_PATH}"

    CUDA_VISIBLE_DEVICES=${DEVICE} python benchmark/curing_model.py \
      --load_path "${LOAD_PATH}" \
      --device cuda \
      --max_length 4096 \
      --num_test_steps 128 \
      --seed 42 \
      > "${SAVE_PATH}/${LATEST}_benchmark.txt"

    echo "Saved: ${SAVE_PATH}/${LATEST}_benchmark.txt"
  done
done << 'EOF'
meta-llama/Llama-3.1-8B 0.88 1024
meta-llama/Llama-2-7b-hf 0.85 1024
meta-llama/Llama-2-13b-hf 0.85 1536
mistralai/Mistral-7B-v0.1 0.89 1024
microsoft/Orca-2-7b 0.85 1024
Qwen/Qwen3-8B-Base 0.91 1024
EOF