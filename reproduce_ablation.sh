#!/bin/sh

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

DEVICE=${1:-0}
echo "Using CUDA device: ${DEVICE}"


MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG=$(printf "%s" "$MODEL" | tr '/:' '__')


# Layer

BASE_SAVE="/data/lucid/curing/cur_decomposed_models_layer"

for LAYER_METRIC in last random; do
  echo "=== Testing layer_metric: ${LAYER_METRIC} ==="

  SAVE_PATH="${BASE_SAVE}/${MODEL_TAG}__layer_${LAYER_METRIC}"
  mkdir -p "${SAVE_PATH}"

  CUDA_VISIBLE_DEVICES=${DEVICE} python CURing/curing.py \
      --model_name "${MODEL}" \
      --layer_metric "${LAYER_METRIC}" \
      --cur_metric "cov_fast" \
      --cur_mode "deim" \
      --min_rank 1024 \
      --energy 0.88 \
      --num_curing_layers 14 \
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

  LATEST=$(cat "${SAVE_PATH}/latest.txt")
  LOAD_PATH="${SAVE_PATH}/${LATEST}"
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


# CUR

BASE_SAVE="/data/lucid/curing/cur_decomposed_models_cur"

for CUR_METRIC in cov_fast weight; do
for CUR_MODE in deim magnitude random; do
  # skip cov_fast + deim / cov_fast + random
  if [ "$CUR_METRIC" = "cov_fast" ]; then
    case "$CUR_MODE" in
      deim|random)
        echo ">>> Skipping incompatible combo: metric=${CUR_METRIC}, mode=${CUR_MODE}"
        continue
        ;;
    esac
  fi

  echo "=== Testing metric: ${CUR_METRIC} mode: ${CUR_MODE} ==="

  SAVE_PATH="${BASE_SAVE}/${MODEL_TAG}__metric_${CUR_METRIC}__mode_${CUR_MODE}"
  mkdir -p "${SAVE_PATH}"

  CUDA_VISIBLE_DEVICES=${DEVICE} python CURing/curing.py \
      --model_name "${MODEL}" \
      --layer_metric "angular" \
      --cur_metric "${CUR_METRIC}" \
      --cur_mode "${CUR_MODE}" \
      --min_rank 1024 \
      --energy 0.88 \
      --num_curing_layers 14 \
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

  LATEST=$(cat "${SAVE_PATH}/latest.txt")
  LOAD_PATH="${SAVE_PATH}/${LATEST}"
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
done