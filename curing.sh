# Default
python CURing/curing.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --save_path "./cur_decomposed_models" \
    --device cuda \
    --dataset "c4" \
    --dataset_category "en" \
    --batch_size 1 \
    --num_calibration_steps 128 \
    --num_curing_layers 10 \
    --max_rank 256 \
    --ffn_module_names gate_proj \
    --attn_module_names q_proj k_proj \
    --seed 42 \
    --model_save
