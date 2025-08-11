import os
import copy
import argparse
import csv
from datetime import datetime
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, DataCollatorForLanguageModeling
from datasets import load_dataset
import gc

import numpy as np
from tqdm import tqdm

import time
import tracemalloc


from cur_models import CURLinear, WandaWrappedModule, rebuild_model_with_W
from cur import apply_cur_to_matrix
from utils import calculate_sparsity, calculate_model_size, calculate_per_layer_frobenius_norm, calculate_per_layer_frobenius_norm_diff, compute_block_distances, get_last_non_padded_tokens


parser = argparse.ArgumentParser(
    description="CUR Decomposition for Transformers")

# Model and Paths
parser.add_argument('--model_name', type=str,
                    # "neuralmagic/Sparse-Llama-3.1-8B-2of4"
                    # "vuiseng9/Meta-Llama-3.1-8B-wanda-unstructured-0.5"  # sparsity 0.5
                    default="meta-llama/Llama-3.1-8B",
                    help='Name or path of the pre-trained model to use.')
parser.add_argument('--save_path', type=str, default="./cur_decomposed_models",
                    help='Directory path where the modified model and outputs will be saved.')
parser.add_argument('--model_save', action='store_true', default=False,
                    help='Set this flag to save the model (default: False).')
parser.add_argument('--device', type=str, default="cuda",
                    help='Device to run the computations on (e.g., "cpu", "cuda").')

# Data Loading and Calibration
parser.add_argument('--dataset', type=str, default="c4",
                    help='Dataset to use.')
parser.add_argument('--dataset_category', type=str, default="en",
                    help='Dataset category to use.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for data loading.')
parser.add_argument('--num_calibration_steps', type=int, default=128,
                    help='Number of calibration steps (number of batches to process).')
parser.add_argument('--max_length', type=int, default=128,
                    help='Max length per calibration dataset.')

# CUR Decomposition Parameters
parser.add_argument('--num_curing_layers', type=int, default=10,
                    help='Number of layers to apply CUR decomposition to.')
parser.add_argument('--max_rank', type=int, default=256,
                    help='Maximum rank for CUR decomposition.')

# Module Names to Process
parser.add_argument('--ffn_module_names', nargs='*', default=["gate_proj"],
                    help='List of FFN module names to process.')
parser.add_argument('--attn_module_names', nargs='*', default=["q_proj", "k_proj"],
                    help='List of attention module names to process.')

# Miscellaneous
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')

args = parser.parse_args()


# Set random seed for reproducibility
set_seed(args.seed)


# Set model name and paths
model_name = args.model_name
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
# Create a timestamped directory to avoid overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
child_path = f"C{args.num_calibration_steps}_N{args.num_curing_layers}_R{args.max_rank}_{timestamp}"
with open(os.path.join(save_path, "latest.txt"), "w") as f:
    f.write(child_path)
save_path = os.path.join(save_path, child_path)
os.makedirs(save_path, exist_ok=True)

# Handle device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == args.device, \
    f"Device does not match the expected one. {args.device} != {device}"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    # print(f"The tokenizer.pad_token set as {tokenizer.eos_token}")
    # TODO: use other token?

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

# Set batch size
batch_size = args.batch_size

# Load the C4 dataset with streaming
num_calibration_steps = args.num_calibration_steps
max_length = args.max_length

dataset = {
    'train': load_dataset(
        args.dataset,
        args.dataset_category,
        split='train',
        streaming=True,
        trust_remote_code=True
    ).take(num_calibration_steps),
}


def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        return_special_tokens_mask=True,
        max_length=max_length,
        truncation=True
    )


tokenized_dataset = {
    'train': dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].features.keys()
    ),
}

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors='pt'
)

calibration_dataloader = DataLoader(
    tokenized_dataset['train'],
    batch_size=batch_size,
    collate_fn=data_collator,
)


# main


# Calculate original model statistics
original_model_size_bits_nonzero = calculate_model_size(
    model, exclude_zeros=True)
original_model_size_bits_zero = calculate_model_size(
    model, exclude_zeros=False)
original_sparsity = calculate_sparsity(model)
original_per_layer_norms = calculate_per_layer_frobenius_norm(model)


# Initialize a dictionary to store WandaWrappedModule instances for all modules
wrapped_modules = {}
# ffn_module_names = [
#     "gate_proj",
#     # "up_proj",
#     # "down_proj",
# ]
# attn_module_names = [
#     "q_proj",
#     "k_proj",
#     # "v_proj",
#     # "o_proj",
# ]
ffn_module_names = args.ffn_module_names
attn_module_names = args.attn_module_names


def register_hooks_for_layers(model, wrapped_modules):
    # TODO: GQA

    for layer_index, layer in enumerate(model.model.layers):
        # Process FFN modules
        for name in ffn_module_names:
            if hasattr(layer.mlp, name):
                module = getattr(layer.mlp, name)
                wrapped_module = WandaWrappedModule(module)
                wrapped_module.register_hook()
                key = f"layer_{layer_index}_mlp_{name}"
                wrapped_modules[key] = wrapped_module

        # Process Attention modules
        for name in attn_module_names:
            if hasattr(layer.self_attn, name):
                module = getattr(layer.self_attn, name)
                wrapped_module = WandaWrappedModule(module)
                wrapped_module.register_hook()
                key = f"layer_{layer_index}_self_attn_{name}"
                wrapped_modules[key] = wrapped_module


# Register hooks for target layers
register_hooks_for_layers(model, wrapped_modules)

# Initialize containers for distances
# - One less (-1) for adjacent layers comparison
# - Another one less (-1) for excluding last layer
all_distances = [[] for _ in range(model.config.num_hidden_layers - 2)]
num_CURing_layer = args.num_curing_layers


# Benchmark-1


def get_gpu_usage():
    return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()


tracemalloc.start()
gpu_mem_start, gpu_max_start = get_gpu_usage()
start_time = time.time()


# Process WANDA
# Pass data through the original model once to collect activations
model.eval()
with torch.no_grad():
    progress_bar = tqdm(
        enumerate(calibration_dataloader),
        total=num_calibration_steps,
        desc=f"Calibration",
    )
    for step, batch in progress_bar:
        if step >= num_calibration_steps:
            break
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(
            inputs,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states

        # Get last non-padded tokens
        last_non_padded_hidden_states = get_last_non_padded_tokens(
            hidden_states, attention_mask)
        # Remove the input embedding layer (first element of hidden_states)
        last_non_padded_hidden_states = last_non_padded_hidden_states[1:]
        # Ensure consistency between number of hidden layers and extracted layers
        assert len(last_non_padded_hidden_states) == model.config.num_hidden_layers, \
            "Length of last_non_padded_hidden_states does not match the expected number of hidden layers."

        # Compute distances for adjacent layers
        distances = compute_block_distances(last_non_padded_hidden_states)
        for i, distance in enumerate(distances):
            all_distances[i].append(distance)

# Calculate average distances for each block
average_distances = [np.mean(block_distances)
                     for block_distances in all_distances]

# Pair last indices and distances
indices_to_cur = [i + 1 for i in range(len(average_distances))]
distances_sorted = sorted(
    zip(indices_to_cur, average_distances), key=lambda x: x[1], reverse=False)


current_mem, peak_mem = tracemalloc.get_traced_memory()
gpu_mem_end, gpu_max_end = get_gpu_usage()
end_time = time.time()
execution_time = end_time - start_time
tracemalloc.stop()

print(f"\nWANDA:")
print(f"  Execution Time\t: {execution_time:.2f} seconds")
print(
    f"  Mem (Current)\t: {current_mem / 10**6:.2f} MB, (Peak): {peak_mem / 10**6:.2f} MB")
print(
    f"  GPU (Current)\t: {gpu_mem_end - gpu_mem_start} bytes, (Peak): {gpu_max_end - gpu_max_start} bytes")

results = {
    "Execution Time (s)": execution_time,
    "Current Memory Usage (MB)": current_mem / 10**6,
    "Peak Memory Usage (MB)": peak_mem / 10**6,
    "GPU Memory Usage (Bytes)": gpu_mem_end - gpu_mem_start,
    "GPU Max Memory Usage (Bytes)": gpu_max_end - gpu_max_start,
}

metrics_csv_path = os.path.join(save_path, "performance_metrics_wanda.csv")
with open(metrics_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(results.keys())
    writer.writerow(results.values())


# ========================

########
# TEST #
########


# Separate into two lists

# Angular Distance
sorted_indices_to_cur = ([pair[0] for pair in distances_sorted])[
    :num_CURing_layer]
sorted_distances = ([pair[1] for pair in distances_sorted])[:num_CURing_layer]

# # Last layers
# sorted_indices_to_cur = list(range(30, 0, -1)[:num_CURing_layer])
# sorted_distances = [0.0 for _ in range(len(sorted_indices_to_cur))]

# # Random layers
# sorted_indices_to_cur = random.sample(range(1, 31), num_CURing_layer)
# sorted_distances = [0.0 for _ in range(len(sorted_indices_to_cur))]

# ========================


# Output results
print("\nLast indices and sorted distances:")
print("  Last Indices:", sorted_indices_to_cur)
print("  Distances:", sorted_distances)
print("  Model: ", end='')
for layer_index in range(len(model.model.layers)):
    if layer_index in sorted_indices_to_cur:
        print("X", end='')
    else:
        print("-", end='')
print()
# Save distances to CSV
distances_csv_path = os.path.join(save_path, "layer_distances.csv")
with open(distances_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Layer_Index", "Average_Distance"])
    for idx, dist in zip(indices_to_cur, average_distances):
        writer.writerow([idx, dist])
# Save selected layers to CSV
selected_layers_csv_path = os.path.join(save_path, "selected_layers.csv")
with open(selected_layers_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Selected_Layer_Index", "Selected_Average_Distance"])
    for idx, dist in zip(sorted_indices_to_cur, sorted_distances):
        writer.writerow([idx, dist])

# Remove all hooks after collection
for wrapped_module in wrapped_modules.values():
    wrapped_module.remove_hook()


# Create a copy of the model to modify
modified_model = copy.deepcopy(model)

max_rank = args.max_rank


# Benchmark-2


tracemalloc.start()
gpu_mem_start, gpu_max_start = get_gpu_usage()
start_time = time.time()


# TODO: tqdm
# TODO: parallel
# Process (CURing) each layer using the collected activations
for layer_index in range(len(model.model.layers) - 1):
    # skip last layer

    layer = model.model.layers[layer_index]
    modified_layer = modified_model.model.layers[layer_index]

    if layer_index in sorted_indices_to_cur:
        # Process FFN modules
        for name in ffn_module_names:
            key = f"layer_{layer_index}_mlp_{name}"
            wrapped_module = wrapped_modules[key]
            activation_norm = wrapped_module.get_activation_norm()
            module = getattr(layer.mlp, name)
            weight = module.weight.data

            # print(f"Processing layer {layer_index}, module '{name}':")
            C, U, R, rank, row_indices, col_indices = apply_cur_to_matrix(
                weight, activation_norm, max_rank)
            # Create CURLinear module
            bias = module.bias.data.clone() if module.bias is not None else None
            cur_linear = CURLinear(C, U, R, bias, row_indices, col_indices)
            # Replace the module in the modified model
            setattr(modified_layer.mlp, name, cur_linear)

        # Process Attention modules
        for name in attn_module_names:
            key = f"layer_{layer_index}_self_attn_{name}"
            wrapped_module = wrapped_modules[key]
            activation_norm = wrapped_module.get_activation_norm()
            module = getattr(layer.self_attn, name)
            weight = module.weight.data

            # print(f"Processing layer {layer_index}, module '{name}':")
            # print(f"  Weight shape: {weight.shape}")
            # print(f"  Activation norm shape: {activation_norm.shape}")
            C, U, R, rank, row_indices, col_indices = apply_cur_to_matrix(
                weight, activation_norm, max_rank)
            # Create CURLinear module
            bias = module.bias.data.clone() if module.bias is not None else None
            cur_linear = CURLinear(C, U, R, bias, row_indices, col_indices)
            # Replace the module in the modified model
            setattr(modified_layer.self_attn, name, cur_linear)


current_mem, peak_mem = tracemalloc.get_traced_memory()
gpu_mem_end, gpu_max_end = get_gpu_usage()
end_time = time.time()
execution_time = end_time - start_time
tracemalloc.stop()

print(f"\nCUR:")
print(f"  Execution Time\t: {execution_time:.2f} seconds")
print(
    f"  Mem (Current)\t: {current_mem / 10**6:.2f} MB, (Peak): {peak_mem / 10**6:.2f} MB")
print(
    f"  GPU (Current)\t: {gpu_mem_end - gpu_mem_start} bytes, (Peak): {gpu_max_end - gpu_max_start} bytes")

results = {
    "Execution Time (s)": execution_time,
    "Current Memory Usage (MB)": current_mem / 10**6,
    "Peak Memory Usage (MB)": peak_mem / 10**6,
    "GPU Memory Usage (Bytes)": gpu_mem_end - gpu_mem_start,
    "GPU Max Memory Usage (Bytes)": gpu_max_end - gpu_max_start,
}

metrics_csv_path = os.path.join(save_path, "performance_metrics_cur.csv")
with open(metrics_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(results.keys())
    writer.writerow(results.values())


# Save the modified model
if args.model_save:
    # modified_model.save_pretrained(save_path)
    torch.save(modified_model, os.path.join(save_path, 'model.pt'))
    tokenizer.save_pretrained(save_path)
    # print("CUR decomposition applied and modified_model saved successfully.")


# Calculate differences in model size and Frobenius norms


cur_model_size_bits_nonzero = calculate_model_size(
    modified_model, exclude_zeros=True)
cur_model_size_bits_zero = calculate_model_size(
    modified_model, exclude_zeros=False)
size_difference_bits_nonzero = original_model_size_bits_nonzero - \
    cur_model_size_bits_nonzero
size_difference_bits_zero = original_model_size_bits_zero - cur_model_size_bits_zero
cur_model_sparsity = calculate_sparsity(modified_model)

# Rebuild the model with W
rebuilt_model = rebuild_model_with_W(modified_model)
rebuilt_model.to(device)
# cur_model_sparsity = calculate_sparsity(rebuilt_model)
cur_per_layer_norms = calculate_per_layer_frobenius_norm(rebuilt_model)
cur_rebuilt_model_sparsity = calculate_sparsity(rebuilt_model)
# Frob norm diff
frob_norm_diff = calculate_per_layer_frobenius_norm_diff(model, rebuilt_model)

# Free GPU mem
model.to('cpu')
del model
gc.collect()
torch.cuda.empty_cache()

# Free GPU mem
modified_model.to('cpu')
del modified_model
gc.collect()
torch.cuda.empty_cache()

# Print per-layer Frobenius norms
print("\nPer-layer Frobenius norms:")
norms_csv_path = os.path.join(save_path, "per_layer_frobenius_norms.csv")
with open(norms_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Layer_Name", "Original_Frobenius_Norm",
                    "CUR_Frobenius_Norm", "Difference"])
    for layer_name in sorted(original_per_layer_norms.keys(), key=lambda x: int(x.split('_')[1]) if x != "Other" else -1):
        original_norm = original_per_layer_norms.get(layer_name, 0.0)
        cur_norm = cur_per_layer_norms.get(layer_name, 0.0)
        norm_difference = frob_norm_diff.get(layer_name, 0.0)
        print(f"{layer_name}:")
        print(f"  Original Frobenius norm: {original_norm}")
        print(f"  CUR-decomposed Frobenius norm: {cur_norm}")
        print(f"  Difference: {norm_difference}")
        writer.writerow([layer_name, original_norm, cur_norm, norm_difference])

# Print total model sparsity
print(f"\nSparsity:")
print(f"  Original model sparsity: {original_sparsity}")
print(f"  CUR-decomposed model sparsity: {cur_model_sparsity}")
print(f"  Rebuilt CUR model sparsity: {cur_rebuilt_model_sparsity}")
sparsity_csv_path = os.path.join(save_path, "model_sparsity.csv")
with open(sparsity_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Model_Version", "Sparsity"])
    writer.writerow(["Original", original_sparsity])
    writer.writerow(["CUR_Decomposed", cur_model_sparsity])
    writer.writerow(["Rebuilt", cur_rebuilt_model_sparsity])

# Print total model sizes
print(f"\nTotal model size (excluding zeros):")
print(
    f"  Original model size: {original_model_size_bits_zero / (8 * 1024 ** 2):.2f} MB")
print(
    f"  CUR-decomposed model size: {cur_model_size_bits_zero / (8 * 1024 ** 2):.2f} MB")
print(
    f"  Size difference (nonzero): {size_difference_bits_nonzero / (8 * 1024 ** 2):.2f} MB")
print(
    f"  Size difference (zero): {size_difference_bits_zero / (8 * 1024 ** 2):.2f} MB")
model_size_csv_path = os.path.join(save_path, "model_sizes.csv")
with open(model_size_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Model_Version", "Size_MB"])
    writer.writerow(
        ["Original", original_model_size_bits_zero / (8 * 1024 ** 2)])
    writer.writerow(
        ["CUR_Decomposed", cur_model_size_bits_zero / (8 * 1024 ** 2)])
    writer.writerow(["Size_Difference_Nonzero",
                    size_difference_bits_nonzero / (8 * 1024 ** 2)])
    writer.writerow(
        ["Size_Difference_Zero", size_difference_bits_zero / (8 * 1024 ** 2)])


# # Sample output
# test_text = "Once upon a time"
# generated_text = generate_text(rebuilt_model, tokenizer, test_text, device=device)
# print("Sample Output:", generated_text)
