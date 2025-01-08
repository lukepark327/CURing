import math
from typing import List

import torch


# Inference


# Test text to see the result output
test_text = "Once upon a time"


def generate_text(model, tokenizer, prompt, max_length=128, device='cpu'):
    """
    Generate text using the given model and prompt.
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def print_dataloader_samples_text(dataloaders, tokenizer):
    for name, dataloader in dataloaders.items():
        print(f"\n--- {name} Dataset ---")

        batch = next(iter(dataloader))  # get first batch
        print(batch['input_ids'].shape)

        if len(batch['input_ids'].shape) == 3:
            # classification
            for idx, option_ids in enumerate(batch['input_ids'][0]):
                decoded_text = tokenizer.decode(
                    option_ids, skip_special_tokens=True)
                print(f"Decoded option {idx + 1}: {decoded_text}")
        else:
            # lm
            decoded_text = tokenizer.decode(
                batch['input_ids'][0], skip_special_tokens=True)
            print(f"Decoded text: {decoded_text}")

        print(batch['labels'])


# Model Properties


def calculate_sparsity(model):
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name and ("self_attn" in name or "mlp" in name):
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

    sparsity = zero_params / total_params
    return sparsity


def calculate_model_size(model, exclude_zeros=True):
    """
    Calculate the total size of the model weights (in bits).

    Parameters:
        model (torch.nn.Module): The model whose size is to be calculated.
        exclude_zeros (bool): Whether to exclude zero elements in the calculation.

    Returns:
        int: Total size of the model weights in bits.
    """
    total_size_bits = 0

    for name, param in model.named_parameters():
        # if "weight" in name:
        if exclude_zeros:
            non_zero_elements = (param != 0).sum().item()
            count_elements = non_zero_elements
        else:
            count_elements = param.numel()  # Total number of elements

        dtype_size = param.element_size() * 8  # Size of each element in bits
        size_bits = count_elements * dtype_size
        total_size_bits += size_bits

    for name, buffer in model.named_buffers():
        if exclude_zeros:
            non_zero_elements = (buffer != 0).sum().item()
            count_elements = non_zero_elements
        else:
            count_elements = buffer.numel()  # Total number of elements

        dtype_size = buffer.element_size() * 8  # Size of each element in bits
        size_bits = count_elements * dtype_size
        total_size_bits += size_bits

    return total_size_bits


def calculate_per_layer_frobenius_norm(model):
    """
    Calculate the per-layer Frobenius norm of the model weights.
    """
    layer_norms = {}

    for name, param in model.named_parameters():
        if "weight" in name:
            # Extract layer identifier
            parts = name.split('.')
            if 'layers' in parts:
                layer_index = parts[parts.index('layers') + 1]
                layer_name = f"Layer_{layer_index}"
            else:
                layer_name = "Other"

            norm_squared = torch.norm(param).item() ** 2

            if layer_name not in layer_norms:
                layer_norms[layer_name] = norm_squared
            else:
                layer_norms[layer_name] += norm_squared

    # Take square root to get the Frobenius norm
    for layer_name in layer_norms:
        layer_norms[layer_name] = math.sqrt(layer_norms[layer_name])

    return layer_norms


def calculate_per_layer_frobenius_norm_diff(model_1, model_2):
    """
    Calculate the per-layer Frobenius norm difference between two models.
    Specifically, for each layer, computes || W1 - W2 ||_F (Frobenius norm)
    where W1 and W2 are the corresponding weight tensors of model_1 and model_2.
    """
    # Convert model_2 named parameters into a dictionary for easy access
    model_2_params = dict(model_2.named_parameters())

    layer_norms_diff = {}

    for name, param1 in model_1.named_parameters():
        if "weight" in name:
            # Get corresponding parameter from model_2
            param2 = model_2_params[name]

            # Compute the difference
            difference = param1 - param2
            # Sum of squares of all elements
            diff_squared = torch.sum(difference * difference).item()

            # Extract layer identifier
            parts = name.split('.')
            if 'layers' in parts:
                layer_index = parts[parts.index('layers') + 1]
                layer_name = f"Layer_{layer_index}"
            else:
                layer_name = "Other"

            # Accumulate the squared difference
            if layer_name not in layer_norms_diff:
                layer_norms_diff[layer_name] = diff_squared
            else:
                layer_norms_diff[layer_name] += diff_squared

    # Take the square root of accumulated sums to get the Frobenius norm
    for layer_name in layer_norms_diff:
        layer_norms_diff[layer_name] = math.sqrt(layer_norms_diff[layer_name])

    return layer_norms_diff


# Angular Distance
# Reference: https://github.com/arcee-ai/PruneMe/blob/main/compute_block_similarity/utils.py#L11


def angular_distance(x_l, x_l_plus_n) -> torch.Tensor:
    """
    Compute the angular distance between layer output tokens.
    """
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    cosine_similarity = (x_l_norm * x_l_plus_n_norm).sum(-1)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def compute_block_distances(hidden_states: List[torch.Tensor]) -> List[float]:
    """
    Compute and return angular distances for each block of layers.
    """
    distances = []
    num_layers = len(hidden_states) - 1  # Excluding the last layer
    for l in range(num_layers - 1):  # Compare adjacent layers only
        # Calculate angular distance between layer l and l+1
        distance = angular_distance(
            hidden_states[l], hidden_states[l + 1]).mean().item()
        distances.append(distance)
    return distances


def get_last_non_padded_tokens(hidden_states, attention_mask) -> List[torch.Tensor]:
    """
    Get last non-padded tokens for each layer.
    """
    last_non_padded_hidden_states = []
    for layer in hidden_states:
        batch_size, _, _ = layer.size()
        batch_last_tokens = []
        for batch in range(batch_size):
            last_non_pad_index = attention_mask[batch].nonzero(as_tuple=True)[
                0].max()
            last_token = layer[batch, last_non_pad_index, :]
            batch_last_tokens.append(last_token.unsqueeze(0))
        last_non_padded_hidden_states.append(
            torch.cat(batch_last_tokens, dim=0))
    return last_non_padded_hidden_states
