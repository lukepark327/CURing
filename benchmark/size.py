# autopep8: off
import os
import sys

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../CURing"))
)
from cur_models import CURLinear  # noqa: F401

CURLinear

# autopep8: on


dtype = torch.float32

MODEL_ID = "/data/lucid/curing/cur_decomposed_models_2/meta-llama_Llama-3.1-8B/cov_fast_deim_B1_C256_N11_RNone_E0.88_20250927_174612/"
# MODEL_ID = "/data/lucid/curing/cur_decomposed_models_4/meta-llama_Llama-3.1-8B/cov_fast_deim_B1_C256_N18_RNone_E0.88_20250927_172033/"
model = torch.load(os.path.join(MODEL_ID, 'model.pt'), weights_only=False)

# MODEL_ID = "meta-llama/Llama-3.1-8B"
# MODEL_ID = "mistralai/Mistral-7B-v0.1"
# MODEL_ID = "meta-llama/Llama-2-7b-hf"
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)

param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
total_gib = (param_bytes + buf_bytes) / (1024 ** 3)

print(f"Parameters: {param_bytes/(1024**3):.4f} GiB")
print(f"Buffers:    {buf_bytes/(1024**3):.4f} GiB")
print(f"Total:      {total_gib:.4f} GiB")
print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")
print(f"dtype: {dtype}")
