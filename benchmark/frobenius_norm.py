# python benchmark/frobenius_norm.py --load_path "./healed/C128_N10_R256_20241224_003822"


# autopep8: off
import os
import sys
import argparse
from datetime import datetime

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import set_seed, get_cosine_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import gc


from peft import LoraConfig, get_peft_model, TaskType


# Add the parent directory of CURing to sys.path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../CURing")))

import cur_models
from cur_models import CURLinear, rebuild_model_with_W

import utils
from utils import calculate_per_layer_frobenius_norm, calculate_per_layer_frobenius_norm_diff
# autopep8: on


os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(
    description="Benchmarking CUR-Decomposed Models")

parser.add_argument('--load_path', type=str, default=None,
                    help='Directory path from where the modified model is loaded.')
parser.add_argument('--save_path', type=str, default=None,
                    help='Directory path where the fine-tuned model and outputs will be saved.')
parser.add_argument('--log_dir', type=str, default=None,
                    help='Directory for TensorBoard logs.')

args = parser.parse_args()


# Set random seed for reproducibility
set_seed(42)


# Load the original model and tokenizer
assert args.load_path is not None, \
    f"No `load_path` defined."

load_path = f"{args.load_path}"
print(f"Loaded from {load_path}")

save_path = args.save_path

# log
if args.log_dir:
    log_dir = args.log_dir
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/forgetting_lora_{timestamp}"
print(f"Logging at {log_dir}")


# Define the CURLinear class to avoid error/warning
CURLinear

model = torch.load(os.path.join(load_path, 'model.pt'), weights_only=False)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", \
    f"Device does not match the expected one. cuda != {device}"

model.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(load_path)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    # print(f"The tokenizer.pad_token set as a {tokenizer.eos_token}")
    # TODO: use other token?


# def freeze_model_except_U_matrices(model):
#     for name, param in model.named_parameters():
#         if 'U' in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False


# freeze_model_except_U_matrices(model)


# Rebuild the model with W
rebuilt_model = rebuild_model_with_W(model)

# Free GPU mem
model.to('cpu')
del model
gc.collect()
torch.cuda.empty_cache()

rebuilt_model.to(device)
model = rebuilt_model

model.eval()


# Original Model


model_name = "meta-llama/Llama-3.1-8B"
original_model = AutoModelForCausalLM.from_pretrained(model_name)
original_model.to(device)

original_model.eval()


# Frobenius Norm


original_per_layer_norms = calculate_per_layer_frobenius_norm(original_model)
cur_per_layer_norms = calculate_per_layer_frobenius_norm(model)
frob_norm_diff = calculate_per_layer_frobenius_norm_diff(original_model, model)

print("Frobenius Norm:")
print(f"  Original:", original_per_layer_norms)
print(f"  CUR     :", cur_per_layer_norms)
print(f"  Diff.   :", frob_norm_diff)
