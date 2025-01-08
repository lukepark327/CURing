# autopep8: off
import os
import sys
import copy
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

from curlora import CURLoRALinear


# Add the parent directory of CURing to sys.path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../CURing")))

import cur_models
from cur_models import CURLinear, rebuild_model_with_W
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
    log_dir = f"runs/UUID_curlora_{timestamp}"
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
tokenizer = AutoTokenizer.from_pretrained(load_path, padding_side='left')
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


# CURLoRA


# TODO: argparse
target_layers = [25, 26, 27, 24, 28, 23, 22, 29, 20, 21]
ffn_module_names = ["gate_proj"]
attn_module_names = ["q_proj", "k_proj"]

# Create a copy of the model to modify
modified_model = copy.deepcopy(model)


# Freeze
for param in modified_model.parameters():
    param.requires_grad = False


# Add CURLoRA each layer
for layer_index in target_layers:
    layer = model.model.layers[layer_index]
    modified_layer = modified_model.model.layers[layer_index]

    # Process FFN modules
    for name in ffn_module_names:
        module = getattr(layer.mlp, name)
        weight = module.weight.data
        bias = module.bias.data.clone() if module.bias is not None else None

        curlora_linear = CURLoRALinear(weight, bias=bias, rank=256)
        # Replace the module in the modified model
        setattr(modified_layer.mlp, name, curlora_linear)

    # Process Attention modules
    for name in attn_module_names:
        module = getattr(layer.self_attn, name)
        weight = module.weight.data
        bias = module.bias.data.clone() if module.bias is not None else None

        curlora_linear = CURLoRALinear(weight, bias=bias, rank=256)
        # Replace the module in the modified model
        setattr(modified_layer.self_attn, name, curlora_linear)


# Free GPU mem
model.to('cpu')
del model
gc.collect()
torch.cuda.empty_cache()

modified_model.to(device)
model = modified_model
model.train()


def freeze_model_except_U_matrices(model):
    for name, param in model.named_parameters():
        if 'U' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


freeze_model_except_U_matrices(model)


# print(model)
# print("=" * 64)
# # for name, module in model.named_modules():
# #     print(f"Module: {name}, Type: {type(module)}")
# # print("=" * 64)
# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
# exit(1)


# Set batch sizes
micro_batch_size = 16
batch_size = 16
gradient_accumulation_steps = batch_size // micro_batch_size

num_dataset = 1024
num_validation_steps = num_dataset // batch_size


# Dataset
raw_dataset_path = "./benchmark/uuid_raw_dataset"
dataset = load_from_disk(raw_dataset_path)


def tokenize_function(examples):
    full_texts = [inp + outp for inp,
                  outp in zip(examples["input_text"], examples["output_uuid"])]
    full_tokenization = tokenizer(
        full_texts, truncation=True, padding='max_length', max_length=128)
    input_tokenization = tokenizer(
        examples["input_text"], truncation=True, padding='max_length', max_length=64)
    output_tokenization = tokenizer(
        examples["output_uuid"], truncation=True, padding='max_length', max_length=64)
    return {
        # "input_text": examples["input_text"],
        # "output_uuid": examples["output_uuid"],
        "input_ids": full_tokenization["input_ids"],
        "attention_mask": full_tokenization["attention_mask"],
        "eval_input_ids": input_tokenization["input_ids"],
        "eval_attention_mask": input_tokenization["attention_mask"],
        "output_eval_ids": output_tokenization["input_ids"],
    }


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["input_text", "output_uuid"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors='pt'
)

train_dataloader = DataLoader(
    tokenized_dataset['train'],
    batch_size=micro_batch_size,
    collate_fn=data_collator,
    shuffle=True
)

validation_dataloader = DataLoader(
    tokenized_dataset['validation'],
    batch_size=micro_batch_size,
    collate_fn=data_collator,
    shuffle=False
)


# Function to evaluate validation loss


# Token-level
# def evaluate_accuracy(model, dataloader, task_name=None, eval_steps=num_validation_steps):
#     model.eval()

#     total_matches = 0
#     total_tokens = 0

#     with torch.no_grad():
#         progress_bar = tqdm(enumerate(dataloader), total=eval_steps, desc=f"Validate {task_name}")
#         for step, batch in progress_bar:
#             if eval_steps is not None and step >= eval_steps:
#                 break

#             input_ids = batch["eval_input_ids"].to(device)
#             attention_mask = batch["eval_attention_mask"].to(device)
#             reference_ids = batch["output_eval_ids"].to(device)

#             output_ids = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=64,
#                 do_sample=False,
#                 pad_token_id=tokenizer.pad_token_id
#             )

#             prompt_len = input_ids.shape[1]
#             ref_len = reference_ids.shape[1]
#             pred_ids = output_ids[:, prompt_len:prompt_len + ref_len]

#             # Calculate character-level accuracy
#             matches = (pred_ids == reference_ids).sum()
#             total = reference_ids.numel()
#             total_matches += matches.item()
#             total_tokens += total

#             progress_bar.set_postfix({
#                 "Accuracy": total_matches / total_tokens * 100 if total_tokens > 0 else 0.0
#             })

#     return total_matches / total_tokens * 100 if total_tokens > 0 else 0.0


# Char-level
def evaluate_accuracy(model, dataloader, task_name=None, eval_steps=num_validation_steps):
    model.eval()

    total_correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader),
                            total=eval_steps, desc=f"Validate {task_name}")
        for step, batch in progress_bar:
            if step >= eval_steps:
                break

            prompt_inputs = {
                "input_ids": batch["eval_input_ids"].to(device),
                "attention_mask": batch["eval_attention_mask"].to(device),
            }

            # Generate the output UUIDs
            output_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            # Decode the texts
            reference_texts = tokenizer.batch_decode(
                batch["output_eval_ids"], skip_special_tokens=True)
            reference_len = len(reference_texts[0])
            prompt_len = prompt_inputs["input_ids"].shape[1]
            generated_texts = tokenizer.batch_decode(
                output_ids[:, prompt_len:], skip_special_tokens=True)
            generated_texts = [(gt.strip())[:reference_len]
                               for gt in generated_texts]

            # Calculate character-level accuracy
            for g, r in zip(generated_texts, reference_texts):
                g = g.strip()
                r = r.strip()
                # Calculate character matches
                matches = sum(1 for gc, rc in zip(g, r) if gc == rc)
                total_correct_chars += matches
                total_chars += max(len(g), len(r))

            # Update progress bar with character-level accuracy
            progress_bar.set_postfix({
                "Accuracy": total_correct_chars / total_chars * 100 if total_chars > 0 else 0.0
            })

    return total_correct_chars / total_chars * 100 if total_chars > 0 else 0.0


# main


# Define optimizer and scheduler
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    # LR Reference: The Unreasonable Ineffectiveness of the Deeper Layers
    # lr=5e-4
    # lr=3e-6  # Mistral
    # lr=3e-4  # Llama, Qwen
    lr=3e-4
)
num_epochs = 1000
total_steps = num_epochs * num_dataset // batch_size
warmup_steps = 100
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


# Fine-tuning loop
# Initialize TensorBoard SummaryWriter
writers = {}
writers['train_UUID'] = SummaryWriter(
    log_dir=os.path.join(log_dir, 'train_UUID'))
writers['validate_UUID'] = SummaryWriter(
    log_dir=os.path.join(log_dir, 'validate_UUID'))

model.train()
optimizer.zero_grad()
global_step = 0  # Initialize a global step counter


# First Validation
eval_result = evaluate_accuracy(
    model, validation_dataloader,
    task_name="UUID", eval_steps=num_validation_steps
)
print(f"\nValidation UUID Accuracy: {eval_result}")
# Log validation loss and perplexity to TensorBoard
writers['validate_UUID'].add_scalar(
    'Validation/Accuracy', eval_result, global_step)


for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(train_dataloader),
                        total=num_dataset // batch_size, desc=f"Epoch {epoch+1}/{num_epochs}")

    for step, batch in progress_bar:
        if step >= num_dataset // batch_size:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Get student outputs with hidden states
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True
        )

        # loss
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Log the training loss to TensorBoard
        writers['train_UUID'].add_scalar(
            'Train/Loss', loss.item(), global_step)
        progress_bar.set_postfix({"Loss": loss.item()})

        global_step += 1  # Increment the global step counter

    # Validation
    if (epoch+1) % 10 == 0:
        eval_result = evaluate_accuracy(
            model, validation_dataloader,
            task_name="UUID", eval_steps=num_validation_steps
        )
        print(f"\nValidation UUID Accuracy: {eval_result}")
        # Log validation loss and perplexity to TensorBoard
        writers['validate_UUID'].add_scalar(
            'Validation/Accuracy', eval_result, global_step)

        model.train()

    if save_path:
        # Save the fine-tuned model
        # model.save_pretrained(save_path)
        torch.save(model, os.path.join(save_path, 'model.pt'))
        tokenizer.save_pretrained(save_path)


# Close the TensorBoard writers
for writer in writers.values():
    writer.close()
