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


from peft_mora import LoraConfig as MoraConfig
from peft_mora import get_peft_model, TaskType


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
    log_dir = f"runs/forgetting_mora_{timestamp}"
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


# LoRA


# TODO: argparse
target_layers = [25, 26, 27, 24, 28, 23, 22, 29, 20, 21]

target_modules_q = []
target_modules_k = []
target_modules_gate = []
for layer in target_layers:
    target_modules_q.extend([f"layers.{layer}.self_attn.q_proj"])
for layer in target_layers:
    target_modules_k.extend([f"layers.{layer}.self_attn.k_proj"])
for layer in target_layers:
    target_modules_gate.extend([f"layers.{layer}.mlp.gate_proj"])

# TODO: r <- argparse
lora_config_q = MoraConfig(
    use_mora=True,
    mora_type=6,  # RoPE
    r=4,  # -> \hat{r}
    # lora_alpha=16,  # not use
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules_q
)
lora_config_k = MoraConfig(
    use_mora=True,
    mora_type=6,  # RoPE
    r=4,  # -> \hat{r}
    # lora_alpha=16,  # not use
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules_k
)
lora_config_gate = MoraConfig(
    use_mora=True,
    mora_type=6,  # RoPE
    r=8,  # -> \hat{r}
    # lora_alpha=16,  # not use
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules_gate
)
model = get_peft_model(model, lora_config_q)
model = get_peft_model(model, lora_config_k)
model = get_peft_model(model, lora_config_gate)
model.train()


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


# Dataset
dataset = {
    'train_mrpc': load_dataset('nyu-mll/glue', 'mrpc', split='train', streaming=False),
    # 'train_hate': load_dataset('ucberkeley-dlab/measuring-hate-speech', split='train', streaming=False),
    'validation_wikitext2': load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation', streaming=False),
}


def tokenize_function_text(examples, key):
    # TODO: For wikitext dataset, removing empty strings and texts start with "= =".
    return tokenizer(
        examples[key],
        return_special_tokens_mask=True,
        max_length=128,
        truncation=True
    )


tokenized_dataset = {
    'train_mrpc': dataset['train_mrpc'].map(
        lambda examples: tokenize_function_text(examples, key='sentence1'),
        batched=True,
        remove_columns=dataset['train_mrpc'].features.keys()
    ),
    # 'train_hate': dataset['train_hate'].map(
    #     lambda examples: tokenize_function_text(examples, key='text'),
    #     batched=True,
    #     remove_columns=dataset['train_hate'].features.keys()
    # ),
    'validation_wikitext2': dataset['validation_wikitext2'].map(
        lambda examples: tokenize_function_text(examples, key='text'),
        batched=True,
        remove_columns=dataset['validation_wikitext2'].features.keys()
    ),
}

data_collator_lm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors='pt'
)

train_dataloaders = {
    'train_mrpc': DataLoader(
        tokenized_dataset['train_mrpc'],
        batch_size=micro_batch_size,
        collate_fn=data_collator_lm,
        shuffle=True
    ),
    # 'train_hate': DataLoader(
    #     tokenized_dataset['train_hate'],
    #     batch_size=micro_batch_size,
    #     collate_fn=data_collator_lm,
    #     shuffle=True
    # ),
}

validation_dataloader = DataLoader(
    tokenized_dataset['validation_wikitext2'],
    batch_size=micro_batch_size,
    collate_fn=data_collator_lm,
    shuffle=False
)


# Function to evaluate validation loss and perplexity


def evaluate(model, dataloader, task_name=None):
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(
            dataloader), desc=f"Validate {task_name}")

        for step, batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Generate outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Loss calculation
            loss = outputs.loss
            batch_tokens = attention_mask.sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            progress_bar.set_postfix({"Loss": loss.item()})

    # Return
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return {'loss': avg_loss, 'perplexity': perplexity.item()}


# main

# Define optimizer and scheduler
optimizers = {
    'train_mrpc': AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        # LR Reference: The Unreasonable Ineffectiveness of the Deeper Layers
        # lr=5e-4
        # lr=3e-6  # Mistral
        # lr=3e-4  # Llama, Qwen
        lr=3e-4
    ),
    # 'train_hate': AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     # LR Reference: The Unreasonable Ineffectiveness of the Deeper Layers
    #     # lr=5e-4
    #     # lr=3e-6  # Mistral
    #     # lr=3e-4  # Llama, Qwen
    #     lr=3e-4
    # ),
}
num_epochs = 100
warmup_steps = 100
schedulers = {
    'train_mrpc': get_cosine_schedule_with_warmup(
        optimizers['train_mrpc'], num_warmup_steps=warmup_steps, num_training_steps=num_epochs *
        len(train_dataloaders['train_mrpc'])
    ),
    # 'train_hate': get_cosine_schedule_with_warmup(
    #     optimizers['train_hate'], num_warmup_steps=warmup_steps, num_training_steps=num_epochs *
    #     len(train_dataloaders['train_hate'])
    # ),
}


# Fine-tuning loop
# Initialize TensorBoard SummaryWriter
writers = {}
for task_name in train_dataloaders.keys():
    task_log_dir = os.path.join(log_dir, task_name)
    writers[task_name] = SummaryWriter(log_dir=task_log_dir)
writers['validation_wikitext2'] = SummaryWriter(
    log_dir=os.path.join(log_dir, 'validation_wikitext2'))

model.train()
for task_name, optimizer in optimizers.items():
    optimizer.zero_grad()
global_step = 0  # Initialize a global step counter


# First Validation
eval_result = evaluate(model, validation_dataloader,
                       task_name="validation_wikitext2")
print(f"\nValidation: {eval_result}")
# Log validation loss and perplexity to TensorBoard
writers['validation_wikitext2'].add_scalar(
    'Validation/Loss', eval_result['loss'], global_step)
writers['validation_wikitext2'].add_scalar(
    'Validation/Perplexity', eval_result['perplexity'], global_step)


# Create a mapping of task types for each train dataset
train_tasks = {
    'train_mrpc': {'dataloader': train_dataloaders['train_mrpc'], 'num_dataset': len(train_dataloaders['train_mrpc'])},
    # 'train_hate': {'dataloader': train_dataloaders['train_hate'], 'num_dataset': len(train_dataloaders['train_hate'])},
}

for task_name, task_info in train_tasks.items():
    optimizer = optimizers[task_name]
    scheduler = schedulers[task_name]

    for epoch in range(num_epochs):
        progress_bar = tqdm(
            enumerate(task_info['dataloader']),
            total=task_info['num_dataset'],
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )

        for step, batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # loss
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log the training loss to TensorBoard
            writers[task_name].add_scalar(
                'Train/Loss', loss.item(), global_step)
            progress_bar.set_postfix({"Loss": loss.item()})

            global_step += 1  # Increment the global step counter

        # Validation
        eval_result = evaluate(model, validation_dataloader,
                               task_name="validation_wikitext2")
        print(f"\nValidation: {eval_result}")
        # Log validation loss and perplexity to TensorBoard
        writers['validation_wikitext2'].add_scalar(
            'Validation/Loss', eval_result['loss'], global_step)
        writers['validation_wikitext2'].add_scalar(
            'Validation/Perplexity', eval_result['perplexity'], global_step)

        model.train()

        if save_path:
            # Save the fine-tuned model
            # model.save_pretrained(save_path)
            torch.save(model, os.path.join(save_path, 'model.pt'))
            tokenizer.save_pretrained(save_path)


# Close the TensorBoard writers
for writer in writers.values():
    writer.close()
