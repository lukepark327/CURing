# python benchmark/baseline_lora.py \
#     --load_path="./cur_decomposed_models/C128_N10_R256_20241207_044514"
#     --save_path "./cur_healed_models_lora"


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
from datasets import load_dataset
import gc

from lm_eval.models import huggingface
from lm_eval import simple_evaluate

from peft import LoraConfig, get_peft_model, TaskType


# Add the parent directory of CURing to sys.path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../CURing")))

import cur_models
from cur_models import CURLinear, rebuild_model_with_W
# autopep8: on


os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description="Benchmarking CUR-Decomposed Models")

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
    log_dir = f"runs/lora_{timestamp}"
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


# Teacher
teacher_model_name = "meta-llama/Llama-3.1-8B"
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False  # Freeze teacher model parameters


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
target_layers = [25, 26, 27, 24, 28, 23, 22, 29, 20, 21, 19, 18, 17, 30, 16, 11, 10, 13, 14, 15]

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
lora_config_q = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules_q
)
lora_config_k = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules_k
)
lora_config_gate = LoraConfig(
    r=8,
    lora_alpha=16,
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

num_validation_steps = 4096 // batch_size
dataset = {
    'train': load_dataset('c4', 'en', split='train', streaming=True).skip(128),
    'validation_c4': load_dataset('c4', 'en', split='validation', streaming=True).take(min(4096, 364608)),
    'validation_wikitext': load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation', streaming=False).take(min(4096, 3760)),
}


def tokenize_function_text(examples):
    # TODO: For wikitext dataset, removing empty strings and texts start with "= =".
    return tokenizer(
        examples['text'],
        return_special_tokens_mask=True,
        max_length=128,
        truncation=True
    )


tokenized_dataset = {
    'train': dataset['train'].map(
        tokenize_function_text,
        batched=True,
        remove_columns=dataset['train'].features.keys()
    ),
    'validation_c4': dataset['validation_c4'].map(
        tokenize_function_text,
        batched=True,
        remove_columns=dataset['validation_c4'].features.keys()
    ),
    'validation_wikitext': dataset['validation_wikitext'].map(
        tokenize_function_text,
        batched=True,
        remove_columns=dataset['validation_wikitext'].features.keys()
    )
}

data_collator_lm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors='pt'
)

train_dataloader = DataLoader(
    tokenized_dataset['train'],
    batch_size=micro_batch_size,
    collate_fn=data_collator_lm,
)

validation_dataloaders = {
    'c4': DataLoader(
        tokenized_dataset['validation_c4'],
        batch_size=micro_batch_size,
        collate_fn=data_collator_lm,
    ),
    'wikitext': DataLoader(
        tokenized_dataset['validation_wikitext'],
        batch_size=micro_batch_size,
        collate_fn=data_collator_lm,
    ),
    'boolq': None,
    'mmlu': None,
}


# Function to evaluate validation loss


def evaluate_lm(model, dataloader, task_name=None, eval_steps=num_validation_steps):
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader),
                            total=eval_steps, desc=f"Validate {task_name}")
        for step, batch in progress_bar:
            if step >= eval_steps:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            total_loss += loss.item() * attention_mask.sum().item()
            total_tokens += attention_mask.sum().item()

            progress_bar.set_postfix({"Loss": loss.item()})

    # Return
    # Compute and return the appropriate metric
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return {'loss': avg_loss, 'perplexity': perplexity.item()}


def evaluate_classification(model, task_name=None, limit=None, fewshot=0):
    outputs = simple_evaluate(
        model=huggingface.HFLM(
            pretrained=model,
            backend='causal',
            tokenizer=tokenizer,
            # batch_size
        ),
        tasks=[task_name],
        limit=limit,
        num_fewshot=fewshot
    )
    accuracy = outputs['results'][task_name]['acc,none']
    stderr = outputs['results'][task_name]['acc_stderr,none']
    return {'accuracy': accuracy, 'stderr': stderr}


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
num_epochs = 1
total_steps = 2001
warmup_steps = 100
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


# Create a mapping of task types for each validation dataset
validation_tasks = {
    'c4': {'dataloader': validation_dataloaders['c4'], 'task_type': 'lm', 'eval_steps': min(4096, 364608) // batch_size},
    'wikitext': {'dataloader': validation_dataloaders['wikitext'], 'task_type': 'lm', 'eval_steps': min(4096, 3760) // batch_size},
    'boolq': {'task_type': 'classification', 'limit': min(4096, 3270), 'fewshot': 0},
    # 57 categiries
    'mmlu': {'task_type': 'classification', 'limit': min(4096, 32), 'fewshot': 5},
}


# FT
# Define loss functions
kl_kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
mse_loss_fn = torch.nn.MSELoss()

# Define the temperature and weighting factors
T = 10.0  # Temperature for KD loss
alpha = 0.1  # Weight for standard cross-entropy loss
# Weight for KL KD loss (0.1/0.3/0.5 from https://arxiv.org/pdf/2204.00408)
beta = 0.0
gamma = 0.9  # Weight for MSE loss over hidden states
# alpha, beta, gamma = (x / (alpha + beta + gamma) for x in (alpha, beta, gamma))
# total_loss = alpha * outputs.loss + beta * kd_loss + gamma * mse_loss

# Fine-tuning loop
# Initialize TensorBoard SummaryWriter
writers = {}
writers['train'] = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
for task_name in validation_dataloaders.keys():
    task_log_dir = os.path.join(log_dir, task_name)
    writers[task_name] = SummaryWriter(log_dir=task_log_dir)

model.train()
optimizer.zero_grad()
global_step = 0  # Initialize a global step counter

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(train_dataloader),
                        total=total_steps, desc=f"Epoch {epoch+1}/{num_epochs}")

    for step, batch in progress_bar:
        if step >= total_steps:
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

        # Get teacher outputs with hidden states (no gradients needed)
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Compute the KD loss (KL divergence)
        kd_loss = 0.0
        if beta != 0.0:
            student_logits = outputs.logits / T
            teacher_logits = teacher_outputs.logits / T
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            kd_loss = kl_kd_loss_fn(
                student_log_probs, teacher_probs) * (T * T)

        # Compute MSE loss over hidden states
        mse_loss = 0
        num_layers = len(outputs.hidden_states) - 1
        for i in range(1, num_layers):  # Exclude embeddings and final layer
            student_hidden = outputs.hidden_states[i]
            teacher_hidden = teacher_outputs.hidden_states[i]
            mse_loss += mse_loss_fn(student_hidden, teacher_hidden)
        # Average MSE loss over layers
        mse_loss = mse_loss / (num_layers - 1)

        # Combine losses
        total_loss = alpha * outputs.loss + beta * kd_loss + gamma * mse_loss
        loss = total_loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Log the training loss to TensorBoard
        writers['train'].add_scalar('Train/Loss', loss.item(), global_step)
        progress_bar.set_postfix({"Loss": loss.item()})

        # validation_interval
        if step % 100 == 0:
            for task_name, task_info in validation_tasks.items():

                if task_info['task_type'] == 'lm':
                    eval_result = evaluate_lm(
                        model, task_info['dataloader'],
                        task_name=task_name, eval_steps=task_info['eval_steps']
                    )
                    val_loss = eval_result['loss']
                    perplexity = eval_result['perplexity']
                    print(
                        f"\nValidation {task_name} Loss: {val_loss}, Perplexity: {perplexity}")

                    # Log validation loss and perplexity to TensorBoard
                    writers[task_name].add_scalar(
                        'Validation/Loss', val_loss, global_step)
                    writers[task_name].add_scalar(
                        'Validation/Perplexity', perplexity, global_step)

                elif task_info['task_type'] == 'classification':
                    eval_result = evaluate_classification(
                        model,
                        task_name=task_name, limit=task_info['limit'], fewshot=task_info['fewshot']
                    )
                    accuracy = eval_result['accuracy']
                    stderr = eval_result['stderr']
                    print(
                        f"\nValidation {task_name} Accuracy: {accuracy} ({stderr})")

                    # Log validation accuracy w/ stderr to TensorBoard
                    writers[task_name].add_scalar(
                        'Validation/Accuracy', accuracy, global_step)
                    writers[task_name].add_scalar(
                        'Validation/Acc_StdError', stderr, global_step)

            model.train()

            if save_path:
                # Save the fine-tuned model
                # model.save_pretrained(save_path)
                torch.save(model, os.path.join(save_path, 'model.pt'))
                tokenizer.save_pretrained(save_path)

        global_step += 1  # Increment the global step counter

# Close the TensorBoard writers
for writer in writers.values():
    writer.close()

if save_path:
    # Save the fine-tuned model
    # model.save_pretrained(save_path)
    torch.save(model, os.path.join(save_path, 'model.pt'))
    tokenizer.save_pretrained(save_path)
