# e.g.) python benchmark/baseline_curing.py --load_path="./models/decomposed/C128_N10_R256_20241202_095839"


# autopep8: off
import os
import sys
import argparse

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_dataset

from lm_eval.models import huggingface
from lm_eval import simple_evaluate


# Add the parent directory of CURing to sys.path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../CURing")))

import cur_models
from cur_models import CURLinear
# autopep8: on


os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(
    description="Benchmarking CUR-Decomposed Models")

parser.add_argument('--load_path', type=str, default=None,
                    help='Directory path from where the modified model is loaded.')

args = parser.parse_args()


# Set random seed for reproducibility
set_seed(42)


# Load the original model and tokenizer
assert args.load_path is not None, \
    f"No `load_path` defined."

load_path = f"{args.load_path}"
print(f"Loaded from {load_path}")

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

# Set model to evaluation mode
model.eval()


# Set batch sizes
micro_batch_size = 16
batch_size = 16

num_validation_steps = 4096 // batch_size
dataset = {
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


# Create a mapping of task types for each validation dataset
validation_tasks = {
    'c4': {'dataloader': validation_dataloaders['c4'], 'task_type': 'lm', 'eval_steps': min(4096, 364608) // batch_size},
    'wikitext': {'dataloader': validation_dataloaders['wikitext'], 'task_type': 'lm', 'eval_steps': min(4096, 3760) // batch_size},
    'boolq': {'task_type': 'classification', 'limit': min(4096, 3270), 'fewshot': 0},
    # 57 categiries
    'mmlu': {'task_type': 'classification', 'limit': min(4096, 32), 'fewshot': 5},
}

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

    elif task_info['task_type'] == 'classification':
        eval_result = evaluate_classification(
            model,
            task_name=task_name, limit=task_info['limit'], fewshot=task_info['fewshot']
        )
        accuracy = eval_result['accuracy']
        stderr = eval_result['stderr']
        print(
            f"\nValidation {task_name} Accuracy: {accuracy} ({stderr})")
