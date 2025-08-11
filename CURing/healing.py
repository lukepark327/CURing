import os
from copy import copy
import argparse
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import set_seed, get_cosine_schedule_with_warmup, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from lm_eval.models import huggingface
from lm_eval import simple_evaluate

import numpy as np
import pandas as pd
from tqdm import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
# import seaborn as sns


from cur_models import CURLinear, ActivationAccumulator, activate_capture_for_all_ActivationAccumulator, deactivate_capture_for_all_ActivationAccumulator, reset_activations_for_all_ActivationAccumulator, activate_capture_for_all_CURLinear_modules, deactivate_capture_for_all_CURLinear_modules, reset_activations_for_all_CURLinear_modules


os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description="Healing CUR-Decomposed Models")

# Model and Paths
parser.add_argument('--teacher_model_name', type=str,
                    # "neuralmagic/Sparse-Llama-3.1-8B-2of4"
                    # "vuiseng9/Meta-Llama-3.1-8B-wanda-unstructured-0.5"  # sparsity 0.5
                    default="meta-llama/Llama-3.1-8B",
                    help='Name or path of the teacher pre-trained model.')
parser.add_argument('--load_path', type=str, default=None,
                    help='Directory path from where the modified model is loaded.')
parser.add_argument('--save_path', type=str, default="./cur_healed_models",
                    help='Directory path where the fine-tuned model and outputs will be saved.')
parser.add_argument('--model_save', action='store_true', default=False,
                    help='Set this flag to save the model (default: False).')
parser.add_argument('--device', type=str, default="cuda",
                    help='Device to run the computations on (e.g., "cpu", "cuda").')

# Data Loading and Preprocessing
parser.add_argument('--train_dataset', type=str, default='c4',
                    help='Name of the training dataset.')
parser.add_argument('--train_dataset_category', type=str, default='en',
                    help='Name of the training dataset category.')
# Fixed.
# parser.add_argument('--validation_dataset', type=str, default='c4',
#                     help='Name of the validation dataset.')
# parser.add_argument('--validation_dataset_category', type=str, default='en',
#                     help='Name of the validation dataset category.')
parser.add_argument('--train_skip',
                    # = num_calibration_steps
                    type=int, default=128,
                    help='Number of training examples to skip.')

# Training Parameters
parser.add_argument('--micro_batch_size', type=int, default=16,
                    help='Batch size for micro-batches.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Total batch size (used for gradient accumulation).')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='Number of training epochs.')
parser.add_argument('--total_steps', type=int, default=2001,
                    help='Total number of training steps.')
parser.add_argument('--warmup_steps', type=int, default=100,
                    help='Number of warmup steps for the learning rate scheduler.')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate for the optimizer.')
parser.add_argument('--max_length', type=int, default=128,
                    help='Max length per calibration dataset.')

# Validation Parameters
parser.add_argument('--validation_interval', type=int, default=100,
                    help='Interval of validation steps.')
parser.add_argument('--num_validation_steps', type=int, default=4096,
                    help='Number of validation steps.')

# Loss Function Parameters
parser.add_argument('--T', type=float, default=10.0,
                    help='Temperature for knowledge distillation loss.')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='Weight for standard cross-entropy loss.')
parser.add_argument('--beta', type=float, default=0.0,
                    help='Weight for KL divergence KD loss.')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='Weight for MSE loss over hidden states.')

# Module Names to Heal
parser.add_argument('--ffn_module_names', nargs='+', default=["gate_proj"],
                    help='List of FFN module names to heal.')
parser.add_argument('--attn_module_names', nargs='+', default=["q_proj", "k_proj"],
                    help='List of attention module names to heal.')

# Miscellaneous
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')
parser.add_argument('--log_dir', type=str, default=None,
                    help='Directory for TensorBoard logs.')

args = parser.parse_args()


# Set random seed for reproducibility
set_seed(args.seed)


# Set model name and paths
teacher_model_name = args.teacher_model_name
print(f"Teacher Model: {teacher_model_name}")
# load_path
if args.load_path:
    child_path = args.load_path
else:
    with open("./cur_decomposed_models/latest.txt", "r") as f:
        child_path = f.read().strip()
load_path = f"./cur_decomposed_models/{child_path}"
print(f"Child Model: {args.load_path}, Loaded from {load_path}")
# Create a timestamped directory to avoid overwriting
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
save_path = os.path.join(save_path, child_path)
os.makedirs(save_path, exist_ok=True)
# log
if args.log_dir:
    log_dir = args.log_dir
else:
    log_dir = f"runs/{child_path}"
print(f"Logging at {log_dir}")

# Handle device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == args.device, \
    f"Device does not match the expected one. {args.device} != {device}"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(load_path)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    # print(f"The tokenizer.pad_token set as a {tokenizer.eos_token}")
    # TODO: use other token?


# Define the CURLinear class to avoid error/warning
CURLinear


# Load the model with custom class
model = torch.load(os.path.join(load_path, 'model.pt'), weights_only=False)
model.to(device)


def freeze_model_except_U_matrices(model):
    for name, param in model.named_parameters():
        if 'U' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


freeze_model_except_U_matrices(model)


# Load teacher model
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name).to(device)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False  # Freeze teacher model parameters


# Set batch sizes and gradient accumulation
micro_batch_size = args.micro_batch_size
batch_size = args.batch_size
gradient_accumulation_steps = batch_size // micro_batch_size

num_validation_steps = args.num_validation_steps // batch_size
dataset = {
    'train': load_dataset(args.train_dataset, args.train_dataset_category, split='train', streaming=True).skip(args.train_skip),
    'validation_c4': load_dataset('c4', 'en', split='validation', streaming=True).take(min(args.num_validation_steps, 364608)),
    'validation_wikitext': load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation', streaming=False).take(min(args.num_validation_steps, 3760)),
    # 'validation_boolq': load_dataset('google/boolq', split='validation', streaming=False).take(min(args.num_validation_steps, 3270)),
    # 'validation_mmlu': load_dataset('cais/mmlu', 'all', split='validation', streaming=False).take(min(args.num_validation_steps, 32)),
}


def tokenize_function_text(examples):
    # TODO: For wikitext dataset, removing empty strings and texts start with "= =".
    return tokenizer(
        examples['text'],
        return_special_tokens_mask=True,
        max_length=args.max_length,
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


# main


# Define optimizer and scheduler
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    # LR Reference: The Unreasonable Ineffectiveness of the Deeper Layers
    # lr=5e-4
    # lr=3e-6  # Mistral
    # lr=3e-4  # Llama, Qwen
    lr=args.learning_rate
)
num_epochs = args.num_epochs
total_steps = args.total_steps
warmup_steps = args.warmup_steps
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


# Validation


max_output_dim = None


def evaluate_lm(model, teacher_model, dataloader, task_name=None, eval_steps=num_validation_steps, draw_heatmap=False):
    global wrapped_modules

    if draw_heatmap:
        # Reset activations
        # Teacher
        reset_activations_for_all_ActivationAccumulator(wrapped_modules)
        activate_capture_for_all_ActivationAccumulator(wrapped_modules)
        # Student
        reset_activations_for_all_CURLinear_modules(model)
        activate_capture_for_all_CURLinear_modules(model)

    # Evalute
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

            # Forward pass with student model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Forward pass with teacher model
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Loss calculation
            loss = outputs.loss
            total_loss += loss.item() * attention_mask.sum().item()
            total_tokens += attention_mask.sum().item()

            progress_bar.set_postfix({"Loss": loss.item()})

    if draw_heatmap:
        # Disable activation capturing after evaluation
        # Teacher
        deactivate_capture_for_all_CURLinear_modules(model)
        # Student
        deactivate_capture_for_all_ActivationAccumulator(wrapped_modules)

    if draw_heatmap:
        # Visualize activations
        # Collect activations from student model
        student_activations_R = {}
        student_row_indices = {}
        for name, module in model.named_modules():
            if isinstance(module, CURLinear):
                # Generate key similar to wrapped_modules
                parts = name.split('.')
                if 'layers' in parts:
                    idx = parts.index('layers')
                    layer_index = parts[idx + 1]
                    rest = parts[idx + 2:]
                    key = f"layer_{layer_index}_{'_'.join(rest)}"
                else:
                    key = name.replace('.', '_')

                # Store activations
                if module.activation_R_accum is not None and module.nsamples > 0:
                    activation_R = (module.activation_R_accum /
                                    module.nsamples).cpu().numpy()
                    student_activations_R[key] = activation_R

                # Store indices
                if hasattr(module, 'row_indices'):
                    student_row_indices[key] = module.row_indices

        # Determine the maximum output dimension
        global max_output_dim
        if max_output_dim is None:
            max_output_dim = 0
            for key, wrapped_module in wrapped_modules.items():
                teacher_output_activation = wrapped_module.get_mean_output_activation().cpu().numpy()
                output_dim = teacher_output_activation.shape[0]
                if output_dim > max_output_dim:
                    max_output_dim = output_dim

        # Generate and save heatmaps
        # Initialize lists to collect activations and labels
        teacher_activations_R_list = []
        student_activations_R_list = []
        layer_labels = []
        for key, wrapped_module in wrapped_modules.items():
            student_activation_R = student_activations_R.get(key)
            row_indices = student_row_indices.get(key)
            # student_activation_C = student_activations_C.get(key)
            # col_indices = student_col_indices.get(key)

            if student_activation_R is None:
                continue
            if row_indices is None:
                continue

            # Extract activations corresponding to R from the teacher model
            teacher_output_activation = wrapped_module.get_mean_output_activation().cpu().numpy()
            output_dim = teacher_output_activation.shape[0]

            # Pad teacher activation to max_output_dim
            padded_teacher_activation_R = np.full((max_output_dim,), np.nan)
            padded_teacher_activation_R[:output_dim] = teacher_output_activation

            # Create a padded student activation array
            padded_student_activation_R = np.full((max_output_dim,), np.nan)
            adjusted_row_indices = [
                idx for idx in row_indices if idx < max_output_dim]
            padded_student_activation_R[adjusted_row_indices] = student_activation_R[:len(
                adjusted_row_indices)]

            # Append activations and labels to the lists
            teacher_activations_R_list.append(padded_teacher_activation_R)
            student_activations_R_list.append(padded_student_activation_R)
            layer_labels.append(f"{key}")

        current_time = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Heatmap
        # Combine activations into arrays
        # Each activation is of shape (max_output_dim,)
        # Stack activations to shape (2 * num_layers, max_output_dim)
        teacher_activations_R_array = np.vstack(teacher_activations_R_list)
        student_activations_R_array = np.vstack(student_activations_R_list)

        # Create a combined data array by interleaving teacher and student activations
        data_array = []
        yticklabels = []
        for layer_label, teacher_act, student_act in zip(layer_labels, teacher_activations_R_array, student_activations_R_array):
            data_array.append(teacher_act)
            yticklabels.append(f'{layer_label} Teacher')
            data_array.append(student_act)
            yticklabels.append(f'{layer_label} Student')

        # Shape: (2 * num_layers, max_output_dim)
        data_array = np.vstack(data_array)

        # Build a mapping from layer labels to their index in layer_labels
        layer_label_to_idx = {label: idx for idx,
                              label in enumerate(layer_labels)}

        # Create a mask for the selected indices (row_indices)
        selected_mask = np.zeros_like(data_array, dtype=bool)
        for layer_label, row_indices in student_row_indices.items():
            if layer_label in layer_label_to_idx:
                idx = layer_label_to_idx[layer_label]
                # Since yticklabels have teacher and student interleaved, the teacher activation is at row 2*idx
                selected_mask[2 * idx, row_indices] = True
                # Student activation is at row 2*idx + 1
                selected_mask[2 * idx + 1, row_indices] = True

        # Separate data into selected and non-selected activations
        selected_data = np.ma.masked_where(~selected_mask, data_array)
        non_selected_data = np.ma.masked_where(selected_mask, data_array)

        # Normalize selected_data and non_selected_data separately
        selected_vmin = np.min(selected_data)
        selected_vmax = np.max(selected_data)
        non_selected_vmin = np.min(non_selected_data)
        non_selected_vmax = np.max(non_selected_data)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(32, 18))

        # Create copies of the colormaps
        # cmap_selected = copy(plt.cm.Reds)
        # cmap_non_selected = copy(plt.cm.Blues)
        cmap_selected = copy(plt.cm.autumn)
        cmap_non_selected = copy(plt.cm.winter)
        # Set masked values to be transparent
        cmap_selected.set_bad(color='none')
        cmap_non_selected.set_bad(color='none')

        # Plot non-selected data first
        im_non_selected = ax.imshow(
            non_selected_data,
            aspect='auto',
            cmap=cmap_non_selected,
            vmin=non_selected_vmin,
            vmax=non_selected_vmax,
            interpolation='none'
        )

        # Plot selected data on top
        im_selected = ax.imshow(
            selected_data,
            aspect='auto',
            cmap=cmap_selected,
            vmin=selected_vmin,
            vmax=selected_vmax,
            interpolation='none'
        )

        # Set yticklabels
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

        # Set labels and title
        ax.set_title('Activation R Comparison Across Layers')
        ax.set_xlabel('Features (Output Dimension Indices)')
        ax.set_ylabel('Layers')

        # Create separate colorbars
        divider = make_axes_locatable(ax)
        cax_selected = divider.append_axes("right", size="5%", pad=0.05)
        cax_non_selected = divider.append_axes("right", size="5%", pad=0.15)

        # Add colorbars
        cb_selected = fig.colorbar(im_selected, cax=cax_selected)
        cb_selected.set_label('Selected Activation Value')
        cb_non_selected = fig.colorbar(im_non_selected, cax=cax_non_selected)
        cb_non_selected.set_label('Non-selected Activation Value')

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f'activations_{current_time}.png'),
            dpi=256
        )
        plt.close()

        # Compute Frobenius norm differences
        differences_list = []
        for key, teacher_act, student_act in zip(layer_labels, teacher_activations_R_array, student_activations_R_array):
            # Mask unmatched rows in the student activations
            # Mask where student activation is valid
            valid_mask = ~np.isnan(student_act)
            valid_teacher_act = teacher_act[valid_mask]
            valid_student_act = student_act[valid_mask]

            # Compute Frobenius norm of the difference
            teacher_norm = np.sqrt(np.sum(valid_teacher_act ** 2))
            student_norm = np.sqrt(np.sum(valid_student_act ** 2))
            diff_norm = np.sqrt(
                np.sum((valid_teacher_act - valid_student_act) ** 2))

            # Add to differences list
            differences_list.append({
                'Layer': key,
                'Frobenius Norm Teacher': teacher_norm,
                'Frobenius Norm Student': student_norm,
                'Frobenius Norm Difference': diff_norm
            })
        # Save differences to CSV
        csv_file_path = os.path.join(
            save_path, f'activations_{current_time}.csv')
        df = pd.DataFrame(differences_list)
        df.to_csv(csv_file_path, index=False)

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


# Initialize a dictionary to store ActivationAccumulator instances
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


def register_hooks(model, wrapped_modules):
    for layer_index, layer in enumerate(model.model.layers):
        # Process FFN modules
        for name in ffn_module_names:
            if hasattr(layer.mlp, name):
                module = getattr(layer.mlp, name)
                wrapped_module = ActivationAccumulator(module)
                wrapped_module.register_hook()
                key = f"layer_{layer_index}_mlp_{name}"
                wrapped_modules[key] = wrapped_module

        # Process Attention modules
        for name in attn_module_names:
            if hasattr(layer.self_attn, name):
                module = getattr(layer.self_attn, name)
                wrapped_module = ActivationAccumulator(module)
                wrapped_module.register_hook()
                key = f"layer_{layer_index}_self_attn_{name}"
                wrapped_modules[key] = wrapped_module


# Register hooks for target layers
register_hooks(teacher_model, wrapped_modules)


# FT
# Define loss functions
kl_kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
mse_loss_fn = torch.nn.MSELoss()

# Define the temperature and weighting factors
T = args.T  # Temperature for KD loss
alpha = args.alpha  # Weight for standard cross-entropy loss
# Weight for KL KD loss (0.1/0.3/0.5 from https://arxiv.org/pdf/2204.00408)
beta = args.beta
gamma = args.gamma  # Weight for MSE loss over hidden states
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

# TODO
# TODO: lm_eval
# Create a mapping of task types for each validation dataset
# validation_tasks = {
#     'c4': {'dataloader': validation_dataloaders['c4'], 'task_type': 'lm', 'eval_steps': min(args.num_validation_steps, 364608) // batch_size},
#     'wikitext': {'dataloader': validation_dataloaders['wikitext'], 'task_type': 'lm', 'eval_steps': min(args.num_validation_steps, 3760) // batch_size},
#     'boolq': {'task_type': 'classification', 'limit': min(args.num_validation_steps, 3270), 'fewshot': 0},
#     # 57 categiries
#     'mmlu': {'task_type': 'classification', 'limit': min(args.num_validation_steps, 32), 'fewshot': 5},
# }
validation_tasks = {}

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

        del outputs.hidden_states
        del teacher_outputs.hidden_states
        torch.cuda.empty_cache()

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

        if step % args.validation_interval == 0:
            for task_name, task_info in validation_tasks.items():

                if task_info['task_type'] == 'lm':
                    eval_result = evaluate_lm(
                        model, teacher_model, task_info['dataloader'],
                        task_name=task_name, eval_steps=task_info['eval_steps'],
                        draw_heatmap=(task_name == 'c4')
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

            # TODO: disabling backup - time eval
            if args.model_save:
                # Save the fine-tuned model
                # model.save_pretrained(save_path)
                torch.save(model, os.path.join(save_path, 'model.pt'))
                tokenizer.save_pretrained(save_path)

        global_step += 1  # Increment the global step counter

    torch.cuda.empty_cache()


# Close the TensorBoard writers
for writer in writers.values():
    writer.close()

if args.model_save:
    # Save the fine-tuned model
    # model.save_pretrained(save_path)
    torch.save(model, os.path.join(save_path, 'model.pt'))
    tokenizer.save_pretrained(save_path)


def remove_hooks(wrapped_modules):
    for wrapped_module in wrapped_modules.values():
        wrapped_module.remove_hook()


# Remove all hooks after collection
remove_hooks(wrapped_modules)


# # Generate text with the fine-tuned model
# prompt = "Once upon a time"
# generated_text = generate_text(model, prompt)
# print("Fine-tuned model output:", generated_text)
