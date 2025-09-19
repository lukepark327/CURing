import os
import argparse
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import set_seed, get_cosine_schedule_with_warmup, DataCollatorForLanguageModeling
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from lm_eval.models import huggingface
from lm_eval import simple_evaluate


from cur_models import CURLinear


# import logging

# logging.disable(logging.CRITICAL)
# logging.getLogger("lm_eval").disabled = True


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
parser.add_argument('--train_dataset', type=str, default='allenai/c4',
                    help='Name of the training dataset.')
parser.add_argument('--train_dataset_category', type=str, default='en',
                    help='Name of the training dataset category.')
parser.add_argument('--train_skip',
                    # = num_calibration_steps
                    type=int, default=256,
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
parser.add_argument('--max_length', type=int, default=4096,
                    help='Max length per dataset.')

# Test Parameters
parser.add_argument('--test_interval', type=int, default=8,
                    help='Interval of test steps.')
parser.add_argument('--num_test_steps', type=int, default=128,
                    help='Number of test steps.')

# Loss Function Parameters
parser.add_argument('--T', type=float, default=1.0,
                    help='Temperature for knowledge distillation loss.')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='Weight for standard cross-entropy loss.')
parser.add_argument('--beta', type=float, default=0.0,
                    help='Weight for KL divergence KD loss.')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='Weight for MSE loss over hidden states.')

# Healing mode: hidden MSE vs. local cov_fast-aligned per-module MSE
parser.add_argument('--healing_mode', type=str, default='hidden_mse',
                    choices=['hidden_mse', 'local_mse', 'lillama'],
                    help='hidden_mse: hidden-state MSE | local_mse: per-module MSE | lillama: Lillama Teacher + Student.')

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
load_path = args.load_path
print(f"Student Model: {load_path}")
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
log_dir = args.log_dir
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
gradient_accumulation_steps = max(1, batch_size // micro_batch_size)
optimizer_step_total = math.ceil(
    args.total_steps / gradient_accumulation_steps)


# shards = [
#     f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(4)
# ]
dataset = {
    'train': load_dataset(
        args.train_dataset, args.train_dataset_category,
        # data_files={"train": shards},
        split='train',
        streaming=True
    ).skip(args.train_skip),
}


def tokenize_function_text(examples):
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
}

data_collator_lm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors='pt'
)


def collate_and_mask(batch):
    out = data_collator_lm(batch)
    if 'labels' in out:
        mask = out['attention_mask'] == 0
        out['labels'][mask] = -100
    return out


train_dataloader = DataLoader(
    tokenized_dataset['train'],
    batch_size=micro_batch_size,
    collate_fn=collate_and_mask,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
)

test_dataloaders = {
    'c4': None,
    'wikitext': None,
    'boolq': None,
    'mmlu': None,
}


# main


# Define optimizer and scheduler
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    # TODO
    # LR Reference: The Unreasonable Ineffectiveness of the Deeper Layers
    # lr=5e-4
    # lr=3e-6  # Mistral
    # lr=3e-4  # Llama, Qwen
    lr=args.learning_rate,
    fused=True,  # TODO
)
num_epochs = args.num_epochs
total_steps = args.total_steps
warmup_steps = min(args.warmup_steps, optimizer_step_total)
if args.healing_mode == 'lillama':
    # Lillama: no scheduler (constant LR)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
else:
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=optimizer_step_total
    )


# Test


def evaluate_lm(model, task_name, limit=None):
    assert task_name in {
        'c4',
        'wikitext',
    }, f"Unsupported task_name: {task_name}"

    seqlen = (
        args.max_length
        or getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", None)
        or 4096
    )

    if task_name == 'c4':
        ds = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.*.json.gz"},
            split="validation",
            streaming=False
        ).select(range(1024))  # TODO
        # .take(1024)  # for streaming
        text = "\n\n".join(ds["text"])
    else:  # 'wikitext'
        ds = load_dataset(
            "wikitext", "wikitext-2-raw-v1",
            split="test",
            streaming=False
        )
        text = "\n\n".join(ds["text"])

    # [1, total_len] tensor
    input_ids = tokenizer(text)["input_ids"]
    # enc = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    enc = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # CPU

    # Evaluate
    model.eval()
    use_cache_flag = model.config.use_cache
    model.config.use_cache = False

    total_len = enc.size(1)
    step = seqlen
    nsamples = total_len // step
    if limit is not None:
        nsamples = min(nsamples, int(limit))

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    n_tokens = 0

    with torch.no_grad():
        pbar = tqdm(range(nsamples), desc=f"Eval({task_name})")
        for i in pbar:
            # batch = enc[:, i * step: (i + 1) * step]             # [1, L]
            # logits = model(batch).logits                         # [1, L, V]
            batch = enc[:, i * step: (i + 1) * step].to(
                device, non_blocking=True)  # [1, L]  # CPU -> GPU
            outputs = model(input_ids=batch, use_cache=False)
            logits = outputs.logits

            # next-token prediction
            step_loss = loss_fn(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            total_loss += step_loss.item()
            step_tokens = batch.size(1) - 1
            n_tokens += step_tokens

            # Monitoring
            avg_ppl = math.exp(total_loss / max(1, n_tokens))
            step_ppl = math.exp(step_loss.item() / max(1, step_tokens))
            pbar.set_postfix(
                avg=f"{avg_ppl:.4f}",
                step=f"{step_ppl:.4f}",
                toks=f"{n_tokens/1e6:.4f}M"
            )

    avg_nll = total_loss / n_tokens
    ppl = math.exp(avg_nll)

    model.config.use_cache = use_cache_flag
    return {'loss': avg_nll, 'perplexity': ppl}


def evaluate_classification(model, task_name, limit=None, fewshot=0):
    outputs = simple_evaluate(
        model=huggingface.HFLM(
            pretrained=model,
            backend='causal',
            tokenizer=tokenizer,
            # batch_size
        ),
        tasks=[task_name],
        limit=limit,
        num_fewshot=fewshot,
    )
    accuracy = outputs['results'][task_name]['acc,none']
    stderr = outputs['results'][task_name]['acc_stderr,none']
    return {'accuracy': accuracy, 'stderr': stderr}


# --- local_mse: module output cache hook ---

class _ModuleOutputCache:
    def __init__(self, store_input: bool = False):
        self.out = None  # Tensor (student: requires_grad, teacher: no grad)
        self.inp = None  # pre-activation input to the module
        self.module = None   # reference to the module (student side used)
        self.ignore = False  # bypass hook during manual calls
        self.store_input = store_input

    def __call__(self, module, inp, out):
        if self.ignore:
            return
        self.out = out
        if self.store_input:
            self.inp = inp[0]
        if self.module is None:
            self.module = module


def _register_module_output_hooks(model, ffn_names, attn_names,
                                  store_input: bool = False):
    caches = {}
    handles = []

    for layer_index, layer in enumerate(model.model.layers):
        # FFN
        for name in ffn_names:
            if hasattr(layer.mlp, name):
                module = getattr(layer.mlp, name)
                # cache = _ModuleOutputCache()
                cache = _ModuleOutputCache(store_input=store_input)
                h = module.register_forward_hook(cache)
                key = f"layer_{layer_index}_mlp_{name}"
                caches[key] = cache
                handles.append(h)
        # Attention
        for name in attn_names:
            if hasattr(layer.self_attn, name):
                module = getattr(layer.self_attn, name)
                # cache = _ModuleOutputCache()
                cache = _ModuleOutputCache(store_input=store_input)
                h = module.register_forward_hook(cache)
                key = f"layer_{layer_index}_self_attn_{name}"
                caches[key] = cache
                handles.append(h)

    return caches, handles


if args.healing_mode in ('local_mse', 'lillama'):
    student_out_caches, student_out_handles = _register_module_output_hooks(
        model, args.ffn_module_names, args.attn_module_names,
        store_input=False)
    teacher_out_caches, teacher_out_handles = _register_module_output_hooks(
        teacher_model, args.ffn_module_names, args.attn_module_names,
        store_input=(args.healing_mode == 'lillama'))


# FT
# Define loss functions
kl_kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
mse_loss_fn = torch.nn.MSELoss()


def lillama_feature_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred, target: [..., D]
    returns scalar: mean( mean(|pred-target|, dim=-1) - logsigmoid(cosine(pred,target)) )
    """
    D = pred.shape[-1]
    pred_flat = pred.reshape(-1, D)
    target_flat = target.reshape(-1, D)
    l1_term = torch.mean(
        torch.abs(pred_flat - target_flat), dim=-1)   # (1/D) * L1
    cos = F.cosine_similarity(pred_flat, target_flat, dim=-1, eps=eps)
    return (l1_term - F.logsigmoid(cos)).mean()


# Define the temperature and weighting factors
T = args.T  # Temperature for KD loss
alpha = args.alpha  # Weight for standard cross-entropy loss
beta = args.beta    # Weight for KL KD (0.1, 0.3, 0.5, ...)
gamma = args.gamma  # Weight for MSE loss over hidden states
# alpha, beta, gamma = (x / (alpha + beta + gamma) for x in (alpha, beta, gamma))
# total_loss = alpha * outputs.loss + beta * kd_loss + gamma * mse_loss

# Fine-tuning loop
# Initialize TensorBoard SummaryWriter
writers = {}
writers['train'] = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
for task_name in test_dataloaders.keys():
    task_log_dir = os.path.join(log_dir, task_name)
    writers[task_name] = SummaryWriter(log_dir=task_log_dir)

model.train()
optimizer.zero_grad(set_to_none=True)
global_step = 0  # Initialize a global step counter

# Create a mapping of task types for each test dataset
test_tasks = {
    # 57 categiries
    'mmlu':     {'task_type': 'classification', 'limit': int(args.num_test_steps / 4), 'fewshot': 5},
    'boolq':    {'task_type': 'classification', 'limit': int(args.num_test_steps / 2), 'fewshot': 0},

    'c4':       {'task_type': 'lm',             'limit': args.num_test_steps, },
    'wikitext': {'task_type': 'lm',             'limit': args.num_test_steps, },
}

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(train_dataloader),
                        total=total_steps, desc=f"Epoch {epoch+1}/{num_epochs}")

    for step, batch in progress_bar:
        if step >= total_steps:
            break
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)

        # Get teacher outputs (with hidden states) (no gradients needed)
        # with torch.no_grad():
        with torch.inference_mode():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(args.healing_mode == 'hidden_mse'),
                use_cache=False
            )

        labels = batch.get('labels', None)
        if labels is not None:
            labels = labels.to(device, non_blocking=True)
        if alpha == 0.0:
            labels = None

        # Get student outputs with hidden states
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # labels=input_ids,
            labels=labels,
            output_hidden_states=(args.healing_mode == 'hidden_mse'),
            use_cache=False
        )
        student_loss = student_outputs.loss
        if (alpha == 0.0) or (labels is None) or (student_loss is None):
            ce_loss = torch.zeros(
                (), device=input_ids.device, dtype=torch.float32)
        else:
            ce_loss = student_loss

        # Compute the KD loss (KL divergence)
        teacher_logits = None
        student_logits = None
        kd_loss = 0.0
        if beta != 0.0:
            student_logits = student_outputs.logits / T
            teacher_logits = teacher_outputs.logits / T

            # TODO
            # KL
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            kd_loss = kl_kd_loss_fn(
                student_log_probs,
                teacher_probs
            ) * (T * T)
            # # RKL
            # teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            # student_probs = F.softmax(student_logits, dim=-1)
            # kd_loss = kl_kd_loss_fn(
            #     teacher_log_probs,
            #     student_probs
            # ) * (T * T)

        # Compute MSE loss
        teacher_hidden = None
        student_hidden = None
        mse_loss = 0.0

        if args.healing_mode == 'hidden_mse':
            num_layers = len(student_outputs.hidden_states)
            # Exclude embeddings and last layer (not changed)
            for i in range(1, num_layers - 1):
                student_hidden = student_outputs.hidden_states[i]
                teacher_hidden = teacher_outputs.hidden_states[i]
                mse_loss += mse_loss_fn(student_hidden, teacher_hidden)
            # Average MSE loss over layers
            mse_loss = mse_loss / max(1, (num_layers - 2))

        elif args.healing_mode == 'local_mse':
            matched = 0
            for k in student_out_caches.keys():
                s_out = student_out_caches[k].out
                t_out = teacher_out_caches[k].out
                if (s_out is None) or (t_out is None):
                    continue
                mse_loss += mse_loss_fn(s_out, t_out)
                matched += 1
            # clear caches
            for c in student_out_caches.values():
                c.out = None
            for c in teacher_out_caches.values():
                c.out = None
            # Average MSE loss over matches
            mse_loss = mse_loss / max(1, matched)

        elif args.healing_mode == 'lillama':
            matched = 0
            for k, s_cache in student_out_caches.items():
                t_cache = teacher_out_caches.get(k, None)
                if t_cache is None:
                    continue
                s_out = s_cache.out
                t_out = t_cache.out
                # (1) Student-only term: student path vs teacher path
                if (s_out is not None) and (t_out is not None):
                    mse_loss += lillama_feature_loss(s_out, t_out)
                    matched += 1
                # (2) Teacher-only term: feed teacher input into the student module
                t_in = getattr(t_cache, "inp", None)
                if (t_in is not None) and (t_out is not None) and (s_cache.module is not None):
                    # prevent hook overwrite during manual call
                    s_cache.ignore = True
                    param = next(s_cache.module.parameters(), None)
                    if param is not None:
                        t_in_local = t_in.to(
                            device=param.device, dtype=param.dtype)
                    else:
                        t_in_local = t_in
                    yhat_t = s_cache.module(t_in_local)
                    s_cache.ignore = False
                    mse_loss += lillama_feature_loss(yhat_t, t_out)
                    matched += 1
            # clear caches
            for c in student_out_caches.values():
                c.out = None
                c.inp = None
            for c in teacher_out_caches.values():
                c.out = None
                c.inp = None
            # average over all matched terms
            mse_loss = mse_loss / max(1, matched)

        del teacher_outputs, student_outputs
        del teacher_logits, student_logits
        del teacher_hidden, student_hidden

        # Combine losses
        total_loss = alpha * ce_loss + beta * kd_loss + gamma * mse_loss
        loss = total_loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Log the training loss to TensorBoard
        # if global_step % 100 == 0:  # TODO: enable
        writers['train'].add_scalar('Train/Loss', loss.item(), global_step)

        progress_bar.set_postfix({"Loss": loss.item()})

        if (step == 0) or ((step + 1) % args.test_interval == 0):
            for task_name, task_info in test_tasks.items():

                if task_info['task_type'] == 'lm':
                    eval_result = evaluate_lm(
                        model, task_name=task_name, limit=task_info['limit']
                    )
                    val_loss = eval_result['loss']
                    perplexity = eval_result['perplexity']
                    print(
                        f"\nTest {task_name} Loss: {val_loss}, Perplexity: {perplexity}")

                    # Log test loss and perplexity to TensorBoard
                    writers[task_name].add_scalar(
                        'Test/Loss', val_loss, global_step)
                    writers[task_name].add_scalar(
                        'Test/Perplexity', perplexity, global_step)

                elif task_info['task_type'] == 'classification':
                    eval_result = evaluate_classification(
                        model,
                        task_name=task_name, limit=task_info['limit'], fewshot=task_info['fewshot']
                    )
                    accuracy = eval_result['accuracy']
                    stderr = eval_result['stderr']
                    print(
                        f"\nTest {task_name} Accuracy: {accuracy} ({stderr})")

                    # Log test accuracy w/ stderr to TensorBoard
                    writers[task_name].add_scalar(
                        'Test/Accuracy', accuracy, global_step)
                    writers[task_name].add_scalar(
                        'Test/Acc_StdError', stderr, global_step)

            model.train()

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


if args.healing_mode in ('local_mse', 'lillama'):
    for h in student_out_handles:
        h.remove()
    for h in teacher_out_handles:
        h.remove()


# # Generate text with the fine-tuned model
# prompt = "Once upon a time"
# generated_text = generate_text(model, prompt)
# print("Fine-tuned model output:", generated_text)
