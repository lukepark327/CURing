# autopep8: off
import os
import sys
import argparse
from tqdm import tqdm
import math

import torch
from transformers import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from lm_eval.models import huggingface
from lm_eval import simple_evaluate

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../CURing"))
)
from cur_models import CURLinear  # noqa: F401

CURLinear

# autopep8: on
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"


parser = argparse.ArgumentParser(description="Test CUR-Decomposed Models")

# Model and Paths
parser.add_argument('--load_path', type=str, default=None,
                    help='Directory path from where the modified model is loaded.')
parser.add_argument('--device', type=str, default="cuda",
                    help='Device to run the computations on (e.g., "cpu", "cuda").')

# Parameters
parser.add_argument('--max_length', type=int, default=4096,
                    help='Max length per dataset.')
parser.add_argument('--num_test_steps', type=int, default=128,
                    help='Number of test steps.')

# Miscellaneous
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')

args = parser.parse_args()


# Set random seed for reproducibility
set_seed(args.seed)


# Set model name and paths
load_path = args.load_path

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


# Load the model with custom class
model = torch.load(os.path.join(load_path, 'model.pt'), weights_only=False)
model.to(device).eval()


def count_params(model, trainable_only=False) -> int:
    if hasattr(model, "num_parameters"):
        try:
            return model.num_parameters(only_trainable=trainable_only)
        except TypeError:
            pass
    params = (p for p in model.parameters() if (
        p.requires_grad or not trainable_only))
    return sum(p.numel() for p in params)


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

            stats = pbar.format_dict
            elapsed = stats["elapsed"]
            toks_per_s = n_tokens / elapsed if elapsed > 0 else 0.0

            pbar.set_postfix(
                avg=f"{avg_ppl:.4f}",
                step=f"{step_ppl:.4f}",
                toks=f"{n_tokens/1e6:.4f}M",
                tps=f"{toks_per_s:,.0f}/s",
            )

    avg_nll = total_loss / n_tokens
    ppl = math.exp(avg_nll)

    model.config.use_cache = use_cache_flag
    return {'loss': avg_nll, 'perplexity': ppl, 'tps': f"{toks_per_s:,.0f}/s"}


def _pick_primary_metric_keys(res: dict):
    preferred = [
        "acc_norm,none", "acc_norm",
        "acc,none", "acc",
        "exact_match,flexible-extract", "exact_match_flexible_extract",
        "exact_match,strict-match", "exact_match_strict_extract",
        "exact_match,none", "exact_match",
    ]
    metric_key = next((k for k in preferred if k in res), None)
    if metric_key is None:
        metric_key = next(k for k in res.keys() if "stderr" not in k)

    base, option = metric_key.split(",")

    stderr_key_candidates = [
        f"{base}_stderr,{option}",
        f"{base}_stderr",
        f"{metric_key}_stderr",
    ]
    stderr_key = next((k for k in stderr_key_candidates if k in res), None)

    return metric_key, stderr_key


def evaluate_classification(model, task_name, limit=None, fewshot=0):
    outputs = simple_evaluate(
        model=huggingface.HFLM(
            pretrained=model,
            backend='causal',
            tokenizer=tokenizer,
            trust_remote_code=True,
            # batch_size
        ),
        tasks=[task_name],
        limit=limit,
        num_fewshot=fewshot,
    )

    # accuracy = outputs['results'][task_name]['acc,none']
    # stderr = outputs['results'][task_name]['acc_stderr,none']

    res = outputs['results'].get(task_name)
    if res is None:
        res = outputs.get('groups', {}).get(task_name)
        assert res is not None, f"Result for task/group not found: {task_name}"

    metric_key, stderr_key = _pick_primary_metric_keys(res)
    accuracy = res[metric_key]
    stderr = res.get(stderr_key, None)

    return {'accuracy': accuracy, 'stderr': stderr}


# Create a mapping of task types for each test dataset
test_tasks = {
    # QA
    # 'social_iqa':    {'task_type': 'classification', 'limit': int(args.num_test_steps / 3), 'fewshot': 0},
    # 'logiqa':        {'task_type': 'classification', 'limit': int(args.num_test_steps / 4), 'fewshot': 5},
    # 'winogrande':    {'task_type': 'classification', 'limit': int(args.num_test_steps / 2), 'fewshot': 5},
    # 'arc_easy':      {'task_type': 'classification', 'limit': int(args.num_test_steps / 4), 'fewshot': 25},
    # 'arc_challenge': {'task_type': 'classification', 'limit': int(args.num_test_steps / 4), 'fewshot': 25},
    # 'piqa':          {'task_type': 'classification', 'limit': int(args.num_test_steps / 2), 'fewshot': 0},
    # 'openbookqa':    {'task_type': 'classification', 'limit': int(args.num_test_steps / 4), 'fewshot': 0},

    # Hard
    # 'gsm8k':            {'task_type': 'classification', 'limit': int(args.num_test_steps),      'fewshot': 8},
    # 'hendrycks_math':   {'task_type': 'classification', 'limit': int(args.num_test_steps),      'fewshot': 4},

    # Slow
    # 'bbh':              {'task_type': 'classification', 'limit': int(args.num_test_steps),      'fewshot': 3},
    # 'gpqa':             {'task_type': 'classification', 'limit': int(args.num_test_steps / 4),  'fewshot': 0},

    # Normal
    'mmlu':             {'task_type': 'classification', 'limit': int(args.num_test_steps / 4),  'fewshot': 5},
    'boolq':            {'task_type': 'classification', 'limit': int(args.num_test_steps / 2),  'fewshot': 0},
    'wikitext':         {'task_type': 'lm',             'limit': args.num_test_steps, },
    'c4':               {'task_type': 'lm',             'limit': args.num_test_steps, },
}


# Test
results = []

for task_name, task_info in test_tasks.items():

    if task_info['task_type'] == 'lm':
        eval_result = evaluate_lm(
            model, task_name=task_name, limit=task_info['limit']
        )
        val_loss = eval_result['loss']
        perplexity = eval_result['perplexity']
        tps = eval_result['tps']
        results.append(
            f"{task_name},\tLoss: {val_loss},\tPerplexity: {perplexity},\tTPS: {tps}"
        )

    elif task_info['task_type'] == 'classification':
        eval_result = evaluate_classification(
            model,
            task_name=task_name, limit=task_info['limit'], fewshot=task_info['fewshot']
        )
        accuracy = eval_result['accuracy']
        stderr = eval_result['stderr']
        results.append(
            f"{task_name},\tAccuracy: {accuracy},\tStd: {stderr}"
        )

print("\n".join(results))
print()
print("Size (B):", count_params(model) / 1000000000)
