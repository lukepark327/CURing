"""
CUDA_VISIBLE_DEVICES=0 python benchmark/performance/llama_c4.py
    --max_steps=128
"""
# autopep8: off
import os
import sys
import argparse
import math
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from datasets import load_dataset
from tqdm import tqdm

# CURing 패키지가 설치돼 있지 않은 경우를 대비해 상대 경로 추가
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../CURing"))
)
from cur_models import CURLinear  # noqa: F401  (미리 선언만 필요)

CURLinear

# autopep8: on
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ----------------------------- 인자 파서 -------------------------------
parser = argparse.ArgumentParser(
    description="Evaluate Llama model on C4")
parser.add_argument(
    "--seqlen",
    type=int,
    default=4096,  # TODO: None
    help="모델 한 pass 길이(기본: model.config.n_positions 또는 4096)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="재현성용 random seed",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=128,
    help="평가할 최대 스텝 수(윈도우 개수). 예: 128"
)
args = parser.parse_args()
print(args)

# TODO
model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-13b-hf"

# ----------------------------- 설정 ------------------------------------
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", f"CUDA 안 잡혔습니다 → {device}"
print(f"[INFO] device  : {device}")

# ----------------------------- 모델 로드 -------------------------------
print(f"[INFO] 모델 로드 중 … {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device).eval()

# 최대 시퀀스 길이 결정
model.seqlen = (
    args.seqlen
    or getattr(model.config, "n_positions", None)
    or getattr(model.config, "max_position_embeddings", None)
    or 4096
)
print(f"[INFO] seqlen   : {model.seqlen}")

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------- C4 데이터 준비 ---------------------------


def get_c4() -> Tuple[List[List[int]], torch.Tensor]:
    """
    훈련(un-ordered 샘플)‧테스트(전체) 토큰 시퀀스를 반환
    """
    testdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.*.json.gz"},
        split="validation",
    ).take(1024)  # TODO
    test_ids = tokenizer("\n\n".join(testdata["text"]))["input_ids"]
    # torch.Tensor (batch 1)
    test_tensor = torch.tensor(test_ids, dtype=torch.long).unsqueeze(0)
    return None, test_tensor


def eval_model(
    model: AutoModelForCausalLM,
    enc: torch.Tensor,
    device: torch.device = torch.device("cuda"),
    max_steps: int | None = None,
) -> float:
    """
    enc: shape [1, total_len]
    시퀀스를 model.seqlen 단위로 잘라 perplexity 계산
    """
    model.eval()
    # 기존 캐시 설정 보존
    use_cache_flag = model.config.use_cache
    model.config.use_cache = False

    enc = enc.to(device)
    total_len = enc.size(1)
    step = model.seqlen
    nsamples = total_len // step
    if max_steps is not None:
        nsamples = min(nsamples, max_steps)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    n_tokens = 0

    with torch.no_grad():
        pbar = tqdm(range(nsamples), desc="Evaluating")
        for i in pbar:
            batch = enc[:, i * step: (i + 1) * step]
            # Forward
            logits = model(batch).logits  # [B, L, V]
            # Shift
            step_loss = loss_fn(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            total_loss += step_loss.item()
            step_tokens = batch.size(1) - 1
            n_tokens += step_tokens

            avg_ppl = math.exp(total_loss / max(1, n_tokens))
            step_ppl = math.exp(step_loss.item() / max(1, step_tokens))
            pbar.set_postfix(
                avg=f"{avg_ppl:.4f}",
                step=f"{step_ppl:.4f}",
                toks=f"{n_tokens/1e6:.4f}M",
            )

    ppl = math.exp(total_loss / n_tokens)

    model.config.use_cache = use_cache_flag
    return ppl


# --------------------------- 평가 실행 ---------------------------------
print("[INFO] C4 다운로드·토큰화 중 …")
_, test_enc = get_c4()
print("[INFO] 평가 시작 …")
perplexity = eval_model(model, test_enc, device=device,
                        max_steps=args.max_steps)

print("=" * 60)
print(f"Perplexity (C4 test) : {perplexity:,.3f}")
print("=" * 60)
