"""
CUDA_VISIBLE_DEVICES=1 python benchmark/performance/CURing_wikitext.py \
    --load_path "./cur_decomposed_models/C128_N10_R256_20250327_014256"
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
    description="Evaluate CUR-Decomposed model on WikiText-2")
parser.add_argument(
    "--load_path",
    type=str,
    required=True,
    help="디스크에 저장된 (어댑터 적용) 모델 디렉터리",
)
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
    default=None,
    help="평가할 최대 스텝 수(윈도우 개수). 예: 128"
)
args = parser.parse_args()
print(args)
load_path = f"{args.load_path}"

# ----------------------------- 설정 ------------------------------------
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", f"CUDA 안 잡혔습니다 → {device}"
print(f"[INFO] device  : {device}")

# ----------------------------- 모델 로드 -------------------------------
print(f"[INFO] 모델 로드 중 … {load_path}")
model = torch.load(os.path.join(load_path, "model.pt"), weights_only=False)
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
tokenizer = AutoTokenizer.from_pretrained(load_path)
# tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------- WikiText-2 데이터 준비 ---------------------------


def get_wikitext2() -> Tuple[List[List[int]], torch.Tensor]:
    """
    훈련(un-ordered 샘플)‧테스트(전체) 토큰 시퀀스를 반환
    """
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
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
print("[INFO] WikiText-2 다운로드·토큰화 중 …")
_, test_enc = get_wikitext2()
print("[INFO] 평가 시작 …")
perplexity = eval_model(model, test_enc, device=device,
                        max_steps=args.max_steps)

print("=" * 60)
print(f"Perplexity (WikiText-2 test) : {perplexity:,.3f}")
print("=" * 60)


# Llama-7B Original
# Perplexity (WikiText-2 test) : 5.116

# Llama-7B max-length=4096 calibration=256 GPU use_lowrank=True
# Perplexity (WikiText-2 test) : 19.735

# Llama-7B max-length=4096 calibration=256 GPU use_lowrank=False
# Perplexity (WikiText-2 test) : 19.266

# Llama-7B max-length=4096 calibration=256 cov max_rank=256
# Perplexity (WikiText-2 test) : 27.576  # WANDA 대비 cov 성능 안 좋음

# Llama-7B max-length=4096 calibration=1024 cov max_rank=256
# Perplexity (WikiText-2 test) : 26.455  # WANDA 대비 cov 성능 안 좋음

# Llama-7B max-length=4096 calibration=256 cov_fast max_rank=256 use_lowrank=True
# Perplexity (WikiText-2 test) : 17.979  # 성능이 좋네?

# Llama-7B max-length=4096 calibration=256 cov_fast N=10 energy=0.30 use_lowrank=True
# Perplexity (WikiText-2 test) : 17.065  # 더 낮은 용량 더 좋은 성능

# Llama-7B max-length=4096 calibration=256 cov_fast max_rank=256 use_lowrank=False
# Perplexity (WikiText-2 test) : 18.726  # 오히려 full SVD 하면 성능 하락?

# Llama-7B max-length=4096 calibration=256 cov max_rank=1024
# Perplexity (WikiText-2 test) : 12.724  # 압축 성능 안 좋음

# Llama-7B max-length=4096 calibration=256 cov max_rank=None
# Perplexity (WikiText-2 test) : 5.680  # 용량 오히려 늘었음

# Llama-7B max-length=4096 calibration=256 Healed=2001
# Perplexity (WikiText-2 test) : 13.271


# llama-13B Original
# Perplexity (WikiText-2 test) : 4.574

# Llama-13B max-length=4096 calibration=256 GPU use_lowrank=True
# Perplexity (WikiText-2 test) : 9.537

# Llama-13B max-length=4096 calibration=256 GPU use_lowrank=False
# Perplexity (WikiText-2 test) : 9.705

# Llama-13B max-length=4096 calibration=256 cov_fast max_rank=256 use_lowrank=True
# Perplexity (WikiText-2 test) : 9.829
# Perplexity (WikiText-2 test) : 9.578  # W.abs() 사용 시

# Llama-13B max-length=4096 calibration=256 cov_fast max_rank=256 use_lowrank=False
# Perplexity (WikiText-2 test) : 9.850

# Llama-13B max-length=4096 calibration=256 Healed=1001
# Perplexity (WikiText-2 test) : 7.427
