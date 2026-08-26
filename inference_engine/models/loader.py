"""
models/loader.py — Model and tokenizer loading with device-aware strategy.

Key design decisions
--------------------
* MPS (Apple Silicon): device_map=None, manual .to("mps") after load.
  device_map="auto" is a CUDA multi-GPU feature; on MPS it can silently
  fall back some ops to float32 and produces inconsistent memory readings.
* CUDA: device_map="auto" for standard multi-GPU handling.
* CPU: torch_dtype=torch.float32 — float16 matmuls on CPU are emulated and
  very slow; float32 is the right default.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from inference_engine.config import Config

logger = logging.getLogger(__name__)


class LoadedModel(NamedTuple):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str


def load_model_and_tokenizer(config: Config) -> LoadedModel:
    """
    Load a causal LM and its tokenizer according to config.device.

    Returns a LoadedModel named tuple so callers can unpack as:
        model, tokenizer, device = load_model_and_tokenizer(config)
    """
    device = config.device
    model_name = config.model_name

    logger.info("Loading model '%s' targeting device '%s'", model_name, device)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    # Ensure a pad token exists (some models omit it).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model — device-conditional loading ────────────────────────────────────
    if device == "cuda":
        # Multi-GPU-aware: let HuggingFace distribute layers.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        logger.info("Model loaded with device_map='auto' on CUDA")

    elif device == "mps":
        # Load to CPU first, then move to MPS in one shot.
        # device_map="auto" on MPS can silently dispatch ops to CPU (float32
        # fallback), which corrupts memory measurements.
        #
        # dtype on MPS is a real correctness knob, not just speed/memory:
        # PyTorch's Metal kernels for bf16/fp16 are less precise than the CPU
        # reference (which does internal fp32 accumulation). For most models
        # (e.g. Qwen2) fp16 is fine. But numerically-sensitive deep models
        # like Gemma 4 (42 layers) accumulate MPS low-precision error layer by
        # layer until the argmax flips and long-form decode collapses into
        # single-token repetition — bf16 fails immediately, fp16 fails on long
        # sequences, only fp32 is stable throughout (verified via CPU-vs-MPS
        # per-layer diff; see docs/specs/gemma4-e4b-adaptation.md §6.5).
        # Default fp16 (fast, half memory); override with MPS_DTYPE=float32 for
        # sensitive models at the cost of ~2x memory.
        import os
        mps_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }.get(os.environ.get("MPS_DTYPE", "float16").lower(), torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=mps_dtype,
            device_map=None,          # <-- intentional, not a mistake
            low_cpu_mem_usage=True,   # reduce peak RAM during load
        )
        model = model.to("mps")
        logger.info("Model loaded to MPS via explicit .to('mps') in %s", mps_dtype)

    else:  # cpu
        # float16 matmuls on CPU are emulated → very slow. Use float32.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded on CPU with float32")

    model.eval()  # disable dropout, set BN to eval mode

    logger.info(
        "Model ready. Parameters: %s M | dtype: %s",
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}",
        next(model.parameters()).dtype,
    )

    return LoadedModel(model=model, tokenizer=tokenizer, device=device)
