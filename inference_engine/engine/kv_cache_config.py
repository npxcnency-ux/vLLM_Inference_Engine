"""Analytical KV-cache sizing from HuggingFace model metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: str
    # Per-layer head_dim.  For homogeneous models every entry equals head_dim.
    # For heterogeneous models (e.g. Gemma 4's mixed attention, where global
    # layers use head_dim 512 and sliding-window layers use 256) this holds the
    # true width of each layer.  ``head_dim`` above is the MAX of these values
    # and is used for the padded tensor pool allocation width.
    head_dims: list[int] = field(default_factory=list)
    bytes_per_token: int = field(init=False)
    bytes_per_token_mb: float = field(init=False)

    def __post_init__(self) -> None:
        # Back-compat: callers that only pass the scalar head_dim (Qwen, older
        # tests) get a homogeneous per-layer list filled in automatically.
        if not self.head_dims:
            self.head_dims = [self.head_dim] * self.num_layers
        # head_dim is the allocation width — always the max across layers.
        self.head_dim = max(self.head_dims)

        dtype_bytes = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
        }.get(self.dtype, 2)
        # Precise byte count uses the true per-layer widths (sum), not the
        # padded max — otherwise heterogeneous models over-report memory.
        # num_kv_heads is uniform across layers for the models we support.
        self.bytes_per_token = (
            2 * self.num_kv_heads * sum(self.head_dims) * dtype_bytes
        )
        self.bytes_per_token_mb = self.bytes_per_token / (1024 * 1024)


def compute_kv_cache_config(model, config) -> KVCacheConfig:
    """Build cache sizing metadata from a HuggingFace model.

    Handles three complications found in modern architectures:

    1. Nested config — multimodal models (e.g. Gemma 4) keep the text-tower
       attention params under ``model.config.text_config``, not the top level.
    2. KV-shared layers — Gemma 4 shares KV across its last
       ``num_kv_shared_layers`` layers, so the real number of cache layers is
       ``num_hidden_layers - num_kv_shared_layers`` (42 - 18 = 24 for E4B).
       Iterating past this count would IndexError against the model's
       past_key_values.
    3. Heterogeneous head_dim — Gemma 4's mixed attention gives global layers
       head_dim 512 and sliding-window layers 256.  ``head_dims`` captures the
       per-layer widths; the pool allocates at the max.
    """
    # 1. Reach the text-tower config (falls back to the config itself).
    if hasattr(model.config, "get_text_config"):
        text_config = model.config.get_text_config()
    else:
        text_config = getattr(model.config, "text_config", model.config)

    num_hidden_layers = text_config.num_hidden_layers
    num_attention_heads = text_config.num_attention_heads
    num_kv_heads = getattr(
        text_config, "num_key_value_heads", num_attention_heads
    )

    # 2. Real cache-layer count = hidden layers minus KV-shared layers.
    num_kv_shared = getattr(text_config, "num_kv_shared_layers", 0) or 0
    cache_layers = num_hidden_layers - num_kv_shared

    # 3. Per-layer head_dim.
    per_layer_config = getattr(text_config, "per_layer_config", None)
    if getattr(text_config, "is_heterogeneous", False) and per_layer_config is not None:
        head_dims = [
            per_layer_config[i].head_dim for i in range(cache_layers)
        ]
    else:
        # Homogeneous: read the scalar head_dim, else derive from hidden size.
        # Wrapped in try/except because accessing head_dim on a heterogeneous
        # config raises AmbiguousGlobalPerLayerAttributeError.
        try:
            hd = getattr(text_config, "head_dim", None)
        except Exception:
            hd = None
        if hd is None:
            hd = text_config.hidden_size // num_attention_heads
        head_dims = [hd] * cache_layers

    return KVCacheConfig(
        num_layers=cache_layers,
        num_kv_heads=num_kv_heads,
        head_dim=max(head_dims),
        head_dims=head_dims,
        dtype=next(model.parameters()).dtype,
        device=config.device,
    )


def estimate_max_sequences(
    kv_cache_config: KVCacheConfig,
    available_memory_mb: float,
    avg_sequence_length: int = 512,
) -> int:
    """Estimate how many average-length sequence caches fit in memory."""
    memory_per_sequence_mb = (
        kv_cache_config.bytes_per_token_mb * avg_sequence_length
    )
    if memory_per_sequence_mb <= 0:
        return 1
    return max(1, math.floor(available_memory_mb / memory_per_sequence_mb))


def format_kv_cache_report(kv_cache_config: KVCacheConfig) -> str:
    """Return a human-readable summary of KV-cache sizing metadata."""
    distinct = sorted(set(kv_cache_config.head_dims))
    return "\n".join(
        [
            "KV Cache Configuration",
            f"  Layers: {kv_cache_config.num_layers}",
            f"  KV heads: {kv_cache_config.num_kv_heads}",
            f"  Head dimension: max={kv_cache_config.head_dim}, "
            f"per-layer distinct={distinct}",
            f"  Dtype: {kv_cache_config.dtype}",
            f"  Bytes per token: {kv_cache_config.bytes_per_token / 1024:.2f} KB",
            f"  Memory per token: {kv_cache_config.bytes_per_token_mb:.6f} MB",
        ]
    )
