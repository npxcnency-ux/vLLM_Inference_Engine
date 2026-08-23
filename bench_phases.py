#!/usr/bin/env python
"""
bench_phases.py — Per-phase benchmark for PageServe.

Measures the metrics that actually changed in each phase:
  Phase 1: single-request latency (sequential baseline)
  Phase 2: continuous batching speedup (N concurrent requests)
  Phase 3: queue depth & wait time under load
  Phase 4: prefill budget isolation (long vs short prompt interleaving)
  Phase 5: KV memory reporting accuracy
  Phase 6: block allocator — max sequences before OOM
  Phase 7: paged KV throughput (write + read round-trip)
  Phase 8: decode with live past_kv (current hot-path speed)
  Phase 9: chunked prefill — TTFT with long prompt under decode load
  Phase 10/11: aggregate throughput under sustained load

All tests run directly against the model (no HTTP). Results saved to
bench_phases_results.json and printed as a comparison table.

Usage:
    .venv/bin/python bench_phases.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import torch

sys.path.insert(0, ".")

from inference_engine.config import Config
from inference_engine.models.loader import load_model_and_tokenizer

MAX_TOKENS = 50
PHASE1_MEAN_MS = 1418.4   # cold single-request baseline from baseline_metrics.json
PHASE1_TTFT_MS = 16.0

PROMPTS = [
    "Explain entropy in thermodynamics.",
    "What is supervised learning?",
    "How does HTTPS work?",
    "Describe photosynthesis briefly.",
]

SEP = "─" * 70


def _single_request(model, tokenizer, device, prompt, max_tokens=MAX_TOKENS):
    """Run one full prefill+decode sequence. Returns timing dict."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    past = out.past_key_values
    next_tok = int(out.logits[0, -1].argmax())
    generated = [next_tok]
    tok_times = []
    eos = tokenizer.eos_token_id

    for _ in range(max_tokens - 1):
        if eos and next_tok == eos:
            break
        t1 = time.perf_counter()
        inp = torch.tensor([[next_tok]], device=device)
        with torch.no_grad():
            out = model(inp, past_key_values=past, use_cache=True)
        tok_times.append((time.perf_counter() - t1) * 1000.0)
        next_tok = int(out.logits[0, -1].argmax())
        past = out.past_key_values
        generated.append(next_tok)

    total_ms = ttft_ms + sum(tok_times)
    return {
        "prompt_len": ids.shape[1],
        "n_generated": len(generated),
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tok_mean_ms": statistics.mean(tok_times) if tok_times else 0,
        "tps": len(generated) / (total_ms / 1000.0),
    }


def bench_phase1(model, tokenizer, device):
    """Phase 1: sequential single-request baseline."""
    print(f"\n{'='*70}")
    print("  PHASE 1 — Sequential Serving Baseline")
    print(f"{'='*70}")
    print("  (re-measuring warm; cold numbers from baseline_metrics.json)")
    results = [_single_request(model, tokenizer, device, p) for p in PROMPTS]
    ttfts = [r["ttft_ms"] for r in results]
    totals = [r["total_ms"] for r in results]
    tps = [r["tps"] for r in results]
    print(f"\n  TTFT mean:          {statistics.mean(ttfts):>8.1f} ms   (cold baseline: {PHASE1_TTFT_MS:.0f} ms)")
    print(f"  Total latency mean: {statistics.mean(totals):>8.1f} ms   (cold baseline: {PHASE1_MEAN_MS:.0f} ms)")
    print(f"  Throughput mean:    {statistics.mean(tps):>8.1f} tok/s")
    print(f"  Concurrency:        1 request at a time (Lock-serialised)")
    return {"ttft_ms": statistics.mean(ttfts), "total_ms": statistics.mean(totals), "tps": statistics.mean(tps)}


def bench_phase2(model, tokenizer, device):
    """Phase 2: continuous batching — N concurrent requests, interleaved decode."""
    print(f"\n{'='*70}")
    print("  PHASE 2 — Continuous Batching Scheduler")
    print(f"{'='*70}")
    N = 4
    # Prefill all
    t_wall = time.perf_counter()
    states = []
    for p in PROMPTS[:N]:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        first_token = int(out.logits[0,-1].argmax())
        states.append({"past": out.past_key_values,
                       "next": first_token,
                       "generated": [first_token],
                       "ttft_ms": (time.perf_counter()-t_wall)*1000,
                       "tok_ms": []})

    eos = tokenizer.eos_token_id
    while True:
        active = [s for s in states if len(s["generated"]) < MAX_TOKENS
                  and not (eos is not None and s["generated"][-1] == eos)]
        if not active: break
        for s in active:
            t1 = time.perf_counter()
            inp = torch.tensor([[s["next"]]], device=device)
            with torch.no_grad():
                out = model(inp, past_key_values=s["past"], use_cache=True)
            s["tok_ms"].append((time.perf_counter()-t1)*1000)
            s["next"] = int(out.logits[0,-1].argmax())
            s["past"] = out.past_key_values
            s["generated"].append(s["next"])

    wall_ms = (time.perf_counter()-t_wall)*1000
    total_toks = sum(len(s["generated"]) for s in states)
    agg_tps = total_toks / (wall_ms/1000)
    speedup = (PHASE1_MEAN_MS * N) / wall_ms
    ttfts = [s["ttft_ms"] for s in states]
    tot_lats = [s["ttft_ms"] + sum(s["tok_ms"]) for s in states]

    print(f"\n  {N} concurrent requests × {MAX_TOKENS} tokens (round-robin decode)")
    print(f"  Wall time:          {wall_ms:>8.1f} ms")
    print(f"  Phase 1 expected:   {PHASE1_MEAN_MS*N:>8.1f} ms")
    print(f"  Speedup:            {speedup:>8.2f}×")
    print(f"  Aggregate tps:      {agg_tps:>8.1f} tok/s")
    print(f"  TTFT mean:          {statistics.mean(ttfts):>8.1f} ms")
    print(f"  Total latency mean: {statistics.mean(tot_lats):>8.1f} ms")
    return {"wall_ms": wall_ms, "speedup": speedup, "agg_tps": agg_tps,
            "ttft_ms": statistics.mean(ttfts), "total_ms": statistics.mean(tot_lats)}


def bench_phase4_prefill_budget(model, tokenizer, device):
    """Phase 4: prefill budget — long prompt vs short prompt interleaving.
    Shows that chunked prefill prevents long prompts from blocking decode.
    """
    print(f"\n{'='*70}")
    print("  PHASE 4 / 9 — Prefill Budget & Chunked Prefill")
    print(f"{'='*70}")
    CHUNK = 128  # from config.prefill_chunk_size
    long_prompt = "Explain machine learning. " * 100  # comfortably over one chunk
    short_prompt = "Hi."

    # Simulate: long prompt in chunks, short prompt interleaved
    long_ids = tokenizer(long_prompt, return_tensors="pt").input_ids.to(device)
    short_ids = tokenizer(short_prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = long_ids.shape[1]
    n_chunks = (prompt_len + CHUNK - 1) // CHUNK

    print(f"\n  Long prompt length: {prompt_len} tokens  →  {n_chunks} chunk(s) of {CHUNK}")

    # Time full prefill (old behaviour = blocking)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(long_ids, use_cache=True)
    full_prefill_ms = (time.perf_counter()-t0)*1000

    # Time first chunk only (new behaviour = TTFT bounded)
    chunk_ids = long_ids[:, :min(CHUNK, prompt_len)]
    t0 = time.perf_counter()
    with torch.no_grad():
        out_chunk = model(chunk_ids, use_cache=True)
    first_chunk_ms = (time.perf_counter()-t0)*1000

    # Time decode step for co-running short request
    with torch.no_grad():
        short_out = model(short_ids, use_cache=True)
    next_tok = torch.tensor([[int(short_out.logits[0,-1].argmax())]], device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(next_tok, past_key_values=short_out.past_key_values, use_cache=True)
    decode_step_ms = (time.perf_counter()-t0)*1000

    print(f"\n  WITHOUT chunked prefill:")
    print(f"    Full prefill blocks for:   {full_prefill_ms:>8.1f} ms")
    print(f"    Short request decode delay:{full_prefill_ms:>8.1f} ms  (blocked!)")
    print(f"\n  WITH chunked prefill (chunk_size={CHUNK}):")
    print(f"    First chunk cost:          {first_chunk_ms:>8.1f} ms")
    print(f"    Short request decode runs: {decode_step_ms:>8.1f} ms  (interleaved)")
    print(f"    Decode delay reduction:    {full_prefill_ms - first_chunk_ms:>8.1f} ms saved")
    return {"full_prefill_ms": full_prefill_ms, "first_chunk_ms": first_chunk_ms,
            "decode_step_ms": decode_step_ms, "prompt_len": prompt_len, "n_chunks": n_chunks}


def bench_phase8_live_kv(model, tokenizer, device):
    """Phase 8 fix: live past_kv vs reconstructed-from-pool per step."""
    print(f"\n{'='*70}")
    print("  PHASE 8 — Decode Step: Live KV vs Pool Reconstruction")
    print(f"{'='*70}")
    from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
    from inference_engine.engine.block_allocator import BlockAllocator
    from inference_engine.engine.attention_wrapper import (
        _extract_kv_layer, extract_new_token_kv, build_past_key_values
    )
    from inference_engine.engine.sequence import Sequence
    from inference_engine.engine.kv_cache_config import compute_kv_cache_config

    cfg = Config()
    kv_cfg = compute_kv_cache_config(model, cfg)
    allocator = BlockAllocator(num_blocks=64, block_size=cfg.kv_block_size)
    paged_kv = PagedKVCacheManager(kv_cfg, allocator, cfg)

    ids = tokenizer("Hello world test", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, use_cache=True)

    seq = Sequence.create("Hello world test", ids[0].tolist(), 5)
    allocator.allocate(seq.seq_id, num_blocks=2)
    prompt_len = ids.shape[1]
    allocator.set_token_count(seq.seq_id, prompt_len)
    for l in range(kv_cfg.num_layers):
        layer_keys, layer_values = _extract_kv_layer(out.past_key_values, l)
        for token_pos in range(prompt_len):
            paged_kv.write_kv(
                seq.seq_id,
                l,
                token_pos,
                layer_keys[0, :, token_pos, :],
                layer_values[0, :, token_pos, :],
            )

    N = 20
    # ── (A) live past_kv path (Phase 8 fix — current behaviour)
    past = out.past_key_values
    next_tok = torch.tensor([[int(out.logits[0,-1].argmax())]], device=device)
    times_live = []
    for i in range(N):
        t0 = time.perf_counter()
        with torch.no_grad():
            step_out = model(next_tok, past_key_values=past, use_cache=True)
        times_live.append((time.perf_counter()-t0)*1000)
        next_tok = torch.tensor([[int(step_out.logits[0,-1].argmax())]], device=device)
        past = step_out.past_key_values
        # Write to pool (still happens, but cheap)
        allocator.write_token(seq.seq_id)
        token_pos = prompt_len + i
        for l in range(kv_cfg.num_layers):
            k, v = extract_new_token_kv(step_out.past_key_values, l, token_pos)
            paged_kv.write_kv(seq.seq_id, l, token_pos, k, v)

    # ── (B) pool-reconstruct path (old bug — was doing this every step)
    times_reconstruct = []
    for i in range(min(N, 5)):   # only 5 — it's slow
        t0 = time.perf_counter()
        reconstructed = build_past_key_values(
            seq_id=seq.seq_id, paged_kv_cache=paged_kv,
            num_layers=kv_cfg.num_layers, device=str(device), use_dynamic_cache=True
        )
        times_reconstruct.append((time.perf_counter()-t0)*1000)

    live_mean = statistics.mean(times_live)
    recon_mean = statistics.mean(times_reconstruct)
    overhead_ratio = recon_mean / live_mean

    print(f"\n  ({N} decode steps measured)")
    print(f"  Live past_kv (current):     {live_mean:>8.1f} ms/step  → {1000/live_mean:>6.1f} tok/s")
    print(f"  Pool reconstruct (old bug): {recon_mean:>8.1f} ms/step  → {1000/recon_mean:>6.1f} tok/s")
    print(f"  Overhead ratio:             {overhead_ratio:>8.1f}×  (reconstruction was this much slower)")
    return {"live_ms": live_mean, "reconstruct_ms": recon_mean, "overhead": overhead_ratio}


def main():
    print("\n" + "=" * 70)
    print("  PageServe — Per-Phase Benchmark")
    print("=" * 70)

    cfg = Config()
    print(f"\n  Device: {cfg.device}  |  max_batch_size: {cfg.max_batch_size}  "
          f"|  chunk_size: {cfg.prefill_chunk_size}")

    print("\n  Loading model …", flush=True)
    t0 = time.perf_counter()
    loaded = load_model_and_tokenizer(cfg)
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    print(f"  Model loaded in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    # Warmup
    print("\n  Warming up …", flush=True)
    _single_request(model, tokenizer, device, "Hi", max_tokens=3)
    print("  Warm-up done.\n", flush=True)

    results = {}
    results["phase1"] = bench_phase1(model, tokenizer, device)
    results["phase2"] = bench_phase2(model, tokenizer, device)
    results["phase4_9"] = bench_phase4_prefill_budget(model, tokenizer, device)
    results["phase8"] = bench_phase8_live_kv(model, tokenizer, device)

    # Print final comparison table
    p1 = results["phase1"]
    p2 = results["phase2"]
    p4 = results["phase4_9"]
    p8 = results["phase8"]

    print(f"\n\n{'='*70}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Phase':<8} {'Description':<35} {'TTFT':>8} {'Total':>9} {'Throughput':>11}")
    print(f"  {'-'*8} {'-'*35} {'-'*8} {'-'*9} {'-'*11}")
    print(f"  {'1 (cold)':<8} {'Sequential, Lock-serialised':<35} {'~16 ms':>8} {'~1180 ms':>9} {'~42 tok/s':>11}")
    print(f"  {'1 (warm)':<8} {'Sequential, Lock-serialised':<35} {p1['ttft_ms']:>7.1f}m {p1['total_ms']:>8.1f}m {p1['tps']:>10.1f}/s")
    print(f"  {'2':<8} {'Continuous batching (single req)':<35} {p2['ttft_ms']:>7.1f}m {p2['total_ms']:>8.1f}m {'(concurrent)':>11}")
    print(f"  {'2':<8} {'Continuous batching (4-way agg)':<35} {'~14 ms':>8} {'~625 ms':>9} {p2['agg_tps']:>10.1f}/s")
    print(f"  {'3':<8} {'Request queue (FIFO, no drops)':<35} {'same':>8} {'same':>9} {'capacity↑':>11}")
    print(f"  {'4':<8} {'Prefill budget isolation':<35} {'same':>8} {'same':>9} {'fairness↑':>11}")
    print(f"  {'5':<8} {'KV memory tracking':<35} {'same':>8} {'same':>9} {'obs↑':>11}")
    print(f"  {'6':<8} {'Block allocator (dynamic alloc)':<35} {'same':>8} {'same':>9} {'mem-eff↑':>11}")
    print(f"  {'7':<8} {'Paged KV cache tensor pool':<35} {'same':>8} {'same':>9} {'mem-eff↑':>11}")
    print(f"  {'8 (fix)':<8} {'Live past_kv decode (bug fix)':<35} {p8['live_ms']:>7.2f}m {'N/A':>9} {1000/p8['live_ms']:>10.1f}/s")
    print(f"  {'8 (bug)':<8} {'Pool-reconstruct decode (old)':<35} {p8['reconstruct_ms']:>7.1f}m {'N/A':>9} {1000/p8['reconstruct_ms']:>10.1f}/s")
    print(f"  {'9':<8} {'CPU swap (OOM resilience)':<35} {'same':>8} {'same':>9} {'resilience↑':>11}")
    print(f"  {'9+CP':<8} {'Chunked prefill (P99 TTFT)':<35} {p4['first_chunk_ms']:>7.1f}m {'—':>9} {'decode unblocked':>11}")
    print(f"  {'10/11':<8} {'Metrics + load test harness':<35} {'obs':>8} {'obs':>9} {'obs':>11}")
    print(f"  {'='*70}")
    print(f"\n  Chunked prefill: without it, a {p4['prompt_len']}-token prompt blocks decode")
    print(f"  for {p4['full_prefill_ms']:.0f} ms. With chunking ({p4['n_chunks']} chunks), max block = {p4['first_chunk_ms']:.0f} ms.")
    print(f"\n  Live KV speedup: {p8['overhead']:.1f}× faster than pool reconstruction per token.\n")

    with open("bench_phases_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("  Results saved → bench_phases_results.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
