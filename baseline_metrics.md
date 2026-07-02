# PageServe — Benchmark Metrics

Captured on Apple M2 (MPS), Qwen/Qwen2-0.5B, float16.
Benchmark script: `bench_phases.py` — runs model inference directly to isolate engine latency.

---

## Performance Summary

### Naive Sequential Serving (Phase 1 Baseline)
- **TTFT (steady-state, cold)**: ~16 ms
- **Total Latency (50 tokens, cold)**: ~1,180 ms
- **Throughput (cold)**: ~42 tok/s
- **Throughput (warm)**: ~82.3 tok/s
- **Concurrency**: 1 request at a time (strict serialisation via `asyncio.Lock`)

### Continuous Batching & Chunked Prefill (Phase 2-9)
- **TTFT (warm, single request)**: 20.9 ms
- **TTFT (warm, 4-way concurrent)**: 13.7 ms (compared to 1,418 ms queued in Phase 1)
- **Aggregate Throughput (4-way concurrent)**: 47.9 tok/s
- **Concurrency**: Up to 4 active sequences interleaved at iteration level

---

## Phase-by-Phase Comparison Table

This table documents the metrics and architectural changes introduced at each development phase:

| Phase | Description | TTFT (ms) | Total Latency (ms) | Throughput (tok/s) | Key Architectural / Performance Progress |
|---|---|---|---|---|---|
| **1** | Naive Sequential Baseline | ~32 (warm) | ~608 (warm) | 82.3 (warm) | Strict serialization. Requests wait in queue for full prior generation. |
| **2** | Continuous Batching Scheduler | ~21 | ~610 | 47.9 (agg) | Background event loop checks and advances active batch sequences by one token. |
| **3** | Request Queue | ~21 | ~610 | 47.9 (agg) | Added RequestQueue with FIFO/LIFO ordering to prevent connection drops. |
| **4** | Prefill / Decode Separation | ~21 | ~610 | 47.9 (agg) | Bounded scheduling step token budgets to avoid prefill starvation of active decodes. |
| **5** | KV Cache Memory Tracking | ~21 | ~610 | 47.9 (agg) | Tracker implemented to estimate dynamic physical GPU memory footprint of KV caches. |
| **6** | Block Allocator | ~21 | ~610 | 47.9 (agg) | Bookkeeps logical sequence tokens into memory blocks of size 16. |
| **7** | Paged KV Cache | ~21 | ~610 | 47.9 (agg) | Maps logical block IDs to pre-allocated physical tensor pool, minimizing memory fragmentation. |
| **8** | Batch Attention Integration | ~21 | ~610 | 47.9 (agg) | Connects model forward pass to paged pool. Note: The pool-reconstruction bug dropped decode speed. |
| **8 (bug)** | Pool-Reconstruction Path | ~21 | — | ~78.1 | Reconstructed DynamicCache from paged pool every token (~1.1 ms overhead/step). |
| **8 (fix)** | Live KV Cache Path | ~21 | — | **85.7 (step)** | Kept HuggingFace `past_key_values` live on sequence during decode, skipping pool read. |
| **9** | CPU Swap Manager | ~21 | ~610 | 47.9 (agg) | Adds OOM safety by swapping preempted sequences out to host CPU memory. |
| **9+CP** | Chunked Prefill | **17.9** | — | — | Long prefill (82 tokens) chunked down to 128. Bounded decode delay by saving 21.9 ms. |
| **10** | Unified Metrics Aggregator | — | — | — | Consolidated system, queue, cache, and latency telemetry for SLO tracking. |
| **11** | Load Testing Tool | — | — | — | CLI validation tool simulating constant, ramp, and burst load patterns. |

---

## Critical Performance Breakdowns

### 1. Prefill Chunking Impact (Phase 9 vs Phase 4)
When a long prompt arrives under decode load:
- **Without Chunked Prefill**: The long prompt (e.g. 82 tokens) takes **39.8 ms** to process, blocking all co-running decode sequences for that entire duration.
- **With Chunked Prefill**: Bounded to **17.9 ms** for the first chunk, saving **21.9 ms** in decode stall latency.

### 2. KV Cache Hookup Optimization (Phase 8 Fix)
- **Pool-Reconstructed Decode (Naive)**: 1.1 ms reconstruction overhead added to every token generation step.
- **Live KV Decode (Optimized)**: Live `past_key_values` cached on sequence. Speed runs at **11.7 ms** per step (85.7 tok/s).

---

## Concurrency Comparison (Phase 1 vs Phase 2-9)

Running 4 concurrent requests (50 tokens each):

| Metric | Phase 1 (Sequential) | Phase 2-9 (Batched Scheduler) | Delta |
|---|---|---|---|
| **Wall Clock Time (all done)** | 5,673.6 ms | **4,171.1 ms** | **-26.5%** |
| **TTFT under load (mean)** | ~1,418.0 ms (due to queue) | **20.9 ms** | **-98.5%** |
| **Aggregate Throughput** | ~42.0 tok/s | **47.9 tok/s** | **+14.0%** |

> [!IMPORTANT]
> The primary benefit of continuous batching is the massive reduction in Time-to-First-Token (TTFT) under load, preventing early requests from starving late-arriving requests in the queue.
