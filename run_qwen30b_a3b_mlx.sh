#!/usr/bin/env bash
# run_qwen30b_a3b_mlx.sh — 在 Apple Silicon Mac 上用 MLX 跑 Qwen3-30B-A3B-Instruct-2507（4bit）
#
# 为什么选它做日常编码 agent：
#   MoE 架构：总参数 30B，但每 token 只激活 ~3B → prefill/decode 都像小模型一样快，
#   质量接近同规模稠密。相比稠密 Qwen3.8-27B（每 token 过全部 27B、prefill ~37 tok/s、
#   decode ~16 tok/s），A3B 在 M4 上 TTFT 和吐字速度都快数倍，交互式体验完全不同。
#   代价：所有专家权重都得常驻内存（~17-18GB，比 27B 略高，32GB 仍够）。
#
# Instruct-2507 = 非思考版（Qwen 把 30B-A3B 拆成 Instruct=不思考 / Thinking=思考）。
#   本身没有 thinking 模式，不会像 Qwen3.8 那样跑飞去"想 21 分钟"，天生适合快速交互。
#
# 提供 OpenAI 兼容接口 (/v1/chat/completions)，端口 8084。
# 与 PageServe(:8001)、Gemma(:8082)、Qwen27B(:8083) 互不干扰，但大模型建议单独跑省内存。
#
# ── 常驻前缀缓存 ────────────────────────────────────────────────────────────
# mlx_lm.server 内置 LRUPromptCache（进程内、无 TTL、按 LRU 淘汰）。
#   - pi 单会话多轮：系统提示+历史是同一条持续增长的前缀，始终"最近使用"，永不被踢，
#     turn2+ 命中缓存秒回；冷 prefill 只付一次。
#   - 只要不重启本进程，缓存就一直在。--prompt-cache-size 调大可缓存更多并行会话前缀。
#
# ── 调用方式 ────────────────────────────────────────────────────────────────
#   curl http://localhost:8084/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
#       "messages": [{"role":"user","content":"用一句话说明快速排序的原理。"}],
#       "max_tokens": 512
#     }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# hf-mirror + 禁用 Xet（无 HF token 时大文件走 Xet 会 401）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

MODEL="${MODEL:-mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit}"
PORT="${PORT:-8084}"
# 缓存更多前缀（默认 10）；纯 LRU，无超时。进程重启即清空。
PROMPT_CACHE_SIZE="${PROMPT_CACHE_SIZE:-20}"

# ── 针对 Mac Air 32GB 的并发/KV 参数 ────────────────────────────────────────
# 内存预算：权重 ~17.2GB + macOS ~7GB + activation ~1-2GB → 留给 KV ~5-7GB。
# 每 token 每并发 KV = 48层 × 4 KV head × 128 × 2(K+V) × 2B = 96KB。
#
# DECODE_CONCURRENCY=6：编码 agent 典型上下文 4-16K，6 路并发 ≈ 4-5GB KV，
#   总占用 ~23-24GB，留 macOS 呼吸空间。MoE 并发越高专家激活越多，
#   per-request 吐字速度会掉得比稠密模型明显，6 是体感/吞吐拐点。
# PROMPT_CONCURRENCY=2：Air 无风扇，prefill 峰值功耗高，>2 容易撞降频，
#   保守 2 让 prefill 稳定不抖。
# PROMPT_CACHE_BYTES=6G：KV 总字节硬上限，防止单个超长 prompt（如 128K → 12GB）
#   直接把整机撑爆；超了自动 LRU 淘汰旧会话，不是拒收新请求。
#
# 使用场景切换：
#   批量离线短 prompt（<2K）：DECODE_CONCURRENCY=12 PROMPT_CONCURRENCY=3
#   单会话长上下文（32K+）：  DECODE_CONCURRENCY=1  PROMPT_CACHE_BYTES=12G
DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-2}"
PROMPT_CONCURRENCY="${PROMPT_CONCURRENCY:-1}"
PROMPT_CACHE_BYTES="${PROMPT_CACHE_BYTES:-3G}"

# ── Speculative decoding（默认关闭）────────────────────────────────────────────
# 实测结果：使用 Qwen3-0.6B-4bit 作为 draft，在中长会话下反而变慢 30%+。
# 原因：draft(base) 与 target(Instruct-2507) 分布差异大→ 接受率低，
# 加上 Air 无风扇→ draft 自身占 GPU 拖累 target。
# 如需开启：设置 ENABLE_SPEC=1（只建议 short prompt 场景，小幅提速）。
ENABLE_SPEC="${ENABLE_SPEC:-0}"
DRAFT_MODEL="${DRAFT_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-4}"

# ── KV cache 量化（需要 patch 过的 mlx-lm）────────────────────────────────────
# 效果：8bit 每 token KV 从 96KB→~51KB（省 ~50%），128K 上下文 KV 从 12GB→6.4GB。
# 实测 decode +10-14%（长会话最明显）；e2e +7-18%。
# ⚠️  依赖本地 patch：
#   1) mlx_lm/server.py 加了 --kv-bits/--kv-group-size/--quantized-kv-start 参数
#   2) mlx_lm/models/cache.py 修 QuantizedKVCache.nbytes 兼容 keys=None
#   3) 启用后强制走 sequential path（QuantizedKVCache 缺 merge classmethod）
#      → continuous batching 失效，但 pi 单会话场景无实际影响
# 详见 docs/qwen3-30b-a3b-perf-tuning.md
# 关闭量化：设置 KV_BITS=0（或空）。
KV_BITS="${KV_BITS:-8}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-64}"
QUANTIZED_KV_START="${QUANTIZED_KV_START:-0}"

EXTRA_ARGS=()
if [ "$ENABLE_SPEC" = "1" ]; then
    EXTRA_ARGS+=(--draft-model "$DRAFT_MODEL" --num-draft-tokens "$NUM_DRAFT_TOKENS")
fi
if [ -n "$KV_BITS" ] && [ "$KV_BITS" != "0" ]; then
    EXTRA_ARGS+=(--kv-bits "$KV_BITS" --kv-group-size "$KV_GROUP_SIZE" --quantized-kv-start "$QUANTIZED_KV_START")
fi

echo "启动 Qwen3-30B-A3B MLX server:"
echo "  model=$MODEL port=$PORT"
if [ "$ENABLE_SPEC" = "1" ]; then
    echo "  ⚠️  speculative=ON draft=$DRAFT_MODEL num-draft-tokens=$NUM_DRAFT_TOKENS"
else
    echo "  speculative=OFF (实测在本组合下不划算；如需开启: ENABLE_SPEC=1)"
fi
echo "  decode-concurrency=$DECODE_CONCURRENCY prompt-concurrency=$PROMPT_CONCURRENCY"
echo "  prompt-cache-size=$PROMPT_CACHE_SIZE prompt-cache-bytes=$PROMPT_CACHE_BYTES"
if [ -n "$KV_BITS" ] && [ "$KV_BITS" != "0" ]; then
    echo "  kv-bits=$KV_BITS kv-group-size=$KV_GROUP_SIZE quantized-kv-start=$QUANTIZED_KV_START (需 patch 过的 mlx-lm)"
else
    echo "  kv-bits=off"
fi
exec mlx_lm.server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --prompt-cache-size "$PROMPT_CACHE_SIZE" \
    --prompt-cache-bytes "$PROMPT_CACHE_BYTES" \
    --decode-concurrency "$DECODE_CONCURRENCY" \
    --prompt-concurrency "$PROMPT_CONCURRENCY" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
