#!/usr/bin/env bash
# run_qwen27b_mlx.sh — 在 Apple Silicon Mac 上用 MLX 跑 Qwen3.8-27B（4bit 量化）
#
# 为什么用 MLX 而非 PageServe(PyTorch)：
#   27B 的 fp16 权重 ~54GB，远超 32GB 内存，PageServe 无法加载（且不支持量化）。
#   MLX 4bit 量化把权重压到 ~15GB，32GB Mac 可跑。MLX 是 Apple 官方框架，
#   低精度实现正确、长文本不退化（不像 PyTorch MPS 上的 Gemma 4）。
#
# 实测（M4 / 32GB）：加载 8s，内存占用 ~15GB，短问答 ~16 tok/s，
#   长文本 ~6.5 tok/s（思考模式吐大量 reasoning token 拖慢）。质量优秀（能写七律 + 自检平仄）。
#
# thinking 模式：Qwen3.8-27B 默认开启思考（reasoning_effort 默认 xhigh），会疯狂
#   过度思考（画个圆圈想 21 分钟）、拖慢生成。这里默认 enable_thinking=false 直接
#   关闭思考，输出空的 <think></think> 后直接给答案，速度更快、无冗长推理。
#   如需开启思考，设 ENABLE_THINKING=true（可再配 REASONING_EFFORT=low/medium/xhigh），
#   或在单次请求的 chat_template_kwargs 里覆盖。
#
# 提供 OpenAI 兼容接口 (/v1/chat/completions)，端口 8083。
# 与 PageServe (:8001) 和 Gemma MLX (:8082) 互不干扰，可共存（但 27B 吃 ~15GB，
# 同时跑多个大模型会内存吃紧，建议单独跑）。
#
# ── 调用方式 ────────────────────────────────────────────────────────────────
# OpenAI 兼容接口 POST /v1/chat/completions。服务端已默认关闭思考（enable_thinking=false）；
# 单次请求可用 chat_template_kwargs 覆盖，如 {"enable_thinking": true}。
#
#   curl http://localhost:8083/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "mlx-community/Qwen3.8-27B-4bit",
#       "messages": [{"role":"user","content":"写一首关于重阳节思念亲人的七言诗"}],
#       "max_tokens": 4096
#     }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# 正文在 choices[0].message.content。27B 生成慢，长任务耐心等（几十秒~几分钟）。

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# hf-mirror + 禁用 Xet（无 HF token 时大文件走 Xet 会 401）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

MODEL="${MODEL:-mlx-community/Qwen3.8-27B-4bit}"
PORT="${PORT:-8083}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"

# 关闭思考时只传 enable_thinking=false（此时 reasoning_effort 无意义）；
# 开启思考时传 enable_thinking=true + reasoning_effort。
if [ "$ENABLE_THINKING" = "true" ]; then
    TEMPLATE_ARGS="{\"enable_thinking\":true,\"reasoning_effort\":\"$REASONING_EFFORT\"}"
else
    TEMPLATE_ARGS="{\"enable_thinking\":false}"
fi

echo "启动 Qwen3.8-27B MLX server: model=$MODEL port=$PORT enable_thinking=$ENABLE_THINKING"
exec mlx_lm.server --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --chat-template-args "$TEMPLATE_ARGS"
