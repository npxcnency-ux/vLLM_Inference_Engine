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
# thinking 模式：由调用方决定，服务端保持中立（不硬设）。
#   OpenAI 客户端在请求体传 chat_template_kwargs.enable_thinking 控制；
#   pi 通过 models.json 的 chatTemplateKwargs 映射到其 thinking level。
#   注意：Qwen3.8 模型自身默认 enable_thinking=true（xhigh 疯狂过度思考，
#   画个圆圈想 21 分钟），所以调用方应显式传 enable_thinking=false 才快。
#   可选：设 ENABLE_THINKING=true/false 环境变量让服务端兜底传一个默认值
#   （配 REASONING_EFFORT=low/medium/xhigh）；不设则完全交给调用方。
#
# 提供 OpenAI 兼容接口 (/v1/chat/completions)，端口 8083。
# 与 PageServe (:8001) 和 Gemma MLX (:8082) 互不干扰，可共存（但 27B 吃 ~15GB，
# 同时跑多个大模型会内存吃紧，建议单独跑）。
#
# ── 调用方式 ────────────────────────────────────────────────────────────────
# OpenAI 兼容接口 POST /v1/chat/completions。thinking 由调用方在请求体控制：
# 加 chat_template_kwargs.enable_thinking（false=直接答/快，true=思考/慢）。
# 不传则走模型默认（xhigh 思考，很慢）——想快务必显式传 false。
#
#   curl http://localhost:8083/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "mlx-community/Qwen3.8-27B-4bit",
#       "messages": [{"role":"user","content":"写一首关于重阳节思念亲人的七言诗"}],
#       "max_tokens": 4096,
#       "chat_template_kwargs": {"enable_thinking": false}
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

# thinking 默认交给调用方决定（pi 等 OpenAI 客户端通过请求体的
# chat_template_kwargs.enable_thinking 控制），服务端不硬设，保持中立。
# 仅当显式设置 ENABLE_THINKING 环境变量时，才作为服务端兜底默认传入
# （未设则完全不传 --chat-template-args；此时不传的请求会走模型自身
# 默认，即 Qwen3.8 的 xhigh 思考——调用方应自行传 enable_thinking）。
ARGS=(--model "$MODEL" --host 0.0.0.0 --port "$PORT")
if [ -n "${ENABLE_THINKING:-}" ]; then
    if [ "$ENABLE_THINKING" = "true" ]; then
        ARGS+=(--chat-template-args "{\"enable_thinking\":true,\"reasoning_effort\":\"${REASONING_EFFORT:-low}\"}")
    else
        ARGS+=(--chat-template-args "{\"enable_thinking\":false}")
    fi
    echo "启动 Qwen3.8-27B MLX server: port=$PORT (服务端兜底 enable_thinking=$ENABLE_THINKING)"
else
    echo "启动 Qwen3.8-27B MLX server: port=$PORT (thinking 由调用方决定)"
fi

exec mlx_lm.server "${ARGS[@]}"
