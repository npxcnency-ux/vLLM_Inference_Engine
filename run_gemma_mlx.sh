#!/usr/bin/env bash
# run_gemma_mlx.sh — 在 Apple Silicon Mac 上用 MLX 跑 Gemma-4-E4B-it
#
# 为什么用 MLX 而非 PageServe(PyTorch)：
#   PyTorch MPS 后端的 bf16/fp16 算子精度不足，Gemma 4 长文本会退化成单字重复
#   （详见 docs/specs/gemma4-e4b-adaptation.md §6.5）。MLX 是 Apple 官方框架，
#   独立实现，不受此缺陷影响：8bit 量化质量正确、~20 tok/s、内存 ~8.5GB。
#
# 提供 OpenAI 兼容接口 (/v1/chat/completions)，端口 8082。
# 与 PageServe (:8001, PyTorch, 跑 Qwen) 互不干扰，可共存。
#
# ── 调用方式 ────────────────────────────────────────────────────────────────
# OpenAI 兼容接口 POST /v1/chat/completions。
# 注意：Gemma 4 默认开 thinking，会先输出思考过程占满 max_tokens 导致正文为空；
# 直接要结果时传 chat_template_kwargs.enable_thinking=false。
#  "max_tokens": 4096, 最大128k～=131072
#
#   curl http://localhost:8082/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "mlx-community/gemma-4-e4b-it-8bit",
#       "messages": [{"role":"user","content":"请写一首关于重阳节思念亲人的诗"}],
#       "max_tokens": 4096,
#       "chat_template_kwargs": {"enable_thinking": false}
#     }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# 正文在 choices[0].message.content；开 thinking 时思考过程在 reasoning_content。

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# hf-mirror + 禁用 Xet（无 HF token 时大文件走 Xet 会 401）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

MODEL="${MODEL:-mlx-community/gemma-4-e4b-it-8bit}"
PORT="${PORT:-8082}"

echo "启动 MLX server: model=$MODEL port=$PORT"
exec mlx_lm.server --model "$MODEL" --host 0.0.0.0 --port "$PORT"
