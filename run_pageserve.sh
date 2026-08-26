#!/usr/bin/env bash
# run_pageserve.sh — 启动 PageServe 连续批处理推理引擎（Phase 2, :8001）
#
# PageServe 是本项目自研的 PyTorch 推理引擎（连续批处理 + 分页 KV + CPU swap）。
# 默认跑 Qwen2-0.5B-Instruct —— Qwen2 在 Apple MPS 上低精度支持成熟、质量正常。
#
# 注意：不要用它跑 Gemma 4（PyTorch MPS 后端精度缺陷会导致长文本退化，
# 详见 docs/specs/gemma4-e4b-adaptation.md §6.5）。Gemma 4 用 MLX：run_gemma_mlx.sh。
#
# 与 MLX 服务 (:8082) 互不干扰，可共存。
#
# ── 调用方式 ────────────────────────────────────────────────────────────────
# PageServe 自有接口 POST /generate（非 OpenAI 格式）。Qwen 用 ChatML 模板：
#
#   curl -X POST http://localhost:8001/generate \
#     -H "Content-Type: application/json" \
#     -d '{
#       "prompt": "<|im_start|>user\n中国首都是哪座城市？<|im_end|>\n<|im_start|>assistant\n",
#       "max_new_tokens": 100
#     }'
#
# 遥测：curl http://localhost:8001/metrics   健康：curl http://localhost:8001/health

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# hf-mirror + 禁用 Xet（无 HF token 时大文件走 Xet 会 401）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

# 模型可通过环境变量覆盖（config.py 会读 MODEL_NAME）
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
PORT="${PORT:-8001}"

echo "启动 PageServe: model=$MODEL_NAME port=$PORT"
exec python -m uvicorn inference_engine.server.app_v2:app --host 0.0.0.0 --port "$PORT"
