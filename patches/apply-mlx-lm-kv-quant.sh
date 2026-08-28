#!/usr/bin/env bash
# apply-mlx-lm-kv-quant.sh — 给已安装的 mlx-lm 打上 KV 量化 patch
#
# 前提：项目内已有 .venv，且里面 pip 装了 mlx-lm。
# 用法：
#   ./patches/apply-mlx-lm-kv-quant.sh          # 打 patch
#   ./patches/apply-mlx-lm-kv-quant.sh --revert # 恢复原版
#
# 只影响 .venv 里的 mlx-lm，不动系统 python。

set -euo pipefail

cd "$(dirname "$0")/.."
VENV=".venv"
PATCH="patches/mlx-lm-kv-quant.patch"

if [ ! -d "$VENV" ]; then
    echo "错误：找不到 $VENV，请先 python3 -m venv .venv 并 pip install mlx-lm"
    exit 1
fi

# 定位 mlx_lm 安装目录
MLX_DIR=$("$VENV/bin/python3" -c "import mlx_lm, os; print(os.path.dirname(os.path.dirname(mlx_lm.__file__)))")
echo "mlx-lm 目录: $MLX_DIR"

MLX_VER=$("$VENV/bin/pip" show mlx-lm 2>/dev/null | awk '/^Version:/ {print $2}')
echo "mlx-lm 版本: $MLX_VER"
if [ "$MLX_VER" != "0.31.3" ]; then
    echo "⚠️  警告：本 patch 基于 mlx-lm 0.31.3 制作，当前版本 $MLX_VER 可能有偏移"
    echo "    继续会尝试 --fuzz 模糊匹配；如失败请：pip install mlx-lm==0.31.3"
fi

MODE="${1:-apply}"

if [ "$MODE" = "--revert" ] || [ "$MODE" = "-R" ]; then
    echo "=== 恢复原版 ==="
    patch -R -p1 -d "$MLX_DIR" < "$PATCH"
    echo "✅ 已恢复为原版 mlx-lm"
else
    echo "=== 应用 patch ==="
    # -N: 已应用就跳过；--dry-run 先探测
    if patch --dry-run -p1 -d "$MLX_DIR" < "$PATCH" >/dev/null 2>&1; then
        patch -p1 -d "$MLX_DIR" < "$PATCH"
        echo "✅ patch 应用成功"
    else
        # 可能已经打过；检测下
        if grep -q "kv-bits" "$MLX_DIR/mlx_lm/server.py"; then
            echo "ℹ️  server.py 里已有 --kv-bits，patch 应已应用，跳过"
        else
            echo "❌ patch 应用失败，试着 --fuzz 强制："
            patch -p1 --fuzz=3 -d "$MLX_DIR" < "$PATCH"
        fi
    fi
fi

echo ""
echo "=== 校验 ==="
"$VENV/bin/mlx_lm.server" --help 2>&1 | grep -E "kv-bits|kv-group-size|quantized-kv-start" \
    && echo "✅ 参数已生效" \
    || echo "❌ 未看到 kv-bits，请检查"
