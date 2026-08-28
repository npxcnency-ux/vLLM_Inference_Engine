# mlx-lm KV cache quantization patch

## 为什么需要

`mlx_lm.server`（截至 0.31.3）**不暴露 `--kv-bits`**，只有 CLI 工具 (`mlx_lm.generate`) 支持 KV 量化。本 patch 把三个参数补到 server：

| 参数 | 用途 |
|---|---|
| `--kv-bits` | 4 或 8；不传就是 fp16（默认行为）|
| `--kv-group-size` | 量化 group 大小，默认 64 |
| `--quantized-kv-start` | 从第 N 个 token 起才量化，0 表示全程 |

## 收益

在 Mac Air M4 32GB + Qwen3-30B-A3B-Instruct-2507-4bit 上实测：

| 指标 | fp16 KV | 8bit KV | 变化 |
|---|---|---|---|
| decode tps (short) | 41.9 | 46.8 | **+11.7%** |
| decode tps (long, 8K ctx) | 31.9 | 36.4 | **+14.1%** |
| 每 token KV 字节 | 96 KB | ~51 KB | **-47%** |
| 128K 上下文 KV 总量 | 12 GB (OOM) | 6.4 GB | **-47%** |

详细分析见 [`docs/qwen3-30b-a3b-perf-tuning.md`](../docs/qwen3-30b-a3b-perf-tuning.md)。

## 使用方法

```bash
# 应用 patch
./patches/apply-mlx-lm-kv-quant.sh

# 恢复原版
./patches/apply-mlx-lm-kv-quant.sh --revert
```

脚本会自动定位 `.venv` 里的 mlx-lm，打完 patch 后运行 `mlx_lm.server --help` 校验参数是否生效。

## 已知代价

**启用 `--kv-bits` 时会自动禁用 continuous batching**（`is_batchable = False`），因为 `QuantizedKVCache` 缺少 `merge()` classmethod，`_merge_caches` 会抛错。

对 pi 单会话（decode-concurrency=1-2）**无实际影响**；对多用户/大 batch 服务不适合。

## 兼容性

- 基于 **mlx-lm 0.31.3** 制作，其他版本会尝试 fuzzy match，可能失败
- 只改两个文件：`mlx_lm/server.py`（+30 行）、`mlx_lm/models/cache.py`（+2 行）
- 完全可逆（`patch -R`）

## Patch 内容

**`mlx_lm/server.py`**（5 处改动）：
1. import 加 `maybe_quantize_kv_cache`
2. `ModelProvider.__init__` 检测到 kv_bits 时强制 `is_batchable=False`
3. `_serve_single` 里给 `stream_generate` 透传 kv_bits/kv_group_size/quantized_kv_start
4. 启动日志打印 KV 量化状态
5. argparse 新增 `--kv-bits/--kv-group-size/--quantized-kv-start`

**`mlx_lm/models/cache.py`**（1 处改动）：
- `QuantizedKVCache.nbytes` 属性兼容 `keys=None`（初始化时未分配的情况）

## 长期方向

想要"量化 KV + continuous batching 并存"，需要给 `QuantizedKVCache` 实现 `merge()` classmethod（处理不同 seq 的 scale/bias 拼接、padding 对齐等）。约 100-200 行工作量。等 mlx-lm 官方 PagedAttention 分支合并后本 patch 会失效但也不再需要。
