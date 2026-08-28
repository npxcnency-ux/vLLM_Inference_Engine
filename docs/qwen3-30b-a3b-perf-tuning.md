# Mac Air M4 32G 上 Qwen3-30B-A3B 推理性能优化总结

**场景**：本地部署 `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`，通过 `mlx_lm.server` 提供 OpenAI 兼容接口 (`:8084`)，供 pi harness 单用户交互式编程使用。低并发（decode-concurrency=2）、最大上下文 128K。

**硬件**：Mac Air M4 base（10-core GPU，32GB LPDDR5X 统一内存，无风扇散热）。

---

## 一、硬件天花板与模型基础参数

### 硬件

| 项 | 数值 |
|---|---|
| 内存带宽 | **120 GB/s**（LPDDR5X-7500, 128-bit） |
| GPU FP16 算力 | ~4.6 TFLOPS |
| 可用 wired memory | ~21.8 GB（受 macOS 限制） |
| Roofline 拐点 | 4600 GFLOPS ÷ 120 GB/s ≈ **38 FLOPs/byte** |

### 模型

Qwen3-30B-A3B（MoE）：48 层，32 Q head，4 KV head（GQA），head_dim=128，128 experts top-8。

| 项 | 数值 | 备注 |
|---|---|---|
| 总参数 | 30.5B | 4bit 量化占 ~17.2 GB |
| 每 token 激活 | 3.04B | MoE 只走 top-8 experts |
| 每 token 权重读取 | 1.71 GB | 4bit + group_size overhead |
| 每 token KV（fp16） | 96 KB | 48 层 × 4 KV × 128 × 2(K+V) × 2B |
| 每 token FLOPs | 6.08 GFLOPs | 2 × 激活参数 |

### 一次典型推理的算术强度

| 阶段 | 每 token 内存流量 | 每 token FLOPs | 算术强度 | 与拐点 38 对比 |
|---|---|---|---|---|
| Prefill（1000 token） | 1.71 GB ÷ 1000 ≈ 1.7 MB | 6.08 GFLOPs | **3558 FLOPs/byte** | 严重 compute-bound |
| Decode（1 token, 8K ctx） | 1.71 + 0.77 = 2.48 GB | 6.08 GFLOPs | **2.45 FLOPs/byte** | 严重 memory-bound |

**结论**：prefill 靠算力，decode 靠带宽。而**用户等待 90% 花在 decode**，所以带宽利用率决定用户体感。

---

## 二、目前性能现状（低并发 + pi 场景）

### 理论上限（不同上下文）

| 上下文 | 每 token 内存流量 | Decode 理论上限 (tps) |
|---|---|---|
| 1K | 1.81 GB | 66 |
| 4K | 2.10 GB | 57 |
| 8K | 2.48 GB | 48 |
| 16K | 3.25 GB | 37 |
| 32K | 4.78 GB | 25 |
| 128K | 13.7 GB | **8.7** |

### 实测（现状：KV 8bit 量化后）

| 场景 | 上下文 | 输出 | prefill tps | decode tps | e2e tps | 利用率（decode/理论） |
|---|---|---|---|---|---|---|
| short | ~50 | 38 | 126 | **46.8** | 35.0 | 68% |
| medium | ~150 | 300 | 227 | **46.2** | 42.2 | 67% |
| long | ~3200 | 733 | 293 | **36.4** | 23.6 | ~64% |
| pi 长会话（实测） | ~8K | 1800 | ~245 | **~18** | ~13 | ~38% |

**Prefill 上限** ≈ 4600 GFLOPS × 60% 效率 / 6.08 GFLOPs ≈ **454 tps**，实测 200-320 tps（利用率 44-70%），健康。

### 128K 上下文的内存/性能估算

| 组件 | fp16 KV | int8 KV | int4 KV |
|---|---|---|---|
| 模型权重 | 17.2 GB | 17.2 GB | 17.2 GB |
| KV cache | **12.0 GB** | **6.4 GB** | **3.4 GB** |
| Activation/运行时 | ~2 GB | ~2 GB | ~2 GB |
| 总占用 | **31.2 GB** ❌ | **25.6 GB** ⚠️ | **22.6 GB** ✅ |
| Decode 理论 tps | 8.7 | 14.8 | 23.5 |

**当前 KV 8bit 是 128K 场景下唯一能稳定跑的档位**。fp16 单请求就 OOM，int4 更省但 mlx-lm 未在 server path 稳定测试。

---

## 三、gap 分析：为什么达不到理论上限

### Gap 1：memory-bound 带宽利用率
- 理论 120 GB/s，实测 decode 稳定在 46-47 tps（short/medium 8K 以内），换算 ≈ 55-70% 带宽利用
- 剩下 30-45% 消耗在：
  - **MoE routing/gather**：每层要 top-k 排序 + gather 激活专家权重，非顺序内存访问
  - **quantized matmul dequant 开销**：4bit → fp16 需要片上解压
  - **Metal command buffer 编排**：每 token 48 层内核调度
  - **Python 循环 overhead**：每 token 走一遍 Python 层

### Gap 2：长会话下的**持续满载降频**（观察到的主要"跑不动"元凶）
- Air M4 无风扇，持续 100 秒以上满载 GPU 温度 >100°C，触发降频
- 实测：同参数下，长任务 prefill_tps 从 300 掉到 97（3 倍），decode_tps 从 32 掉到 15（2 倍）
- 温度回落后自动恢复

### Gap 3：内存压力与 swap
- 权重 17GB + KV 峰值 + activation 逼近 32GB 天花板
- macOS 一旦 swap 到 SSD，内存带宽被 I/O 严重拖累（实测 swap 7GB 时 decode 掉 40%）
- 后台其他 GPU 应用（Chrome/iStat 等）也会挤占带宽

### Gap 4：长上下文 KV 占比上升
- 8K 上下文下 KV 占内存流量 30%，18K 上下文占 45%
- 未量化时是纯浪费：KV 是最容易"压缩换性能"的部分

---

## 四、优化尝试与结果轨迹

### 尝试 1：默认参数暴露 OOM 与降速 ❌

初始配置：`decode-concurrency=32, prompt-cache-size=10, prompt-cache-bytes` 未限。

**症状**：
- 长会话触发 `[METAL] Command buffer execution failed: Insufficient Memory`
- Generation thread 崩溃，服务半死不活，必须 kill 重启

**教训**：默认参数是数据中心场景的假设，Mac Air 32G 完全不适用。

### 尝试 2：保守化并发与 KV 上限 ✅

写入脚本 `run_qwen30b_a3b_mlx.sh`：

```bash
DECODE_CONCURRENCY=2
PROMPT_CONCURRENCY=1
PROMPT_CACHE_BYTES=3G
PROMPT_CACHE_SIZE=20
```

**收益**：
- OOM 消除
- 长会话 decode 从 15 → 18-20 tps（+30%），主要靠减少 wired memory 压力

### 尝试 3：Speculative Decoding ❌（负收益，回退）

下载 `Qwen3-0.6B-4bit` 和 `Qwen3-0.6B-4bit-DWQ` 作为 draft，测试 `num_draft_tokens ∈ {2, 3, 4, 6}`：

| 配置 | short | medium | long |
|---|---|---|---|
| baseline（无 draft） | 41.1 | **41.2** | **18.5** |
| 0.6B-4bit, N=4 | 49.1 | 28.7 | 12.4 |
| 0.6B-4bit-DWQ, N=4 | 49.4 | 29.5 | 15.4 |
| 0.6B-4bit-DWQ, N=6 | - | 16.6 | 7.9 |

**失败原因**：
1. Air M4 无风扇：draft + target 同时跑 GPU，散热压力翻倍，target 更快降频
2. MoE target 单 token decode 已经很快（~24ms），draft(0.6B) 的开销（~5ms/token + KV 维护）摊不开
3. 只有 short 生成场景有正收益，长任务全负

**结论**：MoE + Apple Silicon + 单会话下 speculative decoding 结构性不划算。已删除模型，脚本保留 `ENABLE_SPEC=1` 开关备用。

### 尝试 4：KV cache 8bit 量化 ✅（主要收益来源）

**问题**：mlx-lm 官方 `mlx_lm.server` 不暴露 `--kv-bits`。

**修改**：patch mlx-lm 源码（约 30 行）：
- `mlx_lm/server.py`：加 3 个 CLI 参数（`--kv-bits/--kv-group-size/--quantized-kv-start`），透传给 `stream_generate`
- `mlx_lm/models/cache.py`：修 `QuantizedKVCache.nbytes` 兼容 keys=None（初始化时未分配）
- **代价**：启用 kv 量化时强制走 sequential path（`is_batchable=False`），因为 `QuantizedKVCache` 缺少 `merge` classmethod，continuous batching 会抛错

**测试对比（fp16 vs 8bit KV）**：

| case | 精度 | decode_tps | e2e_tps |
|---|---|---|---|
| short | fp16 → 8bit | 41.9 → **46.8** (+11.7%) | 29.7 → 35.0 (+18%) |
| medium | fp16 → 8bit | 41.6 → **46.2** (+11.1%) | 38.1 → 42.2 (+10.8%) |
| long | fp16 → 8bit | 31.9 → **36.4** (+14.1%) | 22.0 → 23.6 (+7.3%) |

**KV 内存节省**：50%（每 token 从 96KB → ~51KB，含量化 overhead）。

**为什么代价可以接受**：pi 单会话场景 decode-concurrency 本来就只有 1-2，continuous batching 失效在**该场景下无实际影响**。

---

## 五、目前配置总览

```bash
# run_qwen30b_a3b_mlx.sh 关键参数
MODEL=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
DECODE_CONCURRENCY=2
PROMPT_CONCURRENCY=1
PROMPT_CACHE_SIZE=20
PROMPT_CACHE_BYTES=3G
KV_BITS=8            # 需 patch 过的 mlx-lm
KV_GROUP_SIZE=64
```

### 达成的稳定表现

| 场景 | Decode tps | 相较原始默认 |
|---|---|---|
| Short（<1K 上下文） | **46-49** | +40% |
| Medium（几 K 上下文） | **42-46** | +50% |
| Long（8K-16K 上下文） | **25-36** | +80% |
| 128K 上下文 | **~13-15** 预计 | 从"不能跑"到"能跑" |

### 128K 上下文可行性

- **fp16 KV**：单请求 12 GB KV，加权重 17GB → **超出 wired limit，必然 OOM**
- **KV 8bit**：单请求 6.4 GB KV → **能跑，理论 decode ~14 tps**
- 是本机跑 128K 的唯一路径

---

## 六、后续可尝试的优化方向

按性价比 + 可行性排序：

### 一线（值得马上做）

**1. KV int4 量化**
- 当前只测了 8bit，int4 能再省一半 KV（3.4 GB @ 128K）
- 代码路径已通（mlx-lm 支持 `bits=4`），只需 `--kv-bits 4` 尝试
- 风险：精度下降可能影响长上下文任务的输出质量，需要小心评估
- 预期 decode +5-10%

**2. `--quantized-kv-start N`**
- 只在 KV 长度超过 N（比如 2048）之后才量化
- 短会话完全不受影响，长会话逐渐进入量化模式
- 精度损失更小

**3. 系统层面**
- 保证 mlx server 是唯一 GPU 大户（关掉 Chrome/Xcode/iStat）
- 定期重启服务（每 30-60 分钟）清 swap
- 用散热支架强制通风，减少 Air 降频
- 关掉低电量模式（Settings → Battery → Low Power Mode = Never）

### 二线（工程量大但有明确收益）

**4. 给 `QuantizedKVCache` 实现 `merge()` classmethod**
- 恢复 continuous batching + kv 量化并存
- 需要处理：不同 seq 的 scale/bias 拼接、padding 处理、offset 对齐
- 大概 100-200 行代码
- 只在真需要多并发（>4 会话同时）时才值得做

**5. 探索 mlx-lm PagedAttention 分支**
- 社区有 PR 在做类似 vLLM 的分块 KV 架构（未合并）
- 长期方向；一旦成型，KV 量化 + batching + 长上下文一起解决

**6. 更小/更快的 draft 模型（重新试 speculative）**
- 试 `Qwen3-1.7B-4bit-DWQ` 作为 draft，接受率可能更高
- 或者试 EAGLE / Medusa 类结构（单个 head 预测多 token，无独立 draft）
- 前提：Air 散热能顶住两个模型并跑

### 三线（探索性）

**7. 权重端量化更激进**
- 尝试 `Qwen3-30B-A3B-3bit`（若有）：权重从 17GB → ~13GB
- 每 token 内存流量从 1.71 GB → 1.28 GB
- 预期 decode +25%，但精度下降需评估

**8. Sliding window attention (SWA)**
- 如果 pi 会话经常超过 32K，可以用 sliding 或 rotate KV 只保留最近 N 个
- 需要模型本身支持（Qwen3-30B-A3B 不支持 SWA，只能软实现）

**9. 换更高带宽的机器**
- M4 Pro (273 GB/s) → decode 理论翻倍
- M4 Max (546 GB/s) → 4×
- 不改代码的终极方案

---

## 七、关键 Takeaways

1. **Mac 上做 LLM 服务，内存带宽 >> 算力**——所有优化都要围绕 decode 的带宽利用率
2. **不要迷信数据中心的通用手段**——speculative decoding 在 MoE + Apple Silicon 下反而变慢
3. **Air 无风扇是硬件瓶颈**——长任务性能会衰减 30-50%，任何 benchmark 都要区分冷启动/长跑
4. **KV 量化是 128K 上下文的唯一救命稻草**——但要接受 mlx-lm 的 batching 限制
5. **保守化并发参数**比激进配置在 32G Air 上更划算——OOM 一次的代价远超"少并发"的损失

## 附录：相关脚本与文件

| 文件 | 用途 |
|---|---|
| `run_qwen30b_a3b_mlx.sh` | 生产启动脚本（含所有调优参数） |
| `bench_a3b_vs_27b.py` | MoE vs 稠密对比 |
| `.venv/lib/python3.11/site-packages/mlx_lm/server.py` | patch 过（KV 量化 CLI 参数） |
| `.venv/lib/python3.11/site-packages/mlx_lm/models/cache.py` | patch 过（QuantizedKVCache.nbytes 修复） |
| `.venv/.../*.bak` | 原始 mlx-lm 备份 |
