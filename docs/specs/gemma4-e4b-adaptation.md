# Spec: Gemma-4-E4B-it 模型适配

**状态**: 已实现并验证
**日期**: 2026-08-25（MLX 结论补充 2026-08-26）
**影响范围**: `inference_engine/engine/{kv_cache_config,paged_kv_cache,cpu_swap_manager,attention_wrapper,scheduler}.py`、`inference_engine/models/loader.py`

---

## ⭐ TL;DR / 最终推荐（先读这个）

**在 Apple Silicon Mac 上跑 Gemma-4-E4B-it，用 MLX，不要用 PageServe(PyTorch)。**

- **正解**：`mlx_lm.server` + `mlx-community/gemma-4-e4b-it-8bit`。质量正确、~17-24 tok/s、内存 8.5GB。一键启动脚本：`inference/run_gemma_mlx.sh`（详见 §8）。
- **为什么不用 PageServe**：PyTorch 的 MPS 后端 bf16/fp16 算子精度不足，Gemma 4 长文本生成退化成单字重复（§6.5）。MPS 上只有 fp32 正确，但内存 30GB + 慢到不可用。这是 PyTorch MPS 的实现缺陷，**与 PageServe 引擎无关**。
- **本文档 §1-§5 记录的 PageServe 适配**（分页 KV/异构 head_dim/24层）**技术上正确且已完成**，在 CUDA 或未来修好的 MPS 上可用；只是在当前 Apple Silicon + PyTorch 组合下因 §6.5 的后端缺陷不可用于长文本。
- PageServe(:8001, PyTorch) 保持跑 Qwen 等 MPS 支持成熟的模型；MLX(:8082) 跑 Gemma 4。两者独立共存。

---

## 1. 背景与目标

PageServe 原本只支持 Qwen 类**均匀注意力**模型（所有层 head_dim 相同、无 KV 共享、纯全局注意力）。将默认模型换成 `google/gemma-4-E4B-it` 时，服务在启动阶段崩溃：

```
AttributeError: 'Gemma4Config' object has no attribute 'num_attention_heads'
```

**目标**: 改造引擎的 KV cache 数据结构与 config 读取逻辑，使 Gemma 4 能正确生成（含长序列 >512 token），同时保持 Qwen 等常规模型行为完全不变。

**非目标**: 不追求 CPU-swap 降级路径的完美生成质量（该路径原本对 Qwen 也存在语义漂移）；不支持 Gemma 4 的多模态输入（仅纯文本推理）。

---

## 2. Gemma 4 E4B 架构冲突（实测确认，transformers 5.15.1）

引擎的三个内建假设与 Gemma 4 冲突，另有一个滑窗特性影响重建路径：

| # | 冲突 | Gemma 4 实情 | 引擎原假设 |
|---|------|-------------|-----------|
| 1 | **config 嵌套** | 注意力参数在 `model.config.text_config` | 直接读顶层 `model.config` |
| 2 | **KV 共享层** | 42 hidden layers，后 18 层复用 KV（`num_kv_shared_layers=18`），实际 cache **仅 24 层** | cache 层数 = num_hidden_layers |
| 3 | **异构 head_dim** | 24 个 cache 层中，4 个全局层 head_dim=512，20 个滑窗层 head_dim=256 | 所有层 head_dim 相同 |
| 4 | **滑动窗口注意力** | 20 层为 `DynamicSlidingWindowLayer`（窗口 512），4 层为 `DynamicLayer`（全局） | 全部全局注意力 |

**关键实测数据**:
- `text_config`: num_hidden_layers=42, num_attention_heads=8, num_key_value_heads=2（所有层一致）, num_kv_shared_layers=18, is_heterogeneous=True
- 前 24 层 head_dim = `[256,256,256,256,256,512] × 4`，max=512，sum=7168
- 全局层（512）位于 cache 索引 5/11/17/23
- head_dim ≠ hidden_size//num_heads（2560/8=320），**必须读 config，不可除法推导**
- `text_config.head_dim` 直接访问抛 `AmbiguousGlobalPerLayerAttributeError`；须遍历 `text_config.per_layer_config[i].head_dim`
- 架构类 `Gemma4ForConditionalGeneration`（多模态），`AutoModelForCausalLM` 可加载，forward 签名兼容，额外加载 vision(0.33GB)+audio(0.61GB) 塔

---

## 3. 设计方案：padding 到 max head_dim

采用 **padding 到 max head_dim=512 保持规整五维张量** 的方案，而非拆成 per-layer 张量列表。

**理由**:
- 改动面最小，数据结构骨架（`[num_blocks, block_size, num_layers, num_kv_heads, head_dim]` 五维张量）不变
- 正常 decode 路径不从 pool 读（用活缓存 `seq.past_key_values`），padding 对生成无损
- 只需在写入/读回时按每层真实宽度切片；256 层只填前 256 列，尾部保持 0 padding
- 代价：256 层浪费一半尾维显存（padding 到 512），可接受

**精确显存计算**: `bytes_per_token = 2 × num_kv_heads × sum(head_dims) × dtype_bytes`，用 `sum(head_dims)` 而非 padding 后的 `num_layers × max`，避免高估。Gemma4 = 2×2×7168×2 = 57344 B ≈ 56KB/token。

---

## 4. 实现明细

### 4.1 `kv_cache_config.py`（地基）

**`KVCacheConfig` 数据类**:
- 新增 `head_dims: list[int] = field(default_factory=list)`，长度 = num_layers
- `head_dim: int` 语义改为 max(head_dims)，用于张量分配宽度
- `__post_init__` 兜底：`if not self.head_dims: self.head_dims = [self.head_dim] * self.num_layers`（向后兼容——旧测试/Qwen 只传标量时自动填充），随后 `self.head_dim = max(self.head_dims)`
- `bytes_per_token` 改为 `2 * num_kv_heads * sum(head_dims) * dtype_bytes`

**`compute_kv_cache_config`**:
- 取 text_config：`model.config.get_text_config() if hasattr(...) else model.config`
- cache 层数：`num_hidden_layers - (num_kv_shared_layers or 0)`
- 逐层 head_dim：若 `is_heterogeneous` 且有 `per_layer_config` → `[per_layer_config[i].head_dim for i in range(cache_layers)]`；否则 fallback（try 读标量 head_dim，None 则 hidden_size//num_heads）

### 4.2 `paged_kv_cache.py`（写读按真实宽度切）

- `__init__` 新增 `self.head_dims = kv_cache_config.head_dims`；池仍按 `self.head_dim`(=max) 分配
- `write_kv`: `hd = self.head_dims[layer_idx]`；`key_pool[block, slot, layer_idx, :, :hd] = key_tensor`
- `read_kv_sequence` / `read_kv_block`: 切片按 `:hd` 截断，空张量分支宽度用 `hd`

### 4.3 `cpu_swap_manager.py`（CPU 池按 max 分配）

- CPU 镜像池按 `kv_cache_config.head_dim`(max) 分配，与设备池 5D 形状严格一致
- swap_out/swap_in 保持逐层整槽 copy：两侧 padding 列都恒为 0，整槽 copy 语义正确（多拷 0 列属带宽浪费，不影响正确性）

### 4.4 `attention_wrapper.py` / `scheduler.py`（重建路径）

- `reconstruct_dynamic_cache` / `build_past_key_values` 新增可选参数 `model_config=None`；透传给 `DynamicCache(config=...)`（保留为通用能力）
- scheduler 两处 pool 重建调用（prefill 中间 chunk、decode/swap）——见 §5 决策

### 4.5 `models/loader.py`

- MPS 分支 dtype 从 `float16` 改为 `bfloat16`（Gemma 官方推荐，数值更稳）

---

## 5. 关键决策：swap 重建用普通 DynamicCache（偏离初始设计）

**初始设计**用 `DynamicCache(config=model_config)` 重建以保留滑窗语义。**实测发现这会导致字符级乱码**（如 "点头的所有者官难帖导"）。

**根因**: pool 重建是"一次性灌入全量历史 KV"，与滑窗层 `DynamicSlidingWindowLayer` 假设的"逐 token 流式 update"不兼容。灌入超窗口 KV 时，滑窗层截断到最近 511 而全局层保留全量，导致层间 seq_len 不齐、position/mask 错乱。即便未超窗口（如 194 token），config-aware 滑窗层的内部状态与单 token decode 前向也不匹配。

**最终决策**: scheduler 两处 pool 重建（`_prefill_chunk_blocking` 中间 chunk、`_decode_step_single` 的 swap/首次 decode 重建）都传 `model_config=None`，用普通 DynamicCache（全 full-attention 层）重建。`attention_wrapper` 的 `model_config` 参数保留但不使用（通用能力、向后兼容）。

**取舍**: 超长(>512) 序列走 swap 时丢失滑窗语义，但概率极低且只是精度下降非崩溃。swap 本就是显存压力降级路径（Qwen 亦有语义漂移）。核心目标——正常路径正确生成——已达成。

---

## 6. 验证

启动: `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 MODEL_NAME=google/gemma-4-E4B-it`

| 路径 | 判据 | 结果 |
|------|------|------|
| 启动冒烟 | 日志 `Layers: 24`、`per-layer distinct=[256, 512]`、bytes≈56KB | ✅ |
| 短序列中文（<512, 正常 decode 不读 pool） | 连贯无乱码 | ✅ `中国首都是北京。` |
| 长 prompt 多 chunk prefill（>128 token, 读回 pool） | 连贯无乱码 | ✅ |
| CPU-swap（小 KV_NUM_BLOCKS 制造压力） | 无字符级乱码 | ✅（修复后） |
| Qwen 回归 | 行为与改造前一致 | ✅ head_dims 兜底 |
| 单元测试 | 全绿 | ✅ 98 passed |

**已知限制**: swap 降级路径仍有轻微语义漂移/偶发空输出——原引擎既存问题（Qwen 同样存在），非本次引入。

**性能**: Gemma 8B 在 MPS 上 ~3-6 tok/s、TTFT ~11s、内存 ~15GB（模型体量决定，非引擎问题）。

---

## 6.5 重大已知问题：MPS 后端低精度算子精度不足导致生成退化

**症状**: 在 Apple MPS 上用 bf16/fp16 时，Gemma-4-E4B 生成的文本会退化——短问答（如"中国首都是北京。"）正常，但长文本（几十~上百 token）崩坏成单字重复（"酥酥酥"/"烟烟烟"/"老兮老兮"）。精度越低崩得越早。

**根因（经完整对照实验确证，与本引擎无关）**: PyTorch 的 Metal(MPS) 后端在低精度（bf16/fp16）下，各类算子（matmul / RMSNorm / RoPE / MLP 等）的精度普遍低于 CPU 参考实现——CPU 后端内部对这些算子做 fp32 累加保护，MPS 的低精度 Metal kernel 缺少此保护。误差**从第一个 transformer 层起就产生**（hs[1] 相对 CPU 已差 ~10%），逐层累积，在 42 层深网络 + 长 decode 自回归链中放大到压垮 argmax，塌成重复。**不是单一算子错误，是全模型低精度算子的普遍精度损失累积。**

**证据矩阵（关键：2 后端 × 3 精度）**:

| | bf16 | fp16 | fp32 |
|---|:---:|:---:|:---:|
| **CPU** | ✅ 正常 | ✅ 正常 | ✅ 正常 |
| **MPS** | ❌ 立刻崩 | ❌ 长序列崩 | ✅ 正常 |

- CPU 三种精度全对（低精度有内部 fp32 保护）；MPS 只有 fp32 对 → **锁定 MPS 低精度实现**
- 崩溃时机 ∝ 精度：bf16(7位尾数)立刻崩 < fp16(10位)撑到几十token < fp32(23位)全程稳
- 逐层 diff（同 bf16，纯比后端）：MPS vs CPU 从 hs[1] 的 ~10% 单调累积到 hs[42] 的 40%，final logits rel=102%；对比同实验 fp32 全程仅 ~1e-3
- prefill 首 token 两边 argmax 一致 → 误差需 decode 反复累积才压垮，非单次前向发散（排除单算子大错）
- 裸 HF `generate()`（完全绕过 PageServe）同精度同长度同样崩 → **排除引擎**

**排除的假设**: PageServe 引擎、transformers 建模代码、官方权重、tokenizer 模板、采样参数、bf16 精度理论极限（CPU bf16 完美）、单算子大错（逐层平滑无突跳）。

**为何常见 workaround 无效（均实测）**:
- `PYTORCH_ENABLE_MPS_FALLBACK=1`：只回退"未实现"（抛 NotImplementedError）的算子；此处算子都"有实现但精度不足"，fallback 不触发。无效。
- **attention 定点 upcast**：HF `eager_attention_forward` 已把 softmax upcast 到 fp32，但漏了 QK matmul 和 logit softcapping（在 bf16 下算）。实测 monkey-patch 把这两处也 upcast 到 fp32 后**仍崩**——证明坏点不止 attention，MLP/norm/rope 的逐层损失同样致命。只 upcast attention 不够；要 upcast 到全模型就等于 fp32。

**可行方案（按推荐度）**:
- **MPS + fp32**：MPS 上唯一长序列可靠的配置。代价：内存 ~30GB（32GB 机器紧张，需关闭其他占用）、速度慢。
- **换 CUDA 环境**：CUDA kernel（FlashAttention/vLLM）内部做 fp32 累加，bf16 即正常。引擎适配代码直接可用。
- **CPU 推理**：任意精度都正确，但 8B 在 CPU 上极慢，仅适合验证。
- **实用场景改用 Qwen2 系列**：架构老、MPS 低精度支持成熟，bf16 即快又好。
- 升级 PyTorch（MPS 对新架构的低精度支持在持续修复，未来可能改善）。

---

## 7. 使用

**注意 Gemma 4 的对话格式与 Gemma 2/3 不同**：用 `<|turn>role<turn|>`（单 token，id 105/106），**不是** `<start_of_turn>`（那是旧版格式，在 Gemma 4 tokenizer 里会被拆成 7 个碎 token）。正确做法是用 `tokenizer.apply_chat_template()` 生成 prompt，其规范渲染为：

```
<bos><|turn>user\n你的问题<turn|>\n<|turn>model\n
```

直接 curl 需手动拼这个格式：

```bash
curl -X POST http://localhost:8001/generate -H "Content-Type: application/json" -d '{
  "prompt":"<|turn>user\n你的问题<turn|>\n<|turn>model\n",
  "max_new_tokens":100
}'
```

（`<bos>` 由 tokenizer 自动添加，prompt 里可省略。）**但注意**：在 MPS 上无论模板是否正确，长文本生成都会因 §6.5 的数值问题退化。

**下载注意**: 无 HF token 从 hf-mirror 下载时须设 `HF_HUB_DISABLE_XET=1`，否则大权重文件走 Xet 后端会 401 失败。

---

## 8. MLX 独立服务（Apple Silicon 上的推荐方案）

因 §6.5 的 PyTorch MPS 缺陷，Apple Silicon 上跑 Gemma 4 的正解是 MLX（Apple 官方 ML 框架，独立实现，不受 PyTorch MPS 缺陷影响）。

### 8.1 为什么不给 PageServe 接 MLX 后端

探查（见下）表明接后端不划算：
- mlx_lm 的 cache（`KVCache`/`RotatingKVCache`/`QuantizedKVCache`）不暴露可自由读写的张量，MLX 数组又不可原地 mutate（函数式），**PageServe 的分页 KV 池 + 影子写入 + CPU swap 三件套无法套用**，MLX 路径下只能整体停用。
- mlx_lm 本身已内建更完整的能力：`BatchGenerator`（真·张量批处理）、`LRUPromptCache`+`PromptTrie`（前缀缓存，类 SGLang RadixAttention）、`QuantizedKVCache`、以及 OpenAI 兼容的 `mlx_lm.server`。
- 结论：**不改 PageServe，直接用 mlx_lm 自带 server 跑独立服务**。PageServe 保持 PyTorch 栈跑 Qwen，两者互不干扰。

### 8.2 部署

模型：`mlx-community/gemma-4-e4b-it-8bit`（8bit 量化，32GB Mac 官方推荐档；4bit 更省内存 ~6GB）。

一键启动脚本 `inference/run_gemma_mlx.sh`（已固化 hf-mirror + 禁 Xet + 模型/端口）：
```bash
/Users/niupian/lianshan/my_agent/inference/run_gemma_mlx.sh   # 默认 :8082
```
等价手动命令：
```bash
cd inference && source .venv/bin/activate
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  mlx_lm.server --model mlx-community/gemma-4-e4b-it-8bit --host 0.0.0.0 --port 8082
```

### 8.3 调用（OpenAI 兼容 /v1/chat/completions）

**关键**：Gemma 4 默认开 thinking 模式，会先输出思考过程（占满 max_tokens 导致正文为空）。直接要结果时传 `chat_template_kwargs.enable_thinking=false`：

```bash
curl http://localhost:8082/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "mlx-community/gemma-4-e4b-it-8bit",
  "messages": [{"role":"user","content":"请写一首关于重阳节思念亲人的诗"}],
  "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

响应为标准 OpenAI 格式，正文在 `choices[0].message.content`；开 thinking 时思考过程在 `reasoning_content`。

### 8.4 实测（干净环境验证）

| 配置 | 质量 | 速度 | 内存 |
|------|:---:|:---:|:---:|
| MLX 8bit | ✅ 连贯无退化（"丹桂飘香送秋意…"） | 17-24 tok/s（峰值167短输出） | 8.5GB |
| 对比 PyTorch MPS fp16 | ❌ 长文本退化 | 快但不可用 | 15GB |
| 对比 PyTorch MPS fp32 | ✅ | 慢到挂起 | 30GB |

加载 ~5s；前缀缓存生效（响应 `usage.prompt_tokens_details.cached_tokens`）。

### 8.5 端口分工
- **:8082** — MLX / mlx_lm.server / Gemma 4（本方案）
- **:8001** — PageServe / PyTorch / Qwen（原引擎，MPS 支持成熟的模型）
