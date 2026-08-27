"""
验证：量化 KV cache 会不会破坏 prompt cache 的跨轮复用？

方法：用小模型模拟"多轮同前缀"场景。
  turn1: prefill 一段长前缀 P，把 KV cache 留下来。
  turn2: 前缀不变、只追加少量新 token，复用 turn1 的 cache。
        如果复用成功 -> turn2 只需处理"新增 token"，很快。
        如果复用失效 -> turn2 从头重算整个 P，和 turn1 一样慢。

对照三组：
  A. 普通 KVCache 复用
  B. turn1 后把 cache 量化(to_quantized)，再复用       <- 模拟 quantized-kv-start 命中后
  C. 从头就用 kv_bits=8 生成并持有 cache，再复用

只要 B/C 的 turn2 明显快于 turn1，就证明"量化 KV 破坏复用"不成立。
"""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache
from mlx_lm.generate import generate_step, maybe_quantize_kv_cache

MODEL = "mlx-community/gemma-4-e4b-it-8bit"
print(f"loading {MODEL} ...")
model, tokenizer = load(MODEL)

# 一段可复用的"长前缀"（重复放大，制造有意义的 prefill 成本）
prefix_text = ("请阅读以下背景资料并记住。" + "这是第{}段无关紧要的填充内容，用于把上下文撑长以便观测 prefill 成本。".format(0)) * 400
new_text = "\n\n现在请回答：背景资料一共有几段？"

prefix_ids = mx.array(tokenizer.encode(prefix_text))
new_ids = mx.array(tokenizer.encode(new_text))
print(f"前缀 token 数 = {prefix_ids.size}, 新增 token 数 = {new_ids.size}")


def prefill(cache, tokens, kv_bits=None):
    """把 tokens 灌进 cache（只 prefill，不生成），返回耗时秒。"""
    t0 = time.perf_counter()
    # 用 generate_step 跑 1 步即可完成对 tokens 的 prefill（它内部分块处理 prompt）
    gen = generate_step(tokens, model, max_tokens=1, prompt_cache=cache,
                         kv_bits=kv_bits, quantized_kv_start=0)
    for _ in gen:
        break
    mx.eval([c.state for c in cache] if hasattr(cache[0], "state") else [])
    return time.perf_counter() - t0


def run_case(name, quantize_after_turn1=False, kv_bits_all=None):
    print(f"\n=== {name} ===")
    cache = make_prompt_cache(model)

    # turn1: 冷 prefill 整个前缀
    t1 = prefill(cache, prefix_ids, kv_bits=kv_bits_all)
    off1 = cache[0].offset
    typ1 = type(cache[0]).__name__
    print(f"  turn1 冷 prefill 前缀: {t1*1000:7.1f} ms  (cache类型={typ1}, offset={off1})")

    # 可选：turn1 之后把 cache 量化（模拟 quantized-kv-start 命中）
    if quantize_after_turn1:
        maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_group_size=64, kv_bits=8)
        print(f"  → 量化后 cache类型={type(cache[0]).__name__}, offset={cache[0].offset}, "
              f"trimmable={can_trim_prompt_cache(cache)}")

    # turn2: 前缀已在 cache 里，只灌"新增 token"，复用
    t2 = prefill(cache, new_ids, kv_bits=kv_bits_all if kv_bits_all else (8 if quantize_after_turn1 else None))
    off2 = cache[0].offset
    print(f"  turn2 复用+只处理新增: {t2*1000:7.1f} ms  (offset={off2})")
    print(f"  >>> turn2/turn1 = {t2/t1:.3f}  ({'复用成功✅ 快' if t2 < t1*0.5 else '复用失效❌ 没快'})")
    return t1, t2


run_case("A. 普通 KVCache 复用（对照，等价 Gemma 那次）")
run_case("B. turn1 后量化再复用（模拟 quantized-kv-start 命中）", quantize_after_turn1=True)
run_case("C. 全程 kv_bits=8 再复用（你 27B 那次的模式）", kv_bits_all=8)
