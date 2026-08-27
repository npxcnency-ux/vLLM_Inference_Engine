"""
用 27B (Qwen, 标准 KVCache 可量化) 验证核心问题：
  量化后的 KV cache 还能不能被 prompt cache 复用？

短前缀(~2000 token)，只测"复用机制"是否成立，不追求还原 315s。
turn2 只追加少量 token；复用成功则 turn2 << turn1。
"""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache
from mlx_lm.generate import generate_step, maybe_quantize_kv_cache

MODEL = "mlx-community/Qwen3.8-27B-4bit"
print(f"loading {MODEL} ...")
model, tokenizer = load(MODEL)

prefix_text = "请记住以下资料。" + "第这是一段用于撑长上下文的填充内容。".replace("这", "") * 300
new_text = "\n\n请问上面资料大概多长？"
prefix_ids = mx.array(tokenizer.encode(prefix_text))
new_ids = mx.array(tokenizer.encode(new_text))
print(f"前缀 token 数 = {prefix_ids.size}, 新增 token 数 = {new_ids.size}")


def prefill(cache, tokens, kv_bits=None):
    t0 = time.perf_counter()
    for _ in generate_step(tokens, model, max_tokens=1, prompt_cache=cache,
                           kv_bits=kv_bits, quantized_kv_start=0):
        break
    return time.perf_counter() - t0


def _off(c):
    return getattr(c, "offset", "NA")


def run_case(name, quantize_after_turn1=False, kv_bits_all=None):
    print(f"\n=== {name} ===")
    cache = make_prompt_cache(model)
    from collections import Counter
    types = Counter(type(c).__name__ for c in cache)
    print(f"  cache 层类型分布: {dict(types)}")
    t1 = prefill(cache, prefix_ids, kv_bits=kv_bits_all)
    print(f"  turn1 冷 prefill 前缀: {t1*1000:8.1f} ms  (offset={_off(cache[0])})")

    if quantize_after_turn1:
        try:
            maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_group_size=64, kv_bits=8)
            print(f"  → 量化后 cache类型={type(cache[0]).__name__}, offset={cache[0].offset}, trimmable={can_trim_prompt_cache(cache)}")
        except NotImplementedError as e:
            print(f"  → 量化失败: {e}")
            return

    kb2 = kv_bits_all if kv_bits_all else (8 if quantize_after_turn1 else None)
    t2 = prefill(cache, new_ids, kv_bits=kb2)
    print(f"  turn2 复用+只处理新增: {t2*1000:8.1f} ms  (offset={_off(cache[0])})")
    print(f"  >>> turn2/turn1 = {t2/t1:.3f}  ({'复用成功✅' if t2 < t1*0.5 else '复用失效❌'})")


run_case("A. 普通 KVCache 复用（对照）")
run_case("B. turn1 后量化再复用")
run_case("C. 全程 kv_bits=8 再复用（你 27B 那次的模式）")
