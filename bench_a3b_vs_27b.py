"""
对比 Qwen3-30B-A3B (MoE, :8084) vs Qwen3.8-27B (稠密, :8083)
指标：TTFT(首 token 延迟) + decode 速度(tok/s)。同一个 pi 式大 prompt。
"""
import time, json, urllib.request, sys

TARGETS = [
    ("A3B-MoE :8084", "http://localhost:8084/v1/chat/completions", "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"),
    ("27B稠密 :8083", "http://localhost:8083/v1/chat/completions", "mlx-community/Qwen3.8-27B-4bit"),
]

# pi 式大 system prompt（~4k token 量级，模拟真实系统提示）
big_system = ("You are an expert coding assistant operating inside pi, a coding agent harness. "
    "You help users by reading files, executing commands, editing code, and writing new files. "
    "Available tools: read, bash, edit, write, subagent, submit_plan, update_task, add_task, plan_status. "
    "Guidelines: be concise; use bash for file ops; use read to examine files; use edit for precise changes; "
    "keep oldText small but unique; use write only for new files. ") * 55
big_system += "\n可用 skills：\n" + "\n".join(
    f"- skill_{i}: 处理第 {i} 类任务的专用说明，触发条件是用户提到相关关键词时读取对应 SKILL.md 并遵循分步指令。"
    for i in range(30))

Q = "写一个 Python 函数：判断一个字符串是否是有效的括号匹配（支持 ()[]{}），并给出 3 个测试用例。"

def bench(label, url, model, enable_thinking=False, max_tokens=300):
    body = {"model": model,
            "messages": [{"role":"system","content":big_system},{"role":"user","content":Q}],
            "max_tokens": max_tokens, "stream": True, "temperature": 0.0}
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    t0 = time.perf_counter(); ttft=None; n_tok=0; t_first=None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"): continue
                p = line[5:].strip()
                if p == "[DONE]": break
                try: obj = json.loads(p)
                except: continue
                d = obj.get("choices",[{}])[0].get("delta",{})
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter()-t0; t_first = time.perf_counter()
                    n_tok += 1
    except Exception as e:
        print(f"[{label}] 请求失败: {e}"); return None
    total = time.perf_counter()-t0
    decode_s = total - (ttft or 0)
    tps = (n_tok-1)/decode_s if decode_s>0 and n_tok>1 else 0
    print(f"[{label}]  TTFT={ttft:6.1f}s   decode={tps:5.1f} tok/s   输出={n_tok}个chunk   总耗时={total:.1f}s")
    return {"ttft":ttft,"tps":tps,"n":n_tok,"total":total}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv)>1 else "all"
    print(f"prompt: system~{len(big_system)}字符 + 1问题, max_tokens=300, 关思考\n")
    for label,url,model in TARGETS:
        if which!="all" and which not in label: continue
        # 跑两次：第1次冷 prefill，第2次测前缀缓存命中
        print(f"=== {label} ===")
        bench(label+" turn1(冷)", url, model)
        bench(label+" turn2(缓存命中)", url, model)
        print()
