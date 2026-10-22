#!/usr/bin/env python3
"""API benchmark against the running llm_server: api-bench.py [port=8081] [names...]"""
import json, sys, time, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "8081"
names = sys.argv[2:] or ["code_512", "in_en_512", "in_zh_512"]
L = "/zeekr_data/llm/inputs"
for name in names:
    inp = json.load(open(f"{L}/{name}.json")); req = inp["requests"][0]
    body = {"model": "nemotron-bolt-74b", "messages": req["messages"], "max_tokens": 512, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(r, timeout=600)); dt = time.time() - t; u = out.get("usage", {})
    c = u.get("completion_tokens", 0)
    print(f"{name}: prompt={u.get('prompt_tokens')} completion={c} wall={dt:.2f}s api_tps={c/dt:.1f}")
