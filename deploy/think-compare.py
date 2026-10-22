#!/usr/bin/env python3
"""Compare thinking off/on (and medium_effort) on one reasoning question: think-compare.py [port=8081]"""
import json, sys, time, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "8081"
q = ("一个水池有进水管和出水管。只开进水管6小时注满，只开出水管9小时放空。现在池子是空的，两管同时开，多少小时注满？"
     "只给出最终答案和一句话理由。")
for label, kw in (("off", {"enable_thinking": False}), ("on", {"enable_thinking": True}),
                  ("on+medium_effort", {"enable_thinking": True, "medium_effort": True})):
    body = {"model": "nemotron-bolt-74b", "messages": [{"role": "user", "content": q}], "max_tokens": 4096,
            "temperature": 0, "chat_template_kwargs": kw}
    t = time.time()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(r, timeout=900)); dt = time.time() - t
    m = out["choices"][0]["message"]; u = out["usage"]
    rc = m.get("reasoning_content") or ""
    print(f"thinking={label}: {u.get('completion_tokens')} tok in {dt:.1f}s | reasoning chars={len(rc)} | answer: {(m.get('content') or '')[:200]!r}")
