#!/usr/bin/env python3
"""OpenAI tool-calling regression against llm_server: tools-test.py [port=8081]"""
import json, sys, time, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "8081"
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
tools = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command in the workspace.",
          "parameters": {"type": "object", "properties": {
              "command": {"type": "string", "description": "The command to run"},
              "description": {"type": "string", "description": "Short description of what the command does"},
              "workdir": {"type": "string", "description": "Working directory"}}, "required": ["command"]}}}]
msgs = [{"role": "system", "content": "You are a coding agent working in /workspace/thor-u. Use tools when needed."},
        {"role": "user", "content": "列出当前目录的内容。"}]
def call(body, timeout=600):
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=timeout)
for i in range(100):
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read(); break
    except Exception: time.sleep(3)
# 1) non-stream
out = json.load(call({"model": "nemotron-bolt-74b", "messages": msgs, "tools": tools, "max_tokens": 256, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False}}))
ch = out["choices"][0]; m = ch["message"]
print("non-stream finish:", ch["finish_reason"], "| content:", repr((m.get("content") or "")[:80]), "| tool_calls:", json.dumps(m.get("tool_calls"), ensure_ascii=False)[:300])
# 2) stream
body = {"model": "nemotron-bolt-74b", "messages": msgs, "tools": tools, "max_tokens": 256, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}
content = ""; tcs = None; finish = None
with call(body) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:]); c = d["choices"][0]; delta = c.get("delta", {})
        content += delta.get("content") or ""
        if delta.get("tool_calls"): tcs = delta["tool_calls"]
        if c.get("finish_reason"): finish = c["finish_reason"]
print("stream finish:", finish, "| streamed content:", repr(content[:80]), "| tool_calls:", json.dumps(tcs, ensure_ascii=False)[:300])
# 3) tool result round-trip
if m.get("tool_calls"):
    tc = m["tool_calls"][0]
    msgs2 = msgs + [{"role": "assistant", "content": m.get("content"), "tool_calls": m["tool_calls"]},
                    {"role": "tool", "tool_call_id": tc["id"], "content": "total 3\ndrwxr-xr-x 2 u u 4096 README.md\n-rw-r--r-- 1 u u  120 main.py\n-rw-r--r-- 1 u u   42 notes.txt"}]
    out2 = json.load(call({"model": "nemotron-bolt-74b", "messages": msgs2, "tools": tools, "max_tokens": 200, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False}}))
    c2 = out2["choices"][0]
    print("round2 finish:", c2["finish_reason"], "| content:", repr((c2["message"].get("content") or "")[:200]), "| tool_calls:", c2["message"].get("tool_calls"))
