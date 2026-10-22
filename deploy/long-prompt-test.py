#!/usr/bin/env python3
"""Long-prompt (chunked prefill) test against the running server: long-prompt-test.py [port=8081] [approx_tokens=6000]
Plants a code word at the start and the end of a long filler prompt and asks the model to repeat both."""
import json, sys, time, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "8081"
target = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
filler = ("The quick brown fox jumps over the lazy dog while the river flows quietly past the old mill, "
          "and the baker counts loaves before dawn as trains rumble across the iron bridge. ")
# ~30 tokens per sentence
body_text = filler * max(1, target // 30)
prompt = (f"Secret word A is PINEAPPLE-7731.\n\n{body_text}\n\nSecret word B is OCTOPUS-2048.\n\n"
          "Question: what are secret word A and secret word B? Answer with just the two words.")
for i in range(100):
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read(); break
    except Exception: time.sleep(3)
body = {"model": "nemotron-bolt-74b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
t = time.time()
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json"})
try:
    out = json.load(urllib.request.urlopen(r, timeout=900)); dt = time.time() - t; u = out.get("usage", {})
    print(f"prompt_tokens={u.get('prompt_tokens')} completion={u.get('completion_tokens')} wall={dt:.1f}s")
    print("ANSWER:", out["choices"][0]["message"]["content"][:200].replace("\n", " | "))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:400])
except Exception as e:
    print("ERR", e)
