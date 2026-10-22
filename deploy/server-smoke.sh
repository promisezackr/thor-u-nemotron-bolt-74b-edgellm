#!/bin/bash
# server-smoke.sh [port=8081] [engineTag=bolt-spec-long]: (re)start the server, wait for it (python), send one chat request.
cd /zeekr_data/llm; source scripts/env.sh; L=$LLM_HOME; PORT=${1:-8081}; TAG=${2:-bolt-spec-long}
pkill -x llm_server; sleep 3; : > $L/logs/server-bolt-$PORT.log
nohup bash scripts/server-bolt.sh $PORT $TAG 6 > /dev/null 2>&1 < /dev/null &
python3 - "$PORT" <<'PY'
import json, sys, time, urllib.request
port=sys.argv[1]; t0=time.time()
for i in range(200):
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read(); break
    except Exception: time.sleep(3)
print("server up after %.0f s" % (time.time()-t0))
body={"model":"nemotron-bolt-74b","messages":[{"role":"user","content":"用三句话介绍一下杭州。"}],"max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
try:
    t=time.time(); r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    out=json.load(urllib.request.urlopen(r, timeout=600)); dt=time.time()-t; u=out.get("usage",{})
    print("completion_tokens", u.get("completion_tokens"), "wall %.2fs" % dt, "=> %.1f t/s incl. TTFT" % (u.get("completion_tokens",0)/dt))
    print("TEXT:", out["choices"][0]["message"]["content"][:300].replace("\n"," | "))
except Exception as e: print("REQUEST FAILED:", e)
PY
grep -a -E "ERROR|Failed|failed" $L/logs/server-bolt-$PORT.log | head -3 | cut -c1-200
echo SMOKE-DONE
