#!/bin/bash
# Rebuild base+draft engines with KV capacity 131072 from the kept slim base ONNX (export/bolt-long/llm) and the
# fold2 draft ONNX, test long prompts, then restart the server on the new engine.  usage: build-bolt-128k.sh [kv=131072]
cd /zeekr_data/llm; source scripts/env.sh; L=$LLM_HOME; KV=${1:-131072}; OUT=/zeekr_data/llm/export/bolt-long; EB=$BOLT_ENGINES/bolt-spec-128k
log(){ echo "$(date +%H:%M:%S) $*"; }
pool(){ echo "${BOARD_SUDO_PW:-nvidia}" | sudo -S bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; sysctl -w vm.nr_hugepages=$1" >/dev/null 2>&1; grep HugePages_Total /proc/meminfo; }
pkill -x llm_server; sleep 3
[ -f $OUT/llm/model.onnx ] || { log "no base ONNX at $OUT/llm"; exit 1; }
pool 16384
rm -rf $EB; mkdir -p $EB
log "build spec base kv=$KV"
for try in 1 2 3; do timeout 900 llm_build --onnxDir $OUT/llm --engineDir $EB --maxInputLen 2048 --maxKVCacheCapacity $KV --maxBatchSize 2 --maxVerifyTreeSize 16 --specBase > $L/logs/build-spec-base-128k.log 2>&1 && break; log "base attempt $try failed/stalled"; done
[ -e $EB/spec_base.engine ] || { log "BASE BUILD FAILED: $(grep -a '\[ERROR\]' $L/logs/build-spec-base-128k.log | head -1 | cut -c1-160)"; exit 1; }
python3 - <<PY
import json; p="$EB/base_config.json"; c=json.load(open(p)); c["image_token_id"]=18; json.dump(c, open(p,"w"), indent=2)
PY
log "build fold2 draft kv=$KV"
for try in 1 2 3; do timeout 300 llm_build --onnxDir /zeekr_map/llm/export/bolt-lilicorr-draft-fold2/dflash_draft --engineDir $EB --maxInputLen 4096 --maxKVCacheCapacity $KV --maxBatchSize 2 --maxDraftTreeSize 8 --specDraft > $L/logs/build-draft-128k.log 2>&1 && break; log "draft attempt $try failed/stalled"; done
[ -e $EB/spec_draft.engine ] || { log "DRAFT BUILD FAILED: $(grep -a '\[ERROR\]' $L/logs/build-draft-128k.log | head -1 | cut -c1-160)"; exit 1; }
pool 24576
EDGELLM_LILICORR_OFF=1 timeout 400 llm_inference --engineDir $EB --inputFile $L/inputs/code_512.json --outputFile $L/logs/out-128k-code.json --dumpProfile --specDecode --dflashBlockSize 6 > $L/logs/bench-128k-code.log 2>&1
echo "128k engine code_512: $(grep -E 'Overall Tokens/Second|Average Accept' $L/logs/bench-128k-code.log | tr '\n' ' ') $(grep -a '\[ERROR\]' $L/logs/bench-128k-code.log | head -1 | cut -c1-120)"
grep HugePages_Free /proc/meminfo
nohup bash scripts/server-bolt.sh 8081 bolt-spec-128k 6 > /dev/null 2>&1 < /dev/null &
for n in 24000 60000 100000; do echo "== API ~$n"; timeout 900 python3 scripts/long-prompt-test.py 8081 $n; done
grep -a "\[req " $L/logs/server-bolt-8081.log | tail -n 3 | sed "s/.*prompt_tokens/prompt_tokens/" | cut -c1-120
grep HugePages_Free /proc/meminfo
log "BUILD128K DONE"
