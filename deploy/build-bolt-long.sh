#!/bin/bash
# Long-context production engines: base (layer-output DFlash) + fold2 draft, maxInputLen 8192 / KV 32768.
# usage: build-bolt-long.sh [ckptDir=/zeekr_map/llm/models/bolt-edgellm-ckpt]
source /zeekr_data/llm/scripts/env.sh; L=$LLM_HOME
CK=${1:-/zeekr_map/llm/models/bolt-edgellm-ckpt}; OUT=/zeekr_data/llm/export/bolt-long; EB=$BOLT_ENGINES/bolt-spec-long
log(){ echo "$(date +%H:%M:%S) $*"; }
pool(){ echo "${BOARD_SUDO_PW:-nvidia}" | sudo -S bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; sysctl -w vm.nr_hugepages=$1" >/dev/null 2>&1; grep HugePages_Total /proc/meminfo; }
python3 - <<PY
import json; p="$CK/model.safetensors.index.json"; d=json.load(open(p)); d["weight_map"]={k:v for k,v in d["weight_map"].items() if v!="vision-bf16.safetensors"}; json.dump(d, open(p,"w"))
PY
rm -rf $OUT; mkdir -p $OUT; df -h /zeekr_data | tail -1
pool 1100
log "export base (out convention)"
nice -n 5 python3 -m tensorrt_edgellm.scripts.export $CK $OUT --target-sm 110 --dflash-base --dflash-draft-dir $L/models/lilicorr > $L/logs/export-long.log 2>&1 || { log "EXPORT FAILED"; exit 1; }
log "reexternalize"; python3 $L/scripts/reexternalize.py $OUT/llm $L/ext > $L/logs/reexternalize-long.log 2>&1 || { log "REEXT FAILED"; exit 1; }
df -h /zeekr_data | tail -1
pool 16384
log "build spec base 8k/32k"
rm -rf $EB; mkdir -p $EB
for try in 1 2 3; do timeout 900 llm_build --onnxDir $OUT/llm --engineDir $EB --maxInputLen 8192 --maxKVCacheCapacity 32768 --maxBatchSize 2 --maxVerifyTreeSize 16 --specBase > $L/logs/build-spec-base-long.log 2>&1 && break; log "base build attempt $try failed/stalled"; done
[ -e $EB/spec_base.engine ] || { log "BASE BUILD FAILED: $(grep -a '\[ERROR\]' $L/logs/build-spec-base-long.log | head -1 | cut -c1-160)"; exit 1; }
python3 - <<PY
import json; p="$EB/base_config.json"; c=json.load(open(p)); c["image_token_id"]=18; json.dump(c, open(p,"w"), indent=2)
PY
log "build fold2 draft 8k/32k"
for try in 1 2 3; do timeout 300 llm_build --onnxDir /zeekr_map/llm/export/bolt-lilicorr-draft-fold2/dflash_draft --engineDir $EB --maxInputLen 8192 --maxKVCacheCapacity 32768 --maxBatchSize 2 --maxDraftTreeSize 8 --specDraft > $L/logs/build-draft-long.log 2>&1 && break; log "draft build attempt $try failed/stalled"; done
[ -e $EB/spec_draft.engine ] || { log "DRAFT BUILD FAILED: $(grep -a '\[ERROR\]' $L/logs/build-draft-long.log | head -1 | cut -c1-160)"; exit 1; }
pool 24576
for inp in code_512 in_zh_512; do EDGELLM_LILICORR_OFF=1 timeout 400 llm_inference --engineDir $EB --inputFile $L/inputs/$inp.json --outputFile $L/logs/out-long-$inp.json --dumpProfile --specDecode --dflashBlockSize 6 > $L/logs/bench-long-$inp.log 2>&1; echo "long b6 $inp: $(grep -E 'Overall Tokens/Second' $L/logs/bench-long-$inp.log | awk -F: '{print $2}' | tr -d ' ') t/s accept=$(grep 'Average Accept Length' $L/logs/bench-long-$inp.log | awk -F: '{print $2}' | tr -d ' ') $(grep -a '\[ERROR\]' $L/logs/bench-long-$inp.log | head -1 | cut -c1-120)"; done
rm -rf $OUT/llm/model.onnx.data; log "LONG DONE"
