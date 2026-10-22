#!/bin/bash
# sweep-fold.sh <draftOnnxDir> <tag> [blocks="2 3 4 5 6 8"] [input=code_512]: build a (no-head) draft with
# --maxDraftTreeSize 8 on base=bolt-spec-out into engines/bolt-spec-<tag>, then bench each block size.
source /zeekr_data/llm/scripts/env.sh; L=$LLM_HOME; ONNX=$1; TAG=$2; BLOCKS=${3:-"2 3 4 5 6 8"}; INP=${4:-code_512}
E=$BOLT_ENGINES/bolt-spec-$TAG
pool(){ echo "${BOARD_SUDO_PW:-nvidia}" | sudo -S bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; sysctl -w vm.nr_hugepages=$1" >/dev/null 2>&1; }
if [ ! -e $E/spec_draft.engine ]; then
  rm -rf $E; mkdir -p $E
  for f in spec_base.engine base_config.json external_nvfp4_moe_weights.safetensors embedding.safetensors tokenizer.json tokenizer_config.json chat_template.jinja; do ln $BOLT_ENGINES/bolt-spec-out/$f $E/$f; done
  pool 16384
  for try in 1 2 3; do timeout 240 llm_build --onnxDir $ONNX --engineDir $E --maxInputLen 2048 --maxKVCacheCapacity 8192 --maxBatchSize 2 --maxDraftTreeSize 8 --specDraft > $L/logs/build-draft-$TAG.log 2>&1 && break; echo "build attempt $try failed/stalled"; done
  [ -e $E/spec_draft.engine ] || { echo "$TAG BUILD FAILED: $(grep -a '\[ERROR\]' $L/logs/build-draft-$TAG.log | head -1 | cut -c1-160)"; exit 1; }
fi
pool 24576
for b in $BLOCKS; do
  EDGELLM_LILICORR_OFF=1 timeout 300 llm_inference --engineDir $E --inputFile $L/inputs/$INP.json --outputFile $L/logs/out-$TAG-$INP-b$b.json --dumpProfile --specDecode --dflashBlockSize $b > $L/logs/bench-$TAG-$INP-b$b.log 2>&1
  echo "$TAG $INP b$b: $(grep -E 'Overall Tokens/Second' $L/logs/bench-$TAG-$INP-b$b.log | awk -F: '{print $2}' | tr -d ' ') t/s accept=$(grep 'Average Accept Length' $L/logs/bench-$TAG-$INP-b$b.log | awk -F: '{print $2}' | tr -d ' ') draft_ms=$(grep 'Construct Draft Proposal' $L/logs/bench-$TAG-$INP-b$b.log | sed 's/.*Average: //') verify_ms=$(grep 'Base Model Verification' $L/logs/bench-$TAG-$INP-b$b.log | sed 's/.*Average: //') $(grep -a '\[ERROR\]' $L/logs/bench-$TAG-$INP-b$b.log | head -1 | cut -c1-100)"
done
echo "SWEEP $TAG DONE"
