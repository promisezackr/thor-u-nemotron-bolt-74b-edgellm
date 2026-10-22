#!/bin/bash
# Create a spec engine dir that shares the built base engine (hard links) and build a draft variant into it.
# usage: mk-spec-variant.sh <newTag> <draftOnnxDir> [maxDraftTreeSize=8] [baseEngineTag=bolt-spec-in2048-kv8192-bs2]
source /zeekr_data/llm/scripts/env.sh
SRC=$BOLT_ENGINES/${4:-bolt-spec-in2048-kv8192-bs2}; DST=$BOLT_ENGINES/$1; DRAFT=$2; TREE=${3:-8}
mkdir -p $DST
for f in spec_base.engine base_config.json external_nvfp4_moe_weights.safetensors embedding.safetensors tokenizer.json tokenizer_config.json chat_template.jinja; do
  [ -e $DST/$f ] || ln $SRC/$f $DST/$f
done
llm_build --onnxDir $DRAFT --engineDir $DST --maxInputLen 2048 --maxKVCacheCapacity 8192 --maxBatchSize 2 --maxDraftTreeSize $TREE --specDraft > $LLM_HOME/logs/build-draft-$1.log 2>&1 && echo "DRAFT BUILT $DST" || { grep "\[ERROR\]" $LLM_HOME/logs/build-draft-$1.log | head -3; echo "DRAFT BUILD FAILED"; }
