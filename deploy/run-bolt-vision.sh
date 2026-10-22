#!/bin/bash
# Run an image+text request: text engine <engineTag> + vision engine engines/bolt-vision.
# usage: run-bolt-vision.sh <engineTag> <input.json> [extra llm_inference args...]
source /zeekr_data/llm/scripts/env.sh
TAG=$1; IN=$2; shift 2
OUT=$LLM_HOME/logs/out-vision-$(basename $IN .json)-$(date +%H%M%S).json
exec llm_inference --engineDir $BOLT_ENGINES/$TAG --multimodalEngineDir $BOLT_ENGINES/bolt-vision --inputFile $IN --outputFile $OUT --dumpOutput --dumpProfile "$@"
