#!/bin/bash
# OpenAI-compatible server for Bolt 74B with DFlash speculative decoding (no curl on the board: use python).
# usage: server-bolt.sh [port=8081] [engineTag=bolt-spec-128k] [block=6] [think|nothink (default think)]
source /zeekr_data/llm/scripts/env.sh; L=$LLM_HOME
PORT=${1:-8081}; TAG=${2:-bolt-spec-128k}; BLK=${3:-6}; THINK=${4:-think}
THINKFLAG=""; [ "$THINK" = think ] && THINKFLAG="--defaultThinking"
grep -q "HugePages_Total: *24576" /proc/meminfo || echo "${BOARD_SUDO_PW:-nvidia}" | sudo -S bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; sysctl -w vm.nr_hugepages=24576" >/dev/null 2>&1
ip -4 addr show | grep -q "10.0.0.3/" || echo "${BOARD_SUDO_PW:-nvidia}" | sudo -S ip addr add 10.0.0.3/24 dev $(ip -o -4 addr show | grep "10.0.0.2/" | awk '{print $2}' | head -1) 2>/dev/null
mkdir -p $L/logs
echo "starting llm_server on :$PORT engine=$TAG block=$BLK thinking=$THINK (log: $L/logs/server-bolt-$PORT.log)"
exec env EDGELLM_LILICORR_OFF=1 llm_server --engineDir $BOLT_ENGINES/$TAG --host 0.0.0.0 --port $PORT --model nemotron-bolt-74b --specDecode --specDraftTopK 1 --specDraftStep 1 --dflashBlockSize $BLK --maxTokens ${MAX_TOKENS:-8192} --maxTokensCap ${MAX_TOKENS_CAP:-32768} $THINKFLAG >> $L/logs/server-bolt-$PORT.log 2>&1
