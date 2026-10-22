#!/bin/bash
# Current best speculative config: base bolt-spec-out (NVFP4 experts/mamba, BF16 latent/attn) + fold2 no-head
# LiLiCorr draft, DFlash block 6 (57 t/s on code_512, accept ~4.4).  usage: run-bolt-best.sh <input.json> [block=6]
source /zeekr_data/llm/scripts/env.sh
INP=$1; BLK=${2:-6}; L=$LLM_HOME
grep -q "HugePages_Total: *24576" /proc/meminfo || echo "${BOARD_SUDO_PW:-nvidia}" | sudo -S bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; sysctl -w vm.nr_hugepages=24576" >/dev/null 2>&1
EDGELLM_LILICORR_OFF=1 llm_inference --engineDir $BOLT_ENGINES/bolt-spec-fold2 --inputFile $INP --outputFile $L/logs/out-best-$(basename $INP .json).json --dumpProfile --specDecode --dflashBlockSize $BLK
