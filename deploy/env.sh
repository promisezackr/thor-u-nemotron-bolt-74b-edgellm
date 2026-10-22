#!/bin/bash
# Edge-LLM runtime environment for Nemotron 3.5 Bolt on Thor (source this).
# Layout: /zeekr_data/llm/{edgellm/bin,models,scripts,logs,pyenv,edgellm-src}; big outputs on /zeekr_map/llm.
export LLM_HOME=/zeekr_data/llm
export EDGELLM_BIN=$LLM_HOME/edgellm/bin
export EDGELLM_PLUGIN_PATH=$EDGELLM_BIN/libNvInfer_edgellm_plugin.so
export PATH=$EDGELLM_BIN:$LLM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$EDGELLM_BIN:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_CACHE_PATH=$LLM_HOME/.nv_cache
export CUDA_CACHE_MAXSIZE=4294967296
export PYTHONPATH=$LLM_HOME/pyenv/site-packages:$LLM_HOME/edgellm-src
export BOLT_CKPT=$LLM_HOME/models/bolt-edgellm-ckpt
export BOLT_EXPORT=/zeekr_map/llm/export/bolt
export BOLT_ENGINES=/zeekr_data/llm/engines   # same filesystem as the external MoE weights (hard-link, no copy)
