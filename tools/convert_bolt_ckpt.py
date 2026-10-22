#!/usr/bin/env python3
"""Convert primitive-ai/Nemotron-3.5-Bolt-74B-A9B-mixed-NVFP4-FP8 into a checkpoint that
TensorRT Edge-LLM's nemotron_h exporter can consume.

Source formats (ModelOpt 0.47 MIXED_PRECISION, vLLM b12x flavoured):
  * routed/shared experts ........ W4A16_NVFP4 g16  (kept verbatim; shards hard-linked)
  * mamba in_proj / out_proj ...... FP8_PER_CHANNEL_PER_TOKEN  -> NVFP4 W4A16 (default) or BF16
  * fc1/fc2_latent_proj, attn qkvo  Q8_0 (ggml blocks of 32, 34 B)  -> BF16 (default) or NVFP4 W4A16
  * lm_head ...................... Q8_0 -> NVFP4 W4A16 (default) or BF16
  * everything else .............. BF16 copied

Output: <dst>/ with model.safetensors.index.json, hf_quant_config.json, config.json (text-only
nemotron_h, llm_config flattened), config.omni.json (original), tokenizer files.
Keys keep the ``language_model.`` prefix (Edge-LLM strips it automatically).

usage: convert_bolt_ckpt.py SRC DST [--mamba nvfp4|bf16] [--latent bf16|nvfp4] [--attn bf16|nvfp4]
       [--lm-head nvfp4|bf16] [--keep-vision] [--layers 0,1,16]   (layer subset for dry runs)
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LM = "language_model."


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- quant helpers
def dequant_q8_0(t: torch.Tensor) -> torch.Tensor:
    """ggml Q8_0 packed as uint8 [N, K/32, 34] -> float32 [N, K]."""
    assert t.dtype == torch.uint8 and t.shape[-1] == 34, t.shape
    blk = t.contiguous()
    scale = blk[..., :2].contiguous().view(torch.float16).float().squeeze(-1)  # [N, K/32]
    q = blk[..., 2:].contiguous().view(torch.int8).float()  # [N, K/32, 32]
    return (q * scale[..., None]).reshape(t.shape[0], -1)


def dequant_fp8_pc(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return w.float() * s.float()[:, None]


_nvfp4 = None


def nvfp4_quantize(w: torch.Tensor, ws2: torch.Tensor):
    """ModelOpt NVFP4 (block 16) with a fixed per-tensor global scale ws2.
    Returns (packed uint8 [N, K/2], block scales float8_e4m3fn [N, K/16])."""
    global _nvfp4
    if _nvfp4 is None:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
        _nvfp4 = NVFP4QTensor
    q, s_blk, _ = _nvfp4.quantize(w.contiguous(), 16, weights_scaling_factor_2=ws2)
    packed = q._quantized_data if hasattr(q, "_quantized_data") else q
    return packed.to(torch.uint8).contiguous(), s_blk.to(torch.float8_e4m3fn).contiguous()


def global_scale(amax: float) -> torch.Tensor:
    return torch.tensor(amax / (6.0 * 448.0), dtype=torch.float32)


def quantize_rows_chunked(get_rows, n_rows: int, chunk: int):
    """get_rows(lo, hi) -> float32 [hi-lo, K]. Two passes: amax, then quantize."""
    amax = 0.0
    for lo in range(0, n_rows, chunk):
        amax = max(amax, float(get_rows(lo, min(lo + chunk, n_rows)).abs().max()))
    ws2 = global_scale(amax)
    ws, ss = [], []
    for lo in range(0, n_rows, chunk):
        w, s = nvfp4_quantize(get_rows(lo, min(lo + chunk, n_rows)), ws2)
        ws.append(w)
        ss.append(s)
    return torch.cat(ws), torch.cat(ss), ws2.reshape(())


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mamba", default="nvfp4", choices=["nvfp4", "bf16"])
    ap.add_argument("--latent", default="bf16", choices=["nvfp4", "bf16"])
    ap.add_argument("--attn", default="bf16", choices=["nvfp4", "bf16"])
    ap.add_argument("--lm-head", default="nvfp4", choices=["nvfp4", "bf16"])
    ap.add_argument("--keep-vision", action="store_true")
    ap.add_argument("--layers", default="", help="comma list of source layer indices to keep (dry runs)")
    args = ap.parse_args()
    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    cfg = json.load(open(f"{src}/config.json"))
    llm = cfg["llm_config"]
    n_layers_src = len(llm["layers_block_type"])
    keep = list(range(n_layers_src)) if not args.layers else [int(x) for x in args.layers.split(",")]
    remap = {old: new for new, old in enumerate(keep)}  # source layer -> output layer
    subset = len(keep) != n_layers_src
    log(f"source layers {n_layers_src}, keeping {len(keep)}{' (subset)' if subset else ''}")

    idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    qcfg = json.load(open(f"{src}/hf_quant_config.json"))["quantization"]["quantized_layers"]
    new_map = {}
    new_q = {}
    layer_re = re.compile(r"^(language_model\.backbone\.layers\.)(\d+)\.(.*)$")

    def remap_key(k):
        m = layer_re.match(k)
        if not m:
            return k
        old = int(m.group(2))
        if old not in remap:
            return None
        return f"{m.group(1)}{remap[old]}.{m.group(3)}"

    def module_of(k):
        return k.rsplit(".", 1)[0]

    def emit(out: dict, key: str, tensor: torch.Tensor, shard: str):
        out[key] = tensor
        new_map[key] = shard

    def mark_nvfp4(mod_key: str):
        new_q[mod_key] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}

    # ---- 1) expert shards: hard-link verbatim when not subsetting
    shards = sorted(set(idx.values()))
    expert_shards = [s for s in shards if s.startswith("experts-nvfp4-") or s == "nvfp4-shared.safetensors"]
    for s in expert_shards:
        keys = [k for k in idx if idx[k] == s]
        if not subset:
            dstp = f"{dst}/{s}"
            if not os.path.exists(dstp):
                os.link(f"{src}/{s}", dstp)
            for k in keys:
                new_map[k] = s
                mod = module_of(k)
                if mod in qcfg and k.endswith(".weight"):
                    new_q[mod] = qcfg[mod]
            log("linked", s, len(keys), "tensors")
        else:
            out = {}
            with safe_open(f"{src}/{s}", "pt") as f:
                for k in keys:
                    nk = remap_key(k)
                    if nk is None:
                        continue
                    emit(out, nk, f.get_tensor(k), s)
                    if k.endswith(".weight"):
                        new_q[module_of(nk)] = qcfg[module_of(k)]
            if out:
                save_file(out, f"{dst}/{s}", metadata={"format": "pt"})
                log("wrote", s, len(out), "tensors (subset)")

    # ---- 2) bf16 base: LM part (+ vision optionally)
    out_lm, out_vis = {}, {}
    with safe_open(f"{src}/bf16-base.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith(LM):
                nk = remap_key(k)
                if nk is None:
                    continue
                emit(out_lm, nk, f.get_tensor(k), "bf16-base.safetensors")
            else:
                if args.keep_vision:
                    emit(out_vis, k, f.get_tensor(k), "vision-bf16.safetensors")
    save_file(out_lm, f"{dst}/bf16-base.safetensors", metadata={"format": "pt"})
    log("wrote bf16-base.safetensors", len(out_lm))
    if out_vis:
        save_file(out_vis, f"{dst}/vision-bf16.safetensors", metadata={"format": "pt"})
        log("wrote vision-bf16.safetensors", len(out_vis))

    # ---- 3) mamba in/out_proj: FP8 per-channel -> NVFP4 W4A16 / BF16
    for part in ("in", "out"):
        s = f"fp8pc-mamba_{part}.safetensors"
        out = {}
        with safe_open(f"{src}/{s}", "pt") as f:
            wkeys = sorted(k for k in f.keys() if k.endswith(".weight"))
            for k in wkeys:
                nk = remap_key(k)
                if nk is None:
                    continue
                mod_src, mod = module_of(k), module_of(nk)
                w = dequant_fp8_pc(f.get_tensor(k), f.get_tensor(mod_src + ".weight_scale"))
                if args.mamba == "nvfp4":
                    q, sb, s2 = quantize_rows_chunked(lambda lo, hi, w=w: w[lo:hi], w.shape[0], 4096)
                    emit(out, mod + ".weight", q, s)
                    emit(out, mod + ".weight_scale", sb, s)
                    emit(out, mod + ".weight_scale_2", s2, s)
                    mark_nvfp4(mod)
                else:
                    emit(out, mod + ".weight", w.to(torch.bfloat16), s)
        save_file(out, f"{dst}/{s}", metadata={"format": "pt"})
        log("wrote", s, len(out), f"tensors ({args.mamba})")

    # ---- 4) Q8_0 groups: latent, attn, lm_head
    def convert_q8_shard(s: str, mode: str, chunk_rows: int):
        out = {}
        with safe_open(f"{src}/{s}", "pt") as f:
            for k in sorted(f.keys()):
                nk = remap_key(k)
                if nk is None:
                    continue
                mod = module_of(nk)
                t = f.get_tensor(k)
                if mode == "bf16":
                    emit(out, nk, dequant_q8_0(t).to(torch.bfloat16), s)
                else:
                    q, sb, s2 = quantize_rows_chunked(lambda lo, hi, t=t: dequant_q8_0(t[lo:hi]), t.shape[0], chunk_rows)
                    emit(out, mod + ".weight", q, s)
                    emit(out, mod + ".weight_scale", sb, s)
                    emit(out, mod + ".weight_scale_2", s2, s)
                    mark_nvfp4(mod)
        save_file(out, f"{dst}/{s}", metadata={"format": "pt"})
        log("wrote", s, len(out), f"tensors ({mode})")

    convert_q8_shard("q8_0-latent.safetensors", args.latent, 4096)
    convert_q8_shard("q8_0-attn.safetensors", args.attn, 4096)
    convert_q8_shard("q8_0-lmhead.safetensors", args.lm_head, 8192)

    # ---- 5) index + quant config
    total = 0
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(f"{dst}/model.safetensors.index.json", "w"), indent=1)
    json.dump({"producer": {"name": "modelopt", "version": "0.47.0"},
               "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                                "exclude_modules": [], "quantized_layers": new_q}},
              open(f"{dst}/hf_quant_config.json", "w"))
    log("quantized_layers:", len(new_q))

    # ---- 6) configs: text-only nemotron_h (flattened llm_config) + original omni config
    text = dict(llm)
    text["layers_block_type"] = [llm["layers_block_type"][i] for i in keep]
    text["block_configs"] = [llm["block_configs"][i] for i in keep]
    text["num_hidden_layers"] = len(keep)
    text["model_type"] = "nemotron_h"
    text["architectures"] = ["NemotronHForCausalLM"]
    text["torch_dtype"] = "bfloat16"
    text.pop("quantization_config", None)
    text.pop("auto_map", None)
    moe_blocks = [b for b in text["block_configs"] if b.get("block_type") == "moe"]
    if moe_blocks:
        text["n_routed_experts"] = max(b["n_routed_experts"] for b in moe_blocks)
        text["moe_intermediate_size"] = max(b["moe_intermediate_size"] for b in moe_blocks)
        text["num_experts_per_tok"] = moe_blocks[0].get("num_experts_per_tok", llm.get("num_experts_per_tok", 9))
    text["_bolt_source"] = "primitive-ai/Nemotron-3.5-Bolt-74B-A9B-mixed-NVFP4-FP8"
    text["_bolt_convert_args"] = vars(args)
    json.dump(text, open(f"{dst}/config.json", "w"), indent=1)
    omni = dict(cfg)
    omni["llm_config"] = dict(llm, layers_block_type=text["layers_block_type"], block_configs=text["block_configs"],
                              num_hidden_layers=len(keep))
    omni.pop("quantization_config", None)
    json.dump(omni, open(f"{dst}/config.omni.json", "w"), indent=1)
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
               "preprocessor_config.json", "processor_config.json"):
        if os.path.exists(f"{src}/{fn}"):
            shutil.copy(f"{src}/{fn}", f"{dst}/{fn}")
    log("done ->", dst)


if __name__ == "__main__":
    main()
