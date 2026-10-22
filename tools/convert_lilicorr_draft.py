#!/usr/bin/env python3
"""Adapt nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr for TensorRT Edge-LLM's DFlash V1 draft path.

* ``fc`` (6 target layers x 4096 -> 4096) kept W4A16_NVFP4 (Nemotron base => dflash_fc_native_precision)
* ``lilicorr.*`` correlation head dropped (not used by the top-1 DFlash runtime; kept for a later stage)
* config: dflash_config.is_causal=false (block-bidirectional), everything else kept
* hf_quant_config: exclude ``fc`` (and the already-excluded attention/conv tensors)

usage: convert_lilicorr_draft.py SRC DST [--keep-lilicorr]
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def nvfp4_dequant(packed: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
    """ModelOpt NVFP4: uint8 [N, K/2] (two e2m1 nibbles), e4m3 block scales [N, K/16], fp32 global."""
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(packed.shape[0], -1)  # low nibble first
    s = scale.to(torch.float32) * scale2.to(torch.float32)
    return (vals.reshape(packed.shape[0], -1, 16) * s[..., None]).reshape(packed.shape[0], -1)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    keep_lili = "--keep-lilicorr" in sys.argv
    no_sink = "--no-sink" in sys.argv      # debugging variants: drop learned attention sinks
    no_swa = "--no-swa" in sys.argv        # debugging variants: full attention instead of SWA 1024
    no_conv = "--no-conv" in sys.argv      # debugging variants: drop DFlash2 grouped conv (plain DFlash layers)
    fc_fp16 = "--fc-fp16" in sys.argv      # debugging variants: dequantize fc to FP16 (dense FP32 projection path)
    mlp_fp16 = "--mlp-fp16" in sys.argv    # debugging variants: dequantize gate/up/down to FP16
    down_fp16 = "--down-fp16" in sys.argv  # debugging variants: dequantize only down_proj to FP16
    fold = "--fold-scales" in sys.argv     # production: fold power-of-2 scales so fp16 branch outputs cannot overflow
    def _exp(flag, default):  # --fold-up N etc.: power-of-two exponent of the fold (weights x 2^-N)
        return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
    FOLD_UP, FOLD_DOWN = 2.0 ** -_exp("--fold-up", 4), 2.0 ** -_exp("--fold-down", 6)
    FOLD_O, FOLD_FC = 2.0 ** -_exp("--fold-o", 4), 2.0 ** -_exp("--fold-fc", 8)
    # q/k folds are free (q_norm/k_norm are scale-invariant); a v fold is linear through attention and is
    # undone together with the o_proj fold (attn unfold = 1 / (FOLD_V * FOLD_O)).
    FOLD_QK, FOLD_V = 2.0 ** -_exp("--fold-qk", 0), 2.0 ** -_exp("--fold-v", 0)
    fc_scale = 1.0                         # --fc-scale X: multiply fc weights by X (hidden_norm makes the draft scale-invariant)
    if "--fc-scale" in sys.argv:
        fc_scale = float(sys.argv[sys.argv.index("--fc-scale") + 1])
    os.makedirs(dst, exist_ok=True)
    out = {}
    with safe_open(f"{src}/model.safetensors", "pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k.startswith("lilicorr."):
                if not keep_lili:
                    continue
                # head linears are W4A16 NVFP4 in the checkpoint; the head runs in FP32 in-graph -> store FP16
                if k.endswith(".weight") and (k[:-len(".weight")] + ".weight_scale") in keys:
                    mod = k[:-len(".weight")]
                    out[k] = nvfp4_dequant(f.get_tensor(k), f.get_tensor(mod + ".weight_scale"),
                                           f.get_tensor(mod + ".weight_scale_2")).to(torch.float16).contiguous()
                    continue
                if k.endswith(".weight_scale") or k.endswith(".weight_scale_2"):
                    continue
                out[k] = f.get_tensor(k)
                continue
            if no_sink and k.endswith("attention_sink_bias"):
                continue
            if no_conv and (".attention_conv." in k or ".mlp_conv." in k):
                continue
            if fc_fp16 and k.startswith("fc."):
                continue
            if (mlp_fp16 and ".mlp." in k) or (down_fp16 and ".mlp.down_proj." in k):
                if k.endswith(".weight"):
                    mod = k[:-len(".weight")]
                    w = nvfp4_dequant(f.get_tensor(k), f.get_tensor(mod + ".weight_scale"), f.get_tensor(mod + ".weight_scale_2"))
                    out[k] = w.to(torch.float16).contiguous()
                    print("dequant", k, tuple(out[k].shape), "std %.4g" % w.std().item())
                continue
            t = f.get_tensor(k)
            if fold:
                if k.endswith(".mlp.up_proj.weight_scale_2"):
                    t = (t.float() * FOLD_UP).to(torch.float32)
                elif k.endswith(".mlp.down_proj.weight_scale_2"):
                    t = (t.float() * FOLD_DOWN).to(torch.float32)
                elif k.endswith(".self_attn.o_proj.weight"):
                    t = (t.float() * FOLD_O).to(t.dtype)
                elif k.endswith(".self_attn.q_proj.weight_scale_2") or k.endswith(".self_attn.k_proj.weight_scale_2"):
                    t = (t.float() * FOLD_QK).to(torch.float32)
                elif k.endswith(".self_attn.v_proj.weight_scale_2"):
                    t = (t.float() * FOLD_V).to(torch.float32)
                elif k == "fc.weight_scale_2":
                    t = (t.float() * FOLD_FC).to(torch.float32)
            out[k] = t  # fc stays W4A16_NVFP4 unless --fc-fp16
        if fc_fp16:
            w = nvfp4_dequant(f.get_tensor("fc.weight"), f.get_tensor("fc.weight_scale"), f.get_tensor("fc.weight_scale_2"))
            out["fc.weight"] = (w * fc_scale).to(torch.float16).contiguous()
            print("fc dequant std", w.std().item(), "absmax", w.abs().max().item(), "scale", fc_scale)
        elif fc_scale != 1.0:
            out["fc.weight_scale_2"] = (out["fc.weight_scale_2"].float() * fc_scale).to(torch.float32)
            print("fc global scale x", fc_scale, "->", out["fc.weight_scale_2"].item())
    if keep_lili:
        for k in list(out.keys()):
            if ".attn.in_proj_weight" in k or ".attn.in_proj_bias" in k:
                out[k.replace("in_proj_weight", "in_proj.weight").replace("in_proj_bias", "in_proj.bias")] = out.pop(k)
    save_file(out, f"{dst}/model.safetensors", metadata={"format": "pt"})
    print("tensors", len(out))

    cfg = json.load(open(f"{src}/config.json"))
    cfg.setdefault("dflash_config", {})["is_causal"] = bool(cfg["dflash_config"].get("causal", False))
    if no_sink:
        cfg["attention_sink_bias"] = False
        cfg["dflash_config"]["attention_sink_bias"] = False
    if no_swa:
        cfg["use_sliding_window"] = False
        cfg["sliding_window"] = None
        cfg["layer_types"] = ["full_attention"] * len(cfg.get("layer_types", [])) or cfg.get("layer_types")
        cfg["dflash_config"]["use_swa"] = False
    if no_conv:
        cfg["dflash_config"]["no_conv"] = True
    if fold:
        # The model multiplies the (fp16) branch outputs back by these factors in fp32.
        cfg["dflash_config"]["lilicorr_scale_fold"] = {"mlp": 1.0 / (FOLD_UP * FOLD_DOWN), "attn": 1.0 / (FOLD_V * FOLD_O),
                                                       "fc": 1.0 / FOLD_FC}
    cfg["_lilicorr_adapted"] = True
    json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)

    q = json.load(open(f"{src}/hf_quant_config.json"))
    # The drafter has no lm_head of its own: Edge-LLM inherits the base model's (NVFP4 W4A16)
    # lm_head, so it must not be excluded from quantization here.
    excl = [e for e in q["quantization"].get("exclude_modules", []) if "lm_head" not in e]
    pats = ["lilicorr*"] + (["fc"] if fc_fp16 else []) + (["layers.*.mlp*"] if mlp_fp16 else []) + (["layers.*.mlp.down_proj"] if down_fp16 else [])
    for pat in pats:
        if pat not in excl:
            excl.append(pat)
    q["quantization"]["exclude_modules"] = excl
    json.dump(q, open(f"{dst}/hf_quant_config.json", "w"), indent=2)
    for fn in ("README.md", "LICENSE"):
        if os.path.exists(f"{src}/{fn}"):
            shutil.copy(f"{src}/{fn}", f"{dst}/{fn}")
    print("done ->", dst)


if __name__ == "__main__":
    main()
