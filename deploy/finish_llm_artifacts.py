#!/usr/bin/env python3
"""Write the runtime sidecars (config.json, embedding.safetensors, tokenizer, chat template) for an
Edge-LLM LLM export whose exporter died after model.onnx was written, and re-patch the external
weight manifest from the ONNX graph inputs.

usage: finish_llm_artifacts.py <ckpt_dir> <onnx_llm_dir>
"""
import os
import sys
import time

import onnx


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ckpt, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    from tensorrt_edgellm.checkpoint.checkpoint_utils import write_runtime_artifacts
    from tensorrt_edgellm.external_weights import patch_external_weight_manifest
    import tensorrt_edgellm._export_api  # noqa: F401  (registers model classes, as the exporter does)
    from tensorrt_edgellm.model import AutoModel

    log("loading model (weights on CPU) from", ckpt)
    model = AutoModel.from_pretrained(ckpt, device="cpu")
    log("writing runtime artifacts ->", out_dir)
    write_runtime_artifacts(model, ckpt, out_dir)
    del model

    onnx_path = os.path.join(out_dir, "model.onnx")
    m = onnx.load(onnx_path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    ext_names = [i.name for i in m.graph.input if i.name.startswith("_model.") and i.name not in inits]
    ext_file = "external_nvfp4_moe_weights.safetensors"
    assert ext_names and os.path.exists(os.path.join(out_dir, ext_file)), (len(ext_names), ext_file)
    patch_external_weight_manifest(out_dir, [{"file": ext_file, "kind": "nvfp4_moe_weights", "tensors": ext_names}])
    log("manifest:", len(ext_names), "external tensors; done")


if __name__ == "__main__":
    main()
