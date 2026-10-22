#!/usr/bin/env python3
"""Finish MoE weight externalization for an Edge-LLM ONNX export whose externalize step died (ENOSPC).

Streams the NVFP4 MoE plugin initializers out of model.onnx.data into a safetensors file at EXT_DIR
(may be on another filesystem; a symlink is left in the ONNX dir), rewrites model.onnx/model.onnx.data
without them, and patches config.json's external_weight_files manifest.

usage: reexternalize.py <onnx_llm_dir> <ext_dir>
"""
import json
import os
import sys
import time

import onnx

from tensorrt_edgellm.external_weights import (
    _add_external_weight_inputs, _find_nvfp4_moe_weight_initializers,
    _remove_externalized_initializers, _write_external_weight_file_from_external_data,
    patch_external_weight_manifest)

EXT_NAME = "external_nvfp4_moe_weights.safetensors"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    onnx_dir, ext_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    os.makedirs(ext_dir, exist_ok=True)
    model = onnx.load(onnx_path, load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    names = _find_nvfp4_moe_weight_initializers(model)
    log(f"{len(inits)} initializers, {len(names)} MoE plugin tensors to externalize")
    assert names, "no MoE plugin initializers found"

    ext_path = os.path.join(ext_dir, EXT_NAME)
    if os.path.exists(ext_path) and os.path.getsize(ext_path) > 0:
        log("external file already exists, skipping write:", ext_path)
    else:
        log("streaming external weights ->", ext_path)
        _write_external_weight_file_from_external_data(ext_dir, EXT_NAME, names, inits, onnx_path)
        log("wrote", os.path.getsize(ext_path) / 1e9, "GB")
    link = os.path.join(onnx_dir, EXT_NAME)
    if os.path.lexists(link):
        os.remove(link)
    if os.path.dirname(ext_path) != onnx_dir:
        os.symlink(ext_path, link)
        log("symlink", link, "->", ext_path)

    # Rewrite the ONNX without the externalized initializers (load the rest into memory first).
    _add_external_weight_inputs(model, names, inits)
    _remove_externalized_initializers(model, names)
    log("loading remaining external data into memory")
    onnx.load_external_data_for_model(model, onnx_dir)
    old_data = os.path.join(onnx_dir, "model.onnx.data")
    new_data = "model.onnx.data.new"
    log("saving slim ONNX")
    onnx.save_model(model, onnx_path + ".new", save_as_external_data=True, all_tensors_to_one_file=True,
                    location=new_data, convert_attribute=True)
    # Fix the location string inside the new model to the final name, then swap files.
    slim = onnx.load(onnx_path + ".new", load_external_data=False)
    for init in slim.graph.initializer:
        for kv in init.external_data:
            if kv.key == "location" and kv.value == new_data:
                kv.value = "model.onnx.data"
    onnx.save_model(slim, onnx_path + ".new")
    os.remove(old_data)
    os.rename(os.path.join(onnx_dir, new_data), old_data)
    os.replace(onnx_path + ".new", onnx_path)
    log("new model.onnx.data", os.path.getsize(old_data) / 1e9, "GB")

    manifest = [{"file": EXT_NAME, "kind": "nvfp4_moe_weights", "tensors": names}]
    patch_external_weight_manifest(onnx_dir, manifest)
    log("done")


if __name__ == "__main__":
    main()
