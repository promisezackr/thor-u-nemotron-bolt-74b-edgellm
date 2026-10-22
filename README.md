# Nemotron 3.5 Bolt 74B-A9B on NVIDIA DRIVE Thor (TensorRT Edge-LLM, NVFP4 + speculative decoding)

实测在 DRIVE Thor 域控板上以 NVFP4 精度运行 74B-A9B MoE 模型，TensorRT Edge-LLM + DFlash/LiLiCorr 投机解码，单路解码 **48–57 t/s**（无草稿 29 t/s），2 路并发聚合 65–80 t/s，OpenAI 兼容服务。本仓库只包含：目标平台、测试成绩、部署方法与必需脚本/补丁。

## 1. 目标平台

| 项目 | 规格 |
|---|---|
| 板子 | NVIDIA DRIVE AGX Thor (p3960, Tegra264), 12× Neoverse V3AE, 58 GB 统一 LPDDR5X |
| GPU | CC 10.1 / sm_101a, GPU 1530 MHz, EMC 4266 MHz（理论带宽 273 GB/s，实测有效 ~232 GB/s） |
| 系统 | DriveOS 7.0.3 (Ubuntu 24.04), CUDA 12.8 runtime, TensorRT 10.13.3；根文件系统只读，/etc /home 为 overlay（重启清空），持久目录 /zeekr_data /zeekr_map |
| GPU 可见内存 | = 大页池（`vm.nr_hugepages`），运行需 24576 页（48 GB） |
| 推理框架 | TensorRT Edge-LLM main（0.11.x, commit 1231d64）+ 本仓库补丁（SM101 适配、LiLiCorr 草稿、nemotron_h 修复） |
| 模型 | [NVIDIA-Nemotron-3.5-Bolt-74B-A9B](https://huggingface.co/nvidia)（88 层：40 Mamba-2 + 40 latent MoE + 8 attention），草稿 `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` |
| 权重精度 | 路由/共享专家 NVFP4 W4A16 (g16)，Mamba in/out_proj NVFP4 W4A16，latent fc1/fc2 与 attention BF16，lm_head NVFP4（起点为 `primitive-ai/Nemotron-3.5-Bolt-74B-A9B-mixed-NVFP4-FP8`） |

## 2. 测试成绩

条件：板子无其他 GPU 负载，greedy（temperature 0），输出 512 token，`--dflashBlockSize 6`，草稿 fold2（无相关性头）。

### 2.1 Edge-LLM `llm_inference`（不含 prefill）

| 场景 | t/s | 平均接受长度/轮 |
|---|---|---|
| 无草稿（基线） | 29.0 | — |
| 代码 512（4 次） | 48.5 / 51.8 / 55.5 / 57.4 | 3.2–4.4 |
| 英文 512 | 48.4 | 3.70 |
| 中文 512 | 52.3 | 3.94 |
| 图文（512×512 图 + 文本，C-RADIO 视觉编码器） | 42.7 | 3.21 |
| 2 路并发，相同代码提示（聚合） | 78.8–80.2 | 3.2–4.1 |
| 2 路并发，代码+英文混合（聚合） | 64.2–67.4 | 3.1–3.8 |

Block 大小扫描（代码 512，单路）：

| block | 接受长度 | t/s | 草稿 ms/轮 | 验证 ms/轮 |
|---|---|---|---|---|
| 2 | 1.91 | 34.3 | 7.8 | 48.0 |
| 4 | 3.18 | 48.5 | 7.7 | 58.0 |
| 5 | 3.61 | 51.0 | 7.7 | 63.1 |
| **6** | **4.38** | **57.4** | 7.7 | 68.6 |
| 8 | 4.92 | 56.4 | 7.7 | 79.6 |

### 2.2 OpenAI 兼容服务（`llm_server`，引擎 KV 32k）

| 提示 | 输出 token | TTFT | 纯解码 t/s | 含 TTFT t/s |
|---|---|---|---|---|
| 代码 512 | 512 | 197 ms | 53.0 | 51.9 |
| 英文 512 | 374 | 322 ms | 50.0 | 48.0 |
| 中文 512 | 512 | 367 ms | 50.2 | 48.5 |

### 2.3 长提示（分块 prefill，服务实测）

A16 MoE 插件限制单次 prefill ≤ 4096 token（引擎 maxInputLen 2048 × batch 2），超长提示由运行时补丁自动分块（每块 ≤ maxInputLen，Mamba/KV 状态经常驻槽位接续，DFlash 草稿按块做 round-0 prefill），上限 = 引擎 KV 容量。

| 提示 token | TTFT | 纯解码 t/s | 答案（首尾各埋一个暗号词） |
|---|---|---|---|
| 7,464 | 2.8 s | 44.6 | 正确 |
| 14,864 | 5.5 s | 47.2 | 正确 |
| 29,664 | 10.9 s | 46.2 | 正确 |

prefill 吞吐约 2.7k token/s（分块 2048）。

### 2.4 成本结构（为什么单路停在 ~55）

每轮验证读取 ≈ 4.9 GB 非专家权重 + N × 1.4–1.8 GB 路由专家（N = block），block 6 约 16 GB → 理论 63 ms，实测 69 ms（带宽效率 92%）。单路 80 t/s 需要每轮 ≥ 5.5 token 且 ≤ 70 ms，在 4-bit 专家下不可达；需 2-bit 级专家或更高带宽硬件。

## 3. 部署方法

### 3.1 板上目录

```
/zeekr_data/llm/
  edgellm/bin/        llm_build llm_inference llm_server llm_bench visual_build libNvInfer_edgellm_plugin.so.1.0
  edgellm-src/        打补丁的 tensorrt_edgellm Python 包（导出用）
  pyenv/site-packages torch 2.13 aarch64 + tensorrt 10.13.3 Python 绑定（ubuntu2404/sbsa python3-libnvinfer）
  ext/external_nvfp4_moe_weights.safetensors   37.2 GB 外置专家权重（引擎目录用硬链接共享）
  engines/bolt-spec-long/   spec_base.engine + spec_draft.engine + base_config/draft_config + embedding
  engines/bolt-vision/      C-RADIO 视觉引擎
  scripts/            本仓库 deploy/ 下的脚本
  inputs/             本仓库 deploy/inputs/
/zeekr_map/llm/export/      ONNX 导出（基座瘦身版、草稿）
```

磁盘要求：/zeekr_data 做一次基座导出 + 重外置需 **≥ 50 GB 空余**（42 GB 临时 ONNX）。

### 3.2 主机侧：从 checkpoint 到 ONNX 草稿

1. 应用补丁：`cd TensorRT-Edge-LLM && git checkout 1231d64 && git apply patches/edgellm-main-bolt.patch`，按官方手册交叉编译 `llm_build llm_inference llm_server NvInfer_edgellm_plugin`（CuTe DSL A16 组用 sm_101 产物，bf16 变体不可用）。
2. 基座 checkpoint：`tools/convert_bolt_ckpt.py <primitive-ai mixed NVFP4-FP8 目录> <out> --mamba nvfp4 --latent bf16 --attn bf16 --lm-head nvfp4`（专家分片硬链接；Mamba FP8 逐通道 → NVFP4 W4A16）。
3. 草稿：`tools/convert_lilicorr_draft.py <LiLiCorr HF 目录> <out> --fold-scales --fold-up 10 --fold-down 4 --fold-o 8`（**必需**：LiLiCorr 权重 std≈0.25，不折叠会在 fp16 中饱和，接受长度从 4.4 掉到 2.3）。
4. 草稿 ONNX（任意 3 层基座子集即可）：
   `python -m tensorrt_edgellm.scripts.export <base-l3> <out> --target-sm 110 --dflash-draft --dflash-draft-dir <draft>`

### 3.3 板上：导出基座、建引擎

```bash
# 基座导出 + 外置复用 + 2048×2 / KV 32k 引擎 + 草稿引擎 + 基准（约 25 分钟）
bash /zeekr_data/llm/scripts/build-bolt-long.sh /zeekr_map/llm/models/<bolt-edgellm-ckpt>
# 只重建草稿（复用基座引擎，硬链接）
bash /zeekr_data/llm/scripts/mk-spec-variant.sh <newTag> <draftOnnxDir> 8 bolt-spec-long
```

限制：A16 MoE 插件只接受 `maxBatchSize × maxInputLen ≤ 4096`，长提示靠补丁中的分块 prefill（`EDGELLM_PREFILL_CHUNK` 可再调小块长）；上下文上限由 `--maxKVCacheCapacity` 决定（`build-bolt-128k.sh` 从保留的瘦身 ONNX 重建 128k KV 引擎，KV 每 token 仅约 8 KB）。草稿引擎 `--maxDraftTreeSize 8`（tree 6/16 均更差）。`llm_build` 偶发在 TensorRT 编译阶段卡死，脚本内置 timeout + 重试。

### 3.4 运行

```bash
# 服务（OpenAI 兼容，端口 8081；自动补 10.0.0.3 地址与 24576 大页池）
bash /zeekr_data/llm/scripts/server-bolt.sh 8081 bolt-spec-long 6
python3 /zeekr_data/llm/scripts/api-bench.py 8081          # 代码/英文/中文 512 token 基准
# 命令行单次推理 / 基准
bash /zeekr_data/llm/scripts/run-bolt-best.sh /zeekr_data/llm/inputs/code_512.json 6
bash /zeekr_data/llm/scripts/sweep-fold.sh <draftOnnxDir> <tag>        # block 2–8 扫描
# 图文
bash /zeekr_data/llm/scripts/run-bolt-vision.sh bolt-spec-long /zeekr_data/llm/inputs/image_panda.json --specDecode --dflashBlockSize 6
```

服务请求示例：`POST /v1/chat/completions`，`{"model":"nemotron-bolt-74b","messages":[...],"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}`，支持 `stream`。

注意事项：
- 板子默认账号 nvidia/nvidia（脚本中 `BOARD_SUDO_PW` 可覆盖）；本板无开机钩子，重启后需手动运行 `server-bolt.sh`，IP 10.0.0.3 会回退为默认 10.0.0.2。
- 服务常驻占用约 46 GB GPU 内存；跑引擎构建或其他 GPU 任务前先 `pkill -x llm_server`。
- `llm_server` 对 DFlash 必须 `--specDraftStep 1`（脚本已带）；板上无 curl，健康检查用 python；`long-prompt-test.py [port] [tokens]` 做长提示回归。
- 导出后用 `deploy/reexternalize.py` 复用外置专家权重，避免再写 37 GB；core dump 落在 /zeekr_data/logs/core，`env.sh` 已 `ulimit -c 0`。

## 4. 关键修复（补丁内容）

- Edge-LLM NVFP4-A16 Blackwell 内核/插件对 SM101（CUDA 12.8）放开；MoE 插件 num_experts 白名单放开为 [1,512]；外置权重硬链接复用。
- `models/lilicorr/`：LiLiCorr 草稿（DFlash2 卷积层 + SWA/sink 注意力 + 折叠展开），导出后处理 `onnx_dtype_fixup.py`（TensorRT 混合精度），避开 CumSum / softmax-gather 融合 / Expand（Myelin 内部错误）。
- `nemotron_h`：按层 MoE 配置（异构专家数）、DFlash 目标层捕获、NVFP4 A16 q/k/v 用 `qkv_concat`。
- `llmBuilder.cpp`：DFlash 草稿 profile 最小行数；`dflashDecoder.cpp`：可选 LiLiCorr 相关性头走线（实测无增益，默认关闭）。
- `llmRankRuntime.cpp`：分块 prefill（单请求、文本、vanilla/DFlash；草稿按块 round-0 prefill），`tensor.cpp`：reshape 失败打印张量名/容量。
