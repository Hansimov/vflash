# 配置与硬件

配置（profile）固定模型、LoRA 版本、步数和计算精度，默认内存策略随所选硬件而定。请选择同时匹配显卡和已编译资源的配置。
Vflash 不会自动替换设备或注意力后端；目录条目或 CUDA 探测都不能证明某个任务可以运行或达到质量要求。

```bash
vflash profiles
vflash plan ref2va-turbo8-exact-sm89 --gpu 0
```

## 可用配置 {#available}

Ref2VA 根据参考图生成，T2VA 仅用文本，关键帧配置处理 I2VA（首帧）、L2VA（尾帧）和 FL2VA（首尾）。Base 与 Ref 权重相互独立。
[完整流水线](./complete-pipeline)使用匹配的固定配置接收这些输入；原生接口使用编译后的条件 bundle，默认驻留方式可能不同。

| 显卡 | 配置 | 步数 | 权重加载 |
| --- | --- | ---: | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89` | 4 | 常驻显存 |
| RTX 4090 48 GB | `ref2va-turbo8-exact-sm89` | 8 | 常驻显存 |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` | 4 | 从主机内存流式加载 |
| RTX 4090 48 GB | `t2va-turbo4-exact-sm89` | 4 | 常驻显存 |
| 两张 RTX 3080 20 GB | `t2va-turbo4-exact-sm86` | 4 | 流式；`sequence-head` |
| RTX 4090 48 GB | `i2va-base16-bf16-sm89`、`fl2va-base16-bf16-sm89` | 16 | 常驻显存 |
| 两张 RTX 3080 20 GB | `i2va-base16-bf16-sm86`、`fl2va-base16-bf16-sm86` | 16 | 流式；`sequence-head` |
| RTX 4090 48 GB | LightX v0.1 / v1.0 544p 关键帧配对（`i2va-turbo4-v01-544-exact-sm89` 等） | 4 / 8 | 见[流水线配置](../reference/pipeline-profiles#lightx-v01) |

匹配的关键帧配对共享产物，一个已加载的 `H3Pipeline` 可串行处理 I2VA、L2VA 和 FL2VA；L2VA 是请求契约，不是独立的权重配置。
HTTP 服务只加载一个配置（4090 上默认 Turbo4），更换配置需用匹配资源重启服务。

## 支持的显卡 {#heterogeneous}

| 显卡 | 路径 | 说明 |
| --- | --- | --- |
| RTX 4090 48 GB（SM89） | 上述全部配置；原始 v0.1 混合模型（稠密或 Veda）；[最长边 2720 的原生高清](./complete-pipeline#modes-and-limits)；[5–15 秒/九图扩展混合](./complete-pipeline#extended-hybrid) | 参考平台。实测设备报告 48 GiB，24 GiB 显卡未确认能承载相同任务。 |
| 两张 RTX 3080 20 GB（SM86） | Ref4 原生（`sequence-head` 或 `tensor`）与完整流水线（`sequence-head`）；T2VA 与 Base16 关键帧使用 `sequence-head` | PCIe 3.0 x16、无 peer access，不需要 NVLink。 |
| 单张 RTX 3080 20 GB | Ref4 原生 latent | 已检查容量及串行/重叠加载的可重复性。 |
| H100 SXM 80 GB（SM90） | 原始 v0.1 混合 I2VA + `veda-triton`；≥75 GiB 可[主干常驻](#h100-resident) | 有完整成片证据；H100 PCIe 仍为预览。 |
| RTX PRO 6000 Blackwell 96 GB（SM120） | 混合 I2VA、L2VA、FL2VA 和图像 Ref + `veda-triton`；≥90 GiB 可[主干常驻](#sm120-resident) | Server 与 Workstation 均有完整产物证据，各模式范围见下文。 |
| H100 NVL | [有界块环 I2VA](#small-blackwell) | 画布与时长采用下表实测范围。 |
| RTX 5090 32 GB（SM120） | 块环，或小规格部署[显式开启常驻](#small-blackwell) | 常驻部署自行限定已测请求范围，例如 1MP / 5 秒。 |
| SM103、MIG 分区 | 预览 | 整卡资格作为独立步骤完成。 |

SM90/SM120 上的原始 v0.1 混合路径请准备至少 128 GB 主机内存并留余量；仅凭显存不能判断是否兼容。

## 内存与驻留 {#memory}

| 策略 | 权重位置 | 适用场景 |
| --- | --- | --- |
| 常驻（单 4090 原生配置默认） | 显存 | 多次请求复用权重，且显存足够放激活 |
| `block-ring`（完整流水线核心与 SM86 默认） | 主机内存，按块流式加载 | 用主机内存换显存余量；20 GB 显卡必需 |
| SM90/SM120 显式 `weight_residency="resident"` | 显存主干 | 持续暖态批次或降低主机内存（见下文） |

命令行用 `--weight-residency block-ring`，或使用 [Python 会话选项](./python#memory)。按实测的 928×512、四步原生任务，**每个 worker 至少预留
64 GiB 可用系统内存**。完整流水线还持有编码器和官方 VAE：集成检查使用 240 GiB 主机内存上限，一条代表性的三图 SM86 请求主机 RSS 达
114.67 GiB。这些是测试预算而非最低要求；更大输入和并发 worker 需各自测量。双 SM86 的编码与解码在主卡上进行，两卡共同去噪
（[双卡设置](./getting-started#native-latents)、[测量范围](../reference/benchmarks#sm86-parallel)）。

<a id="sm120-resident"></a>

### H100 与 Blackwell 主干常驻 {#h100-resident}

`H3Pipeline(..., weight_residency="resident")` 可在至少 75 GiB 显存的单张 SM90，或至少 90 GiB 的单张 SM120 上常驻主干。需要原始 v0.1
`HybridModel`、`attention_backend="veda-triton"`，并在**启动 Python 进程前**设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`。
SM90 范围为单首帧 I2VA、最长 15 秒、画布最多 1536×864。0.17.1 起，96 GiB SM120
也支持 L2VA、FL2VA 和图像 Ref 常驻。图像条件请求遵循原生高清联合预算
`宽×高×模型帧数 ≤ 2048²×124`，同时采用[完整流水线](./complete-pipeline#extended-hybrid)的各模式限制。
多参考图范围为最多 960×544、五秒。已完成的 SM120 产物包括：

| 模式 | 实测请求 |
| --- | --- |
| I2VA | 2048×2048 / 5 秒 |
| L2VA | 1344×768 / 15 秒 |
| FL2VA | 1440×1440 / 10 秒 |
| Ref2VA | 两张或九张图片，960×544 / 5 秒 |

这些记录确认对应输入的执行与交付；创作品质结合原始要求和参考图判断。块环保持默认内存策略。
下表保留早期同输入 I2VA 常驻对照；当前注意力方案的测量见[0.17 发布记录](../reference/releases#_0-17-0)。

| 同实例 A/B/A2 | H100 SXM 块环/常驻/块环 | RTX PRO 6000 Server 块环/常驻/块环 |
| --- | --- | --- |
| 初始化（秒） | 119.0 / 128.2 / 115.3 | 105.2 / 123.3 / 108.2 |
| 暖态 512²/5 秒（秒） | 27.1 / 17.0 / 19.6 | 21.0 / 15.7 / 19.2 |
| 1536×864/15 秒（秒） | — / 227.3 / — | 256.1 / 245.3 / 253.7 |
| 主机 RSS 峰值（GiB） | 103.0 / 62.9 / 102.9 | 108.7 / 68.7 / 108.7 |
| 去噪分配峰值（GiB） | 70.3（常驻，15 秒） | 70.3（常驻） |

常驻缩短暖态请求，但启动成本更高；本样本中约五到六条暖态短请求可摊平差异。适合持续批次或需要降低主机内存时，应以启动加批次总时间判断。

### NVL 与 RTX 5090 有界案例 {#small-blackwell}

| 显卡 | 已完成 I2VA（生成秒） | 限制 |
| --- | --- | --- |
| H100 NVL | 1920×1088/10 秒 288.1，1536×864/15 秒 263.9，2048²/5 秒 321.6，1440²/10 秒 289.9 | 初始化 46.6 秒；其他模式未资格 |
| RTX 5090 32 GB / 600 W | 2048²/5 秒 451.0（峰值 27.0 GiB），1440²/10 秒 392.4，1536×864/15 秒 364.9 | 块环 + expandable segments；500 W 卡 2048²/5 秒失败 |

上表记录对应块环案例的容量、完整解码和多时间点目视结果。
32 GiB RTX 5090 的小规格常驻部署可在启动前设置 `VFLASH_SM120_RESIDENT_GIB=30`（0.16.1 起支持）。
默认门槛保持 90 GiB；应用层按实测限制请求，例如 1MP / 5 秒。更大规格沿各自块环测量范围使用。
详见[0.16.1 发布记录](../reference/releases#_0-16-1)。

### 注意力选择

`veda_dense_edge_layers=3`、`4`、`5` 分别选择 44/6、42/8、40/10 的 Veda/密集层分配。
公开默认值为 **40/10**。SM120 的 `veda_dense_backend="sageattention2"` 在密集和稀疏注意力中均使用
INT8 QK / FP8 PV；`"frost-flash"` 使用 BF16 密集注意力和 INT8 QK / FP16 PV 稀疏注意力。
FP8 模型权重属于独立选项。[0.17 对照](../reference/releases#_0-17-0)记录这些显式配置的完整引擎耗时与具体场景品质证据。

### 容器中的设备发现

在分区或容器环境中设置 `VFLASH_DEVICE_DISCOVERY=cuda-visible`。隔离的 CUDA 子进程会将可见 UUID 与**分区显存**和 NVIDIA 枚举核对，绝不
用整卡容量代替。设备发现需显式启用，失败时明确报错。

### 为其他架构做可移植准备

可以在租用 GPU 之前完成 CPU 准备：

```python
from pathlib import Path
from vflash.pipeline.assets import load_prepared_pipeline_assets
from vflash.pipeline.portable import prepare_portable_assets
from vflash.native.h3_runtime_artifact import load_h3_runtime_artifact
from vflash.native.h3_schedule_overlay import load_h3_schedule_overlay
from vflash.native.h3_prepared_hybrid import prepare
from vflash.adapters.checkpoints import IndexedCheckpoint

source = load_prepared_pipeline_assets(Path("/models/prepared-sm89.json"))
prepared = prepare_portable_assets(source.assets, Path("/models/sm120-view"), "sm120")
artifact = load_h3_runtime_artifact(prepared.assets.artifact, verify_content_hashes=False)
overlay = load_h3_schedule_overlay(prepared.assets.schedule_overlay, artifact=artifact)
prepare(Path("/models/hybrid-sm120"), artifact, overlay, IndexedCheckpoint(Path("/models/transformer_ref")))
```

把新凭据和 `HybridModel(Path("/models/hybrid-sm120"))` 传给流水线，并选择 `attention_backend="veda-triton"`。已导入的源资源保持只读挂载。
可移植准备记录原编译目标，创建目录视图和小型调度副本，不对超大权重计算哈希；不得对服务文件建立硬链接，否则链接计数变化会使已有凭据失效。

## 文本生成视频 {#t2va}

`t2va-turbo4-exact-sm89` 使用 Base4 v1.0 产物、Base 辅助张量、6/3 视频/音频调度和不含参考图的 bundle，仅凭文本生成
（[准备资源](../reference/pipeline-profiles)）。Ref2VA 产物不能执行该任务。`t2va-turbo4-exact-sm86` 需要两张 3080 并使用 `sequence-head`；
应用自有阶段完成了 928×512/5 秒和 640×352/10 秒，不代表更大输入。SM89 上重复固定请求时，14 个条件张量和两个最终 FP32 latent 全部一致。

## Turbo LoRA 支持 {#lora}

Vflash 无需 LightX2V 框架即可运行固定版本的 [LightX2V H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) 适配器。适配器、调度和步数必须与已编译资源一致。

| 适配器 | 硬件 | 上游文件 |
| --- | --- | --- |
| Ref Turbo4 v0.1 | 一/两张 3080、一张 4090 | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` |
| Ref Turbo8 v1.0 768p | 一张 4090 | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |
| 关键帧 Turbo4 v0.1 544p | 一张 4090 | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` |
| 关键帧 Turbo8 v1.0 544p | 一张 4090 | `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` |
| T2VA Turbo4 v1.0 768p | SM89 或 SM86 双卡（alpha 128 / rank 128） | `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` |

固定版本见[运行资源](../reference/runtime-assets#versions)。Base16 关键帧配置不使用适配器，只支持表中列出的文件与版本。Turbo4/8 为蒸馏配置：
步数更少、计算更省，但不承诺达到 50 步基础模型的质量；`exact` 指注意力路径，不代表不同显卡产生相同张量。双卡保持精确注意力，但浮点舍入不同。

## 不在支持范围内 {#scope}

各配置的媒体验证范围以[完整配置表](../reference/pipeline-profiles)为准。单 SM86 与 Ref Turbo8 提供原生 bundle 到 latent 的接口。未列出的模式、适配器、
量化、时间设置、显存规格和更大的设备组都不在这些配置之内。HTTP API 提供一条串行去噪通道；账户、计费和分布式调度由应用负责。
