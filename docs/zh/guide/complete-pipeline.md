# 生成视频 {#generate-a-video}

先按[运行时与接口](./runtime-selection)选择完整视频路径：准备权重的 Pika6＋Larry 音轨、原版 BF16 Pika、固定配置 `H3Pipeline` 各有自己的输入、资产和依赖。下文介绍固定配置路径，不会自动选择 Pika。

`H3Pipeline`（Python）和 `vflash generate`（命令行）根据提示词，以及可选的图片或一段短参考视频，生成完整 MP4。
Vflash 负责原生去噪、阶段生命周期、本地媒体读取和 MP4 交付；文本/图像编码器使用固定版本的 Diffusers 与
Transformers 代码，视频与音频由官方 H3 VAE 解码。不需要 LightX2V 运行时或应用服务器。

本页描述当前版本。各版本的改动与对应证据见[版本更新](../reference/releases)，硬件容量见[配置与硬件](./profiles)。

## 模式与限制 {#modes-and-limits}

| 模式 | 输入 | 提示词标签 | 配置 |
| --- | --- | --- | --- |
| T2VA | 仅提示词 | — | `t2va-turbo4-exact-sm89` |
| Ref2VA | 1–3 张有序图片（扩展混合模型可到 9 张） | 按输入顺序 `<Picture 1>` … | Ref4（`ref2va-turbo4-exact-*`） |
| I2VA | 一张首帧锚点 | `<Picture 1>` | Base16 或 LightX v0.1 关键帧配置 |
| L2VA | 一张尾帧锚点 | `<Picture 1>` | 同一组关键帧配置 |
| FL2VA | 首尾两张锚点 | 按时间顺序 `<Picture 1>`、`<Picture 2>` | 同一组关键帧配置 |
| 参考视频 | 一段 2–5 秒视频 | `<Video 1>` | 单 SM89 Ref4；单 SM89／SM120 Veda 混合模型 |

参考图是视觉参考，不是时间轴位置。关键帧配置可用同一个已准备的流水线串行处理 I2VA、L2VA 和 FL2VA，无需重启；
`result.request_mode` 记录实际输入类型，`profile_id` 记录已准备的配置身份。其他配置会在执行前拒绝模式切换。
提示词按原文使用；请说明每张图的主体和作用。

**画布。** 宽高须为 32 的倍数，宽高比在 1:4 到 4:1 之间。普通请求最多 1,048,576 像素。图像条件原生高清最多
2048² 像素、最长边不超过 2720，例如 2720×1536（16:9）、1536×2720（9:16）、2368×1760（4:3）或 2048²。
参考视频输出为五秒：Veda 混合模型上限为 1,048,576 像素，原 Ref4 保持 475,136 像素。

**时长。** Turbo T2VA 与 Ref4 为五秒。Base16 关键帧配置接受 5–10 秒整数时长；[扩展混合模型](#extended-hybrid)
接受 5–15 秒。交付为 24 fps，例如 5 秒对应 124 个模型帧/120 个交付帧，15 秒对应 362/360。

**联合预算。** 画布与时长不能独立组合：`宽×高×模型帧数` 不得超过 `2048²×124`，因此 4MP 画布只能是五秒请求，
2048² 十五秒会被拒绝。三张以上的多图 Ref 限五秒和 960×544。这些是软件上限，服务系统只能对外提供在实际硬件上
通过资格验证的组合。

### 分配硬件前检查请求 {#check-a-request-before-allocating-hardware}

`validate_request` 执行与 `H3Pipeline.generate` 相同的准入检查，但不导入 Torch、不加载权重、不打开媒体，也不初始化 CUDA：

```python
from pathlib import Path
from vflash.pipeline import VideoRequest, validate_request

request = VideoRequest("Describe the motion.", first_frame=Path("first.png"),
                       width=2720, height=1536, duration_seconds=5)
validate_request(request, profile_id="i2va-turbo4-v01-544-exact-sm89",
                 hybrid=True, attention_backend="veda-triton",
                 parallel_strategy="single", weight_residency="block-ring")
```

通过准入只说明软件上可执行，不代表显存、资源完整性、硬件资格或输出质量。

## 安装与准备资源 {#installation-and-assets}

从发布版本的检出目录安装 pipeline 扩展，并确保 `PATH` 中有 `ffmpeg` 和 `ffprobe`：

```bash
python -m pip install '.[pipeline]'
python -m vflash.install_sol   # 仅单张 SM89 Base16 默认的 Sol 需要；不下载模型
```

扩展会固定适配器实现，包括 Diffusers 提交 `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`。先选择
[固定模型配置](../reference/pipeline-profiles)，再用六个显式路径描述资源：

| 字段 | 内容 |
| --- | --- |
| `model_directory` | `MiniMaxAI/MiniMax-H3` 在 `42ed227ee7df40d41602854ae760620d6eb651fe` 的官方 Diffusers 组件，含 `transformer_ref`（Ref）或 `transformer`（T2VA、关键帧） |
| `adapter_path` | Turbo 配置固定的 BF16 适配器（见[运行资源](../reference/runtime-assets)），Base16 为 `null` |
| `decoder_directory` | 同一版本的 `FL2VA` 目录，含 `video_vae` 和 `audio_vae` |
| `artifact` | 该配置完整的 BF16 原生产物 |
| `schedule_overlay` | 匹配的训练 Euler 调度（Turbo 4 次评估；Base16 16 次，偏移 12/3） |
| `auxiliary_tensor` | 与之匹配的原生输入/输出张量 |

后三项可用[官方权重编译](./compile-weights)生成，或对照[资源契约](../reference/runtime-assets)检查已有产物；目前不提供现成模型包。
把资源放到最终的只读位置后执行一次准备：

```bash
vflash prepare-pipeline --assets pipeline-assets.json --receipt prepared-assets.json
```

准备过程会按上游清单或产物 manifest 校验所有被使用的文件，并生成绑定本机文件系统的凭据；之后启动只检查文件身份和时间戳，
不再重复计算哈希。移动或修改资源后需要重新准备。

## 命令行生成 {#generate-from-the-command-line}

```bash
vflash generate \
  --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --reference subject.png --reference setting.png \
  --width 928 --height 512 --seed 1234 --gpu 0 \
  --output video.mp4 --trust-local-code
```

- Ref2VA：一到三个 `--reference`，顺序与提示词一致。
- T2VA：准备 `t2va-turbo4-exact-sm89`，不传图片参数。
- 关键帧：`--first-frame`（I2VA）、`--last-frame`（L2VA）或两者（FL2VA）；用 `--duration N` 指定时长。不要与 `--reference` 混用。
- 双卡：`--gpu 0 --peer-gpu 1 --strategy sequence-head`（SM86 双卡必需；两张相同 RTX 4090 48 GB 的 Base16 关键帧可选）。
- 显式 Veda 边缘层数：`--veda-dense-edge-layers 3`（见[注意力](#veda-dense-edges)）。

进度以 JSON 行写到 stderr，结果写到 stdout。每条命令都会自行加载并关闭模型；要在多次请求间常驻模型，请使用同一个
Python 流水线。容器用法见 [Docker 生成](./docker#pipeline)。

## Python 生成 {#generate-from-python}

在其他代码初始化 CUDA 之前，于独立进程中运行流水线，并显式选择设备：

```python
from pathlib import Path
from vflash.hardware import discover_nvidia_devices
from vflash.pipeline import H3Pipeline, VideoRequest, load_prepared_pipeline_assets

devices = discover_nvidia_devices()
prepared = load_prepared_pipeline_assets(Path("prepared-assets.json"))
with H3Pipeline(prepared, device=devices[0], trust_local_code=True) as pipeline:
    pipeline.prepare()                       # 可选：接收请求前预加载
    result = pipeline.generate(
        VideoRequest(prompt=Path("prompt.txt").read_text(encoding="utf-8"),
                     references=(Path("subject.png"), Path("setting.png")), seed=1234),
        Path("video.mp4"),
        progress=lambda event: print(event.stage, event.completed, event.total),
    )
    print(result.output_path, result.elapsed_seconds)
```

关键帧请求使用 `first_frame=`、`last_frame=` 或两者。双卡时传入 `peer_device=devices[1], strategy="sequence-head"`，
编码与解码仍在第一张卡上。两张 RTX 4090 处理一个请求能降低延迟，但消耗更多 GPU 秒；同时有两个请求时，应使用两个独立的单卡 worker。

**生命周期。** 构造时在 CPU 上检查配置；首次 `generate` 会先校验输入再加载模型。无效输入不会破坏已加载的实例；执行失败会
退役该流水线并清理临时文件。输出路径不得已存在，只有编码、探测和阶段清理全部成功后才发布视频。`close()` 在 CUDA 完成栅栏后
释放自有模型，绝不重置其他所有者的上下文。原生分块环为编码器和 VAE 留出显存，CPU 上的模型主副本常驻，因此主机内存需要容纳它们。
进度回调同步执行，可抛出异常取消，但不能在回调中调用 `close`。`trust_local_code=True` 会从已校验的本地快照加载官方解码器文件，
请先阅读[许可证](../reference/license)。

在只读容器中，请为 Triton 提供可写且可执行的缓存目录；`noexec` 临时文件系统不能作为该缓存。

管道媒体交付采用120秒停滞时限与600秒编码总时限；RGB输入块写入完成或编码文件增长会刷新停滞计时。
到期后回收自有编码进程并清理临时媒体。直接调用`encode_mp4`时可设置`stall_timeout_seconds`和
`timeout_seconds`，文件输入采用总时限。

## 交付选项 {#delivery-options}

| 选项 | 作用 |
| --- | --- |
| `keyframe_delivery_profile="exact-v1"` | 解码后恢复每个输入端点（LANCZOS 缩放到画布，四帧羽化）；默认保留 VAE 重建 |
| `audio_delivery_profile="web-v1"` | 有界增益，目标 −18 LUFS、−2 dBTP 限幅；默认 `unchanged` 保留解码波形 |
| `audio_delivery_profile="silent-v1"` | 原时长的数字静音立体声轨，只跳过波形解码；仅用于明确要求整段静音 |
| `generate_audio=False` | 仅视频生成：去噪前去掉音频行，不写音轨；同一 seed 下画面可能不同 |
| `generate_rgb(request, Path("x.npy"))` | 仅视频的 uint8 RGB 数组 `[模型帧, 高, 宽, 3]`；`required_frames=` 只解码所需帧对应的 VAE 块 |

命令行支持 `--keyframe-delivery-profile` 和 `--audio-delivery-profile`。`generate_audio=False` 与 RGB 交付仅限 Python 单卡路径。
完整流水线还会在官方音频解码器截断前施加联动立体声余量；可复用的上下文为
`vflash.media.audio_headroom.audio_decoder_headroom`，它是采样峰值保护，不会放大安静音频。以上选项都不能改善音色或同步。

## 使用参考视频 {#reference-video}

```bash
vflash generate --prepared-assets prepared-assets.json --prompt-file video-prompt.txt \
  --reference-video reference.mp4 --gpu 0 --seed 1234 --output variation.mp4 --trust-local-code
```

Python 中使用 `VideoRequest(prompt=..., reference_video=Path("reference.mp4"))`，以 **`<Video 1>`**
指代视频。Veda 混合模型在单 SM89／SM120 上复用共享官方 Ref 图；同一实例交替处理图像和视频。
输入为一段完整的 2–5 秒 MP4、MOV 或 WebM，最多 20 MiB、1344×768 源像素，一路方形像素视频流，
帧率上限 240 fps。源音频丢弃。官方编码规范化为 24 fps，保持固定时间对齐和前缀预算；
120 帧输入按官方网格取前 107 帧并编码为 32 个时间位置。输出是参考引导的五秒新视频。

每个条件编码器拥有最多 128 MiB 的 CPU 参考 latent 缓存，相同解码身份和几何重复使用时复用官方 VAE
结果；文本与随机噪声每次重新计算。缓存随 pipeline 关闭释放，调用者负责素材授权和隔离。
默认 `reference_video_profile="full"` 保留完整空间信息；显式 `"spatial-half"` 在编码后平均池化空间网格，
保留全部已编码时间位置。该近似选项只用于 Veda 混合路径，可能改变人脸、细节与构图。
身份一致性优先使用 full。具体张量协议见[条件契约](../reference/runtime-assets#video-conditioning)。

参考 latent 复用和压缩方案受 MIT 许可的
[RefMod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/517e8c419a8584fbf69fceb487bc39d83e6a6ec8) 启发。
本实现复用官方编码结果并提供显式空间池化；RefMod 的逐素材 MSE 拟合和音色参考属于另外的方案。

### 外观图与动作视频组合（0.21）

0.21混合路径将一份参考视频与一至三张有序图像组合：

```python
request = VideoRequest(
    prompt="<Picture 1> 提供外观；<Video 1> 提供动作和摄影机时序。",
    references=(Path("appearance.png"),),
    reference_video=Path("motion.mp4"),
    width=960, height=544, duration_seconds=5,
)
```

图像职责使用`<Picture 1>`至`<Picture 3>`，视频使用`<Video 1>`。图像沿目标面积、仅缩小的match策略，
视频沿官方尺寸规范；执行路径为单SM89／SM120 Veda混合模型，默认完整空间参考，输出保持五秒、最多1,048,576像素。
图像顺序、身份与目标画布参与参考缓存身份，同一导引搭配新外观时重新编码组合条件。

首条执行证据覆盖48 GiB SM89、一张外观图与五秒导引：960×544完整调用280.7秒，编码60.2秒、去噪201.3秒；
会话初始化另计90.9秒。成片保持人物外观与完整交接动作。同一简单动作的简洁／细化I2VA提示也完成交接，
调用76.1／70.5秒，编码和冷暖状态分别记录。按具体控制需求选择动作参考；更广场景、三图容量与目标硬件性能继续实测。
对于粗糙动作导引，可显式选择`reference_video_profile="spatial-half"`，仅池化最后一份视频latent的空间网格，
此前所有外观图latent及视频时间位置完整保留。相同初始场景的条件行数从32,766降至8,574，
SM89调用168.9秒（编码59.1、去噪90.1），仍完成交接。SM120完整／压缩分别在两个缓存宿主完成于110.3／79.4秒，
包含不同的冷编码成本；去噪为74.0／31.8秒。这里记录单场景的近似控制结果，更广的动作和身份覆盖继续实测。
0.21增加组合输入；此前版本保留图像／视频分别输入的合同。

## 集成与调优

<span id="extended-hybrid"></span><span id="exact-text-encoder-prefix"></span><span id="veda-dense-edges"></span><span id="memory-conditioning"></span><span id="fp8-loaders"></span><span id="timing-and-diagnostics"></span><span id="what-a-correctness-result-means"></span><span id="qualification-evidence"></span>
- [条件编码、显存与硬件配置](./pipeline-conditioning) — 将固定配置流水线接入新设备或修改数值路径时，先保留相同分辨率、配置和种子的可播放对照。

<span id="optional-audio-only-six-step-refinement-source-main"></span><span id="audio-hidden-context"></span><span id="audio-kv-overlap"></span><span id="audio-ring-weight-cache"></span><span id="explicit-native-face-repaint"></span><span id="原片音轨重建019"></span><span id="更长音频采样019"></span><span id="紧凑参数表019"></span>
- [原生音轨细化与人脸重绘](./pipeline-refinement) — 画面达标后可单独选择音频补采。人脸重绘另有遮罩、权重与执行限制，保存成片时应记录实际启用的选项。

<span id="保留原生音视频潜变量"></span><span id="实验性原生音画目标前缀"></span>
- [使用原生音视频潜变量续接](./native-av-prefix) — 通过 Python 在媒体编码前保留模型输出，并为下一段请求提供兼容的历史。相邻窗口的重叠长度、源片切点和音频时钟必须一致。

<span id="实验性-pika-图片参考运行时"></span><span id="实验性-qad-画面权重0260a2"></span>
- [Pika 参考图配方与 QAD 画面权重](./pika-reference-weights) — 运行环境配置见 [Pika H3 Flex](./pika-flex)。按参考输入、时长、权重与设备选择以下准备权重的路径。
