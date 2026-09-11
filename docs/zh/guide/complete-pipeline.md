# 生成视频

Vflash 0.3.2 支持纯文字生成，也支持提示词加一至三张有序参考图，输出五秒 MP4。当前 main 还提供预览版官方 Base16 I2VA 链路，输入一张明确的首帧。连续请求使用 Python 接口，单次生成也可以使用容器命令行。完整链路支持单张 RTX 4090 48 GB 上的 T2VA Base4 和 Ref2VA Turbo4；Ref4 也支持双张 RTX 3080 20 GB。预览版 Base16 I2VA 使用单张 RTX 4090 48 GB。

单张 4090 的 Ref4 还支持[短视频参考](#reference-video)，同一实例可以交替处理图片和视频，无需切换权重。

## 各阶段的职责

Vflash 负责原生去噪、模型生命周期、本地图像读取和 MP4 输出。文本和图像编码采用固定版本的 Diffusers 与 Transformers；Turbo 配置还使用 PEFT 适配器。音视频解码采用 H3 官方 VAE。它们是明确列出的依赖，不会被描述成新实现的原生内核。运行时不依赖 LightX2V 或业务服务。

Turbo 请求使用四次去噪计算；预览版 Base16 I2VA 使用 16 次。两者都输出五秒视频并采用原生 24 fps 时钟。宽高必须是 32 的整数倍，总像素不超过 `928 × 512`，宽高比在 1:4 至 4:1 之间。模型生成 124 帧，交付前 120 帧。提示词原样传入。Ref2VA 的一至三张图片按照传入顺序编号为 `<Picture 1>`、`<Picture 2>`、`<Picture 3>`，请在提示词中说明每张图的主体和用途。这些是视觉参考，不代表视频中的帧位置，也不是严格的关键帧约束。I2VA 输入一张独立的第零帧锚点，提示词可将其称为 `<Picture 1>`。

准备资产前先选择[固定模型配置](../reference/pipeline-profiles)。Ref2VA 使用 `transformer_ref` 与 Ref4 v0.1；T2VA 使用 `transformer` 与 Base4 v1.0；预览版 I2VA 使用官方 `transformer`，无 LoRA 执行 16 次计算。不同配置分别持有已准备的资产和常驻实例；模式不匹配的请求会在执行前拒绝。

## 安装和模型资产

从发布版源码安装完整链路依赖，并在 `PATH` 中提供 `ffmpeg` 与 `ffprobe`：

```bash
python -m pip install '.[pipeline]'
```

依赖固定到经过核查的适配器版本，其中 Diffusers 的提交为 `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`。安装包不会自动下载模型权重。

本地资产配置明确指定六个路径：

| 字段 | 内容 |
| --- | --- |
| `model_directory` | `MiniMaxAI/MiniMax-H3` 在 `42ed227ee7df40d41602854ae760620d6eb651fe` 版本的官方 Diffusers 组件，包含所选模式的 `transformer_ref` 或 `transformer` |
| `adapter_path` | Turbo 配置使用[运行资产](../reference/runtime-assets)中的固定 BF16 LoRA；两种 Base16 I2VA 配置均填 `null` |
| `decoder_directory` | 同一官方 H3 版本的 `FL2VA` 目录，包含 `video_vae` 和 `audio_vae` |
| `artifact` | 所选配置的完整 BF16 原生资产；只有 Turbo 配置包含运行时 LoRA 残差 |
| `schedule_overlay` | 对应的 training-Euler 调度：Turbo 为四次计算，Base16 I2VA 为 16 次且视频/音频 shift 为 12/3 |
| `auxiliary_tensor` | 与上述资产匹配的原生输入、输出权重 |

后三项是准备好的原生资产，不能直接填入任意官方 checkpoint。可以按[官方权重编译流程](./compile-weights)创建；使用已有资产前，请核对[资产合同](../reference/runtime-assets)。目前尚未发布可直接下载的原生模型包。

先把资产放入最终的只读快照，再执行准备步骤。该步骤会按照内置的官方文件清单或原生资产清单，逐一核验实际字节，并检查来源、LoRA 和调度。这是一次性的磁盘操作。准备记录绑定到当前文件系统；后续启动和请求仅检查文件身份、大小及时间戳，不会反复计算模型权重哈希。移动或修改资产后需要重新准备。

```bash
vflash prepare-pipeline \
  --assets pipeline-assets.json --receipt prepared-assets.json
```

## 使用命令行生成

将 H3 提示词写入 `prompt.txt`，然后按期望的顺序传入参考图片：

```bash
vflash generate \
  --prepared-assets prepared-assets.json \
  --prompt-file prompt.txt \
  --reference subject.png --reference setting.png \
  --width 928 --height 512 --seed 1234 --gpu 0 \
  --output video.mp4 --trust-local-code
```

Ref2VA 的 `--reference` 可以出现一至三次。T2VA 在准备时指定 `--profile t2va-turbo4-exact-sm89`，生成时不传图像参数。预览版 I2VA 使用与硬件匹配的 SM89 或 SM86 Base16 配置，生成时传入且只传一个 `--first-frame first-frame.png`，不可与 `--reference` 混用；SM86 配置还须传入 `--peer-gpu 1 --strategy sequence-head`。对应的 Python 请求是 `VideoRequest(prompt=..., first_frame=Path("first-frame.png"), seed=...)`。进度以 JSON 行写入 stderr，stdout 输出最终结果。每次命令独立加载并释放模型。预装环境及完整挂载示例见 [Docker 生成](./docker#pipeline)。

## 一个明确拥有资源的实例

在独立进程中创建实例，且此前不能由其他代码初始化 CUDA。显卡由调用方明确选择。下例要求该进程只能看到一张兼容显卡：

```python
from pathlib import Path
from vflash.hardware import discover_nvidia_devices
from vflash.pipeline import H3Pipeline, VideoRequest, load_prepared_pipeline_assets

devices = discover_nvidia_devices()
if len(devices) != 1:
    raise RuntimeError("make one SM89 GPU visible to this process")
prepared = load_prepared_pipeline_assets(Path("prepared-assets.json"))
with H3Pipeline(prepared, device=devices[0], trust_local_code=True) as pipeline:
    result = pipeline.generate(
        VideoRequest(
            prompt=Path("prompt.txt").read_text(encoding="utf-8"),
            references=(Path("subject.png"), Path("setting.png")),
            seed=1234,
        ),
        Path("video.mp4"),
        progress=lambda event: print(event.stage, event.completed, event.total),
    )
    print(result.output_path, result.elapsed_seconds)
```

原有的单图参数 `reference=Path(...)` 仍可使用；它与 `references=(...)` 二选一。

使用双张 RTX 3080 20 GB 时，先准备 [SM86 资产](../reference/pipeline-profiles#sm86)，让进程仅看到这两张卡，并向 `H3Pipeline` 传入 `peer_device=devices[1], strategy="sequence-head"`。编码和解码使用 `devices[0]`，原生去噪使用双卡。单 SM86 或 `tensor` 完整链路会在模型加载前拒绝。命令行对应选项为 `--gpu 0 --peer-gpu 1 --strategy sequence-head`。

0.3.2 的构造函数只在 CPU 上检查配置；首次 `generate` 先完整读取、校验素材，再加载模型。服务可在接单前调用 `pipeline.prepare()` 预加载模型。重复调用会复用已有阶段，输入错误也不会销毁健康、已加载的实例。预加载不会执行条件编码或编译每种输入形状；首次使用仍可能产生算子准备成本。

`trust_local_code=True` 允许加载已核验本地快照中的官方解码器 Python 文件。请先阅读[模型与代码许可证](../reference/license)。

同一实例可以连续处理多个请求。原生权重采用 block ring，为轮流使用显卡的编码器与 VAE 留出空间；CPU 模型副本由实例持有以便复用，因此也需要足够的主机内存。进度回调同步执行，抛出异常可取消请求；不要在回调内部调用 `close`。任务队列、账号、存储及多进程调度由应用负责。

集成验证使用了 240 GiB 主机内存上限。这是已测预算，不是测得的最低要求；去噪器单独运行时的 64 GiB 建议不包含这些编码器和解码器。

`VideoResult.elapsed_seconds` 覆盖成功 `generate` 从输入校验到参考素材清理的完整耗时，包括首次模型加载。`stages.initialization_seconds` 是本次调用中的加载耗时，显式预加载后为零；`stages.request_elapsed_seconds` 仅扣除这部分加载，便于比较请求。`stages.session_initialization_seconds` 保留实例最初的加载成本，也可读取 `pipeline.initialization_seconds`。`stages.input_preparation` 包含资产检查和素材加载，其中 `reference_loading_seconds` 已计入准备耗时。编码和媒体阶段还分别记录 `weight_resume_seconds`、`suspend_seconds`，以及 `capture_call_seconds` 或 `decode_call_seconds`；外层阶段同时包含同步回调和协调开销。这些明细存在包含关系，不能全部视作独立耗时相加。

输出路径不能已经存在。编码、媒体核验以及 GPU 阶段清理成功后才发布视频。执行失败会关闭此实例并移除临时文件。`close()` 在等待 CUDA 完成后释放持有的模型与 hook，不会重置其他调用方的 CUDA 上下文。模型关闭后，CUDA 库仍可能保留进程级工作缓冲；需要释放整个 CUDA 上下文时，应退出该独立进程。

在只读容器中运行时，需要为 Triton 提供可写、且允许加载编译后共享库的缓存目录。带有 `noexec` 的临时文件系统不能作为该缓存目录。

## 使用参考视频 {#reference-video}

安装完整链路依赖或使用带版本号的 pipeline 镜像，使用与图片生成相同的 `ref2va-turbo4-exact-sm89` 准备资产和单张 RTX 4090 48 GB。这是依据参考生成新视频，不是逐帧精确编辑。

```bash
vflash generate \
  --prepared-assets prepared-assets.json --prompt-file video-prompt.txt \
  --reference-video reference.mp4 --gpu 0 --seed 1234 \
  --output variation.mp4 --trust-local-code
```

Python 使用 `VideoRequest(prompt=..., reference_video=Path("reference.mp4"))`。提示词以 **`<Video 1>`** 指代视频，保留其中的空格。同一 Ref4 实例可先接收图片、再接收视频、再回到图片，不切换模型或 LoRA。此前生成的本地 MP4 也可以作为下一次的 `reference_video`；引擎不会自动载入对话历史或改写提示词。

输入限一段完整的 **2–5 秒 MP4、MOV 或 WebM**，最大 **20 MiB**，源画面最多 **475,136 像素**（928×512 的面积）。源帧率可以是分数，必须可读取、大于零且不超过 240 fps。只接受一个视频流、方形像素和无歧义的直角旋转。已有源音轨会被丢弃，不参与新音轨生成；暂不支持图视频混合、音频输入、SM86 或 Ref8 视频条件。输出仍为五秒、24 fps，沿用上文的 32 对齐与画布面积限制。

CPU 解码器先固定文件副本，检查完整视频并重采样到 24 fps，保留片尾不足一帧的时间区间。它向官方视频 setup 传递源尺寸 RGB，再由官方缩放一次。视频参考尺寸独立于输出画布，**不使用图片的只缩小 `match` 策略**。入口也限制官方归一化后的画布：长边最多 1376、面积最多 1376×768、时序参考行最多 33,024。因此，即使源像素合格，过于狭长的比例仍可能被拒绝。预算限制不代表每种比例都已通过质量验证。临时文件和解码 RGB 只存活于当前请求。

视频条件比静态图片更耗资源：官方 VAE 编码完整时序块，文本编码器从视频中采样。请根据输入准备、条件编码、去噪和媒体交付的实际耗时规划服务；这里不承诺视频输入延迟。原生接口和支持范围见[视频条件包](../reference/runtime-assets#video-conditioning)。

## 如何理解正确性验证

条件张量与最终 latent 的实现一致性，应与画面质量分别评估。动作、外观、指令与声音都需要对照用户原始要求；张量一致不能让质量不合格的结果通过。

媒体验证分别检查编码前的画面与音频、五秒交付时长、帧数和声道。H.264 和 AAC 均为有损编码，整段 MP4 的哈希不能用于判断去噪器或音频解码器的数值等价性。

发布检查分别覆盖 14 个条件张量、最终 FP32 音视频 latent、解码画面及可播放成片。官方音频 VAE 在重复请求间可能产生微小浮点差异，早于 PCM 或 AAC 编码；因此不承诺音频逐位复现。音频有限值、声道、帧数、时钟及完整解码仍须通过检查。具体案例与适用范围见[发布验收](../reference/releases)。
