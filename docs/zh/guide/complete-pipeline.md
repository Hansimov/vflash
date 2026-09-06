# 生成视频

Vflash 0.2.0 支持纯文字生成，也支持提示词加一至三张有序参考图，输出五秒 MP4。连续请求使用 Python 接口，单次生成也可以使用容器命令行。完整链路支持单张 RTX 4090 48 GB 上的 T2VA Base4 和 Ref2VA Turbo4。

## 各阶段的职责

Vflash 负责原生四步去噪、模型生命周期、本地参考图读取和 MP4 输出。文本编码和参考图编码采用固定版本的 Diffusers、Transformers、PEFT 适配器；音视频解码采用 H3 官方 VAE。它们是明确列出的依赖，不会被描述成新实现的原生内核。运行时不依赖 LightX2V 或业务服务。

每个请求使用四次去噪计算、五秒视频和原生 24 fps。宽高必须是 32 的整数倍，总像素不超过 `928 × 512`，宽高比在 1:4 至 4:1 之间。模型生成 124 帧，交付前 120 帧。提示词原样传入。Ref2VA 的一至三张图片按照传入顺序编号为 `<Picture 1>`、`<Picture 2>`、`<Picture 3>`，请在提示词中说明每张图的主体和用途。这些是视觉参考，不代表视频中的帧位置，也不是严格的关键帧约束。

准备资产前先选择[固定模型配置](../reference/pipeline-profiles)。Ref2VA 使用 `transformer_ref` 与 Ref4 v0.1；T2VA 使用 `transformer` 与 Base4 v1.0。两种模式分别持有已准备的资产和常驻实例；模式不匹配的请求会在执行前拒绝。

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
| `adapter_path` | [运行资产](../reference/runtime-assets) 中与所选配置匹配的固定 BF16 LoRA |
| `decoder_directory` | 同一官方 H3 版本的 `FL2VA` 目录，包含 `video_vae` 和 `audio_vae` |
| `artifact` | 所选配置的完整 BF16 原生资产，采用运行时 LoRA 残差 |
| `schedule_overlay` | 对应的四次计算 training-Euler 调度；Ref4 的视频/音频 shift 为 12/3，T2VA 为 6/3 |
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

Ref2VA 的 `--reference` 可以出现一至三次。T2VA 在准备时指定 `--profile t2va-turbo4-exact-sm89`，生成时不传 `--reference`；对应的 Python 请求是 `VideoRequest(prompt=..., seed=...)`。进度以 JSON 行写入 stderr，stdout 输出最终结果。每次命令独立加载并释放模型。预装环境及完整挂载示例见 [Docker 生成](./docker#pipeline)。

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

`trust_local_code=True` 允许加载已核验本地快照中的官方解码器 Python 文件。请先阅读[模型与代码许可证](../reference/license)。

同一实例可以连续处理多个请求。原生权重采用 block ring，为轮流使用显卡的编码器与 VAE 留出空间；CPU 模型副本由实例持有以便复用，因此也需要足够的主机内存。进度回调同步执行，抛出异常可取消请求；不要在回调内部调用 `close`。任务队列、账号、存储及多进程调度由应用负责。

集成验证使用了 240 GiB 主机内存上限。这是已测预算，不是测得的最低要求；去噪器单独运行时的 64 GiB 建议不包含这些编码器和解码器。

`VideoResult.elapsed_seconds` 覆盖成功 `generate` 调用从输入校验到参考图清理的完整耗时。`stages.input_preparation` 包含资产检查和图片加载，其中 `reference_loading_seconds` 已计入准备耗时。编码和媒体阶段还分别记录 `weight_resume_seconds`、`suspend_seconds`，以及 `capture_call_seconds` 或 `decode_call_seconds`；外层阶段耗时同时包含同步回调和协调开销。模型初始化单独报告；各阶段明细存在包含关系，不能全部视作独立耗时相加。

输出路径不能已经存在。编码、媒体核验以及 GPU 阶段清理成功后才发布视频。执行失败会关闭此实例并移除临时文件。`close()` 在等待 CUDA 完成后释放持有的模型与 hook，不会重置其他调用方的 CUDA 上下文。模型关闭后，CUDA 库仍可能保留进程级工作缓冲；需要释放整个 CUDA 上下文时，应退出该独立进程。

在只读容器中运行时，需要为 Triton 提供可写、且允许加载编译后共享库的缓存目录。带有 `noexec` 的临时文件系统不能作为该缓存目录。

## 如何理解正确性验证

条件张量与最终 latent 的实现一致性，应与画面质量分别评估。动作、外观、指令与声音都需要对照用户原始要求；张量一致不能让质量不合格的结果通过。

媒体验证分别检查编码前的画面与音频、五秒交付时长、帧数和声道。H.264 和 AAC 均为有损编码，整段 MP4 的哈希不能用于判断去噪器或音频解码器的数值等价性。

发布检查分别覆盖 14 个条件张量、最终 FP32 音视频 latent、解码画面及可播放成片。官方音频 VAE 在重复请求间可能产生微小浮点差异，早于 PCM 或 AAC 编码；因此不承诺音频逐位复现。音频有限值、声道、帧数、时钟及完整解码仍须通过检查。具体案例与适用范围见[发布验收](../reference/releases)。
