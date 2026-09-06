# 完整视频链路预览

此开发分支增加了从提示词和一张参考图生成 MP4 的 Python 接口。首个目标是单张 48 GiB SM89 显卡上的 Ref2VA Turbo4。它尚未包含在当前仅输出 latent 的公开发行版中；发布前还需要完成 GPU 验证，以及可独立运行的原生资产编译流程。

## 各阶段的职责

Vflash 负责原生四步去噪、模型生命周期、本地参考图读取和 MP4 输出。文本编码和参考图编码采用固定版本的 Diffusers、Transformers、PEFT 适配器；音视频解码采用 H3 官方 VAE。它们是明确列出的依赖，不会被描述成新实现的原生内核。运行时不依赖 LightX2V 或业务服务。

首个接口支持一张参考图、四次去噪计算、五秒视频和原生 24 fps。宽高必须是 32 的整数倍，总像素不超过 `928 × 512`，宽高比在 1:4 至 4:1 之间。模型生成 124 帧，交付前 120 帧。提示词原样传入，图片标签为 `<Picture 1>`。

## 安装和模型资产

从此分支安装完整链路依赖，并在 `PATH` 中提供 `ffmpeg` 与 `ffprobe`：

```bash
python -m pip install '.[pipeline]'
```

依赖固定到经过核查的适配器版本，其中 Diffusers 的提交为 `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`。安装包不会自动下载模型权重。

本地资产配置明确指定六个路径：

| 字段 | 内容 |
| --- | --- |
| `model_directory` | `MiniMaxAI/MiniMax-H3` 在 `42ed227ee7df40d41602854ae760620d6eb651fe` 版本的官方 Diffusers 组件目录，包括 `transformer_ref` |
| `adapter_path` | [运行资产](../reference/runtime-assets) 中固定的 Ref2VA Turbo4 v0.1 BF16 LoRA |
| `decoder_directory` | 同一官方 H3 版本的 `FL2VA` 目录，包含 `video_vae` 和 `audio_vae` |
| `artifact` | 完整 BF16 Ref4 原生资产，采用运行时 LoRA 残差 |
| `schedule_overlay` | 对应的四次计算 training-Euler 调度，视频 shift 12、音频 shift 3 |
| `auxiliary_tensor` | 与上述资产匹配的原生输入、输出权重 |

后三项是准备好的原生资产，不能直接填入任意官方 checkpoint。此分支尚未提供独立的编译流程或可直接下载的模型包。使用已有资产前，请核对[资产合同](../reference/runtime-assets)。

先把资产放入最终的只读快照，再执行准备步骤。该步骤会按照内置的官方文件清单或原生资产清单，逐一核验实际字节，并检查来源、LoRA 和调度。这是一次性的磁盘操作。准备记录绑定到当前文件系统；后续启动和请求仅检查文件身份、大小及时间戳，不会反复计算模型权重哈希。移动或修改资产后需要重新准备。

```python
from pathlib import Path
from vflash.pipeline import PipelineAssets, prepare_pipeline_assets

assets = PipelineAssets.from_json(Path("pipeline-assets.json"))
prepare_pipeline_assets(assets, Path("prepared-assets.json"))
```

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
            reference=Path("reference.png"),
            seed=1234,
        ),
        Path("video.mp4"),
        progress=lambda event: print(event.stage, event.completed, event.total),
    )
    print(result.output_path, result.elapsed_seconds)
```

`trust_local_code=True` 允许加载已核验本地快照中的官方解码器 Python 文件。请先阅读[模型与代码许可证](../reference/license)。

同一实例可以连续处理多个请求。原生权重采用 block ring，为轮流使用显卡的编码器与 VAE 留出空间；CPU 模型副本由实例持有以便复用，因此也需要足够的主机内存。进度回调同步执行，抛出异常可取消请求；不要在回调内部调用 `close`。任务队列、账号、存储及多进程调度由应用负责。

输出路径不能已经存在。编码、媒体核验以及 GPU 阶段清理成功后才发布视频。执行失败会关闭此实例并移除临时文件。`close()` 在等待 CUDA 完成后释放持有的模型与 hook，不会重置其他调用方的 CUDA 上下文。

## 如何理解正确性验证

条件张量与最终 latent 的实现一致性，应与画面质量分别评估。动作、外观、指令与声音都需要对照用户原始要求；张量一致不能让质量不合格的结果通过。

媒体验证分别检查编码前的画面与音频、五秒交付时长、帧数和声道。H.264 和 AAC 均为有损编码，整段 MP4 的哈希不能用于判断去噪器或音频解码器的数值等价性。
