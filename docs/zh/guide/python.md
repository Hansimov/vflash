# Python 集成

需要完整 MP4、并希望在多次请求之间保持模型常驻时，使用 **`H3Pipeline`**。只有在应用已自行构建兼容的条件
bundle、需要 latent 输出时，才使用 **`NativeEngineSession`**（[见下文](#native-session)）。两者都固定一个配置和
设备选择，并**串行**处理请求。需要进程隔离和 HTTP 队列时，运行 [Docker 服务](./docker)。

## 常驻流水线 {#pipeline}

在专用进程中运行流水线，在任何代码初始化 CUDA 之前选择显卡，并在 worker 生命周期内只保留一个实例：

```python
from pathlib import Path
from vflash.hardware import discover_nvidia_devices
from vflash.pipeline import H3Pipeline, VideoRequest, load_prepared_pipeline_assets

devices = {d.index: d for d in discover_nvidia_devices()}   # 索引与 `vflash doctor` 一致
prepared = load_prepared_pipeline_assets(Path("prepared-assets.json"))

with H3Pipeline(prepared, device=devices[0], trust_local_code=True) as pipeline:
    pipeline.prepare()                                  # 接收任务前先加载
    for name, prompt, first in jobs:                    # 你的队列
        result = pipeline.generate(
            VideoRequest(prompt=prompt, first_frame=first, width=928, height=512,
                         duration_seconds=5, seed=1234),
            Path(f"out/{name}.mp4"),
            progress=lambda e: print(e.stage, e.completed, e.total),
        )
        print(result.output_path, result.elapsed_seconds)
```

`VideoRequest` 用 `references=(…)` 表示 Ref2VA，用 `first_frame=` / `last_frame=` 表示关键帧，两者都不传即 T2VA，
均需在所准备配置的[模式与限制](./complete-pipeline#modes-and-limits)之内。关键帧配置在 I2VA、L2VA 与 FL2VA
之间切换无需重新加载。

## 长时长、多参考图与原生高清 {#hybrid}

十五秒请求、最多九张参考图以及 1MP 以上（最高 2720×1536）的画布，需要原始 LightX v0.1 关键帧配置配合
**混合模型**与 Veda 注意力：

```python
from vflash.pipeline import HybridModel

pipeline = H3Pipeline(
    prepared, device=devices[0], trust_local_code=True,
    hybrid_model=HybridModel(Path("models/MiniMax-H3/transformer_ref")),
    attention_backend="veda-sm89",                       # SM89；SM90/SM120 使用 "veda-triton"
    veda_predictor=Path("models/veda-predictor.safetensors"),
    veda_dense_edge_layers=3,                            # 44 稀疏层 / 6 稠密层
)
```

用 `python -m vflash.install_veda` 安装 Veda 并提供其预测器（见 [Veda 注意力](./veda)）。可以在不加载任何内容的
情况下按同一准入规则检查请求：

```python
from vflash.pipeline import validate_request
validate_request(VideoRequest("…", first_frame=Path("f.png"), width=2720, height=1536),
                 profile_id=prepared.profile_id, hybrid=True, attention_backend="veda-triton",
                 parallel_strategy="single", weight_residency="block-ring")
```

## 显存布局 {#memory}

| 设置 | 去噪器权重位置 | 适用 |
| --- | --- | --- |
| `weight_residency="block-ring"`（完整流水线默认） | 主机内存，按块流式加载 | 为编码器和 VAE 留出显存；需要大量主机内存 |
| 单张 ≥75 GiB SM90 或 ≥90 GiB SM120 上 `weight_residency="resident"` | 显卡 | 暖态请求更快；需在启动 Python 前设置 `PYTORCH_ALLOC_CONF=expandable_segments:True` |

常驻执行支持单首帧 I2VA、最长 15 秒：SM90 画布最多 1536×864 像素；SM120 遵循原生高清预算
`宽×高×模型帧数 ≤ 2048²×124`（0.15）。单 SM89 原生会话与 3080 会话有各自默认值，见[选择硬件与配置](./profiles#memory)。

## 可选音频修复 {#audio}

使用 Veda 44/6 的混合 I2VA 可以在视频完成后追加六次原模型音频评估：

```python
from vflash.native.h3_audio_refinement import AudioRefinementRequest
from vflash.native.h3_refinement_schedule import refinement_schedule

refine = AudioRefinementRequest(refinement_schedule(steps=6, denoise=0.5), tuple(audio_tables))
result = pipeline.generate(request, Path("out.mp4"), audio_refinement=refine)
```

视频帧保持不变。前提条件、可选的隐藏上下文复用及其限制见[生成视频](./complete-pipeline#optional-audio-only-six-step-refinement-source-main)。

## 进度、结果与错误 {#lifetime}

- **进度**事件包含 `stage`、`completed` 和 `total`，覆盖编码、去噪、可选音频修复与解码。回调同步执行；在回调中抛出
  异常会取消请求。不要在回调中调用 `close()`。
- **耗时。** `result.elapsed_seconds` 包含本次调用中的模型加载；`result.stages.request_elapsed_seconds` 不含加载。
  冷、暖请求请分开报告。
- **失败。** 非法输入在加载前被拒绝，流水线仍可继续使用。执行失败会使实例退役并清理临时文件：请新建流水线（或新的
  worker 进程）。
- **输出**只在编码和探测成功后发布；输出路径必须不存在。
- **关闭**使用 `with` 或 `close()`，会等待设备完成后释放本实例拥有的模型。

## 原生会话（latent） {#native-session}

`NativeEngineSession` 从编译好的条件 bundle 运行一个原生配置，输出视频与音频 latent，适合自行负责条件构建与
解码的应用。

```python
from vflash.catalog import ProfileCatalog
from vflash.native.runner import NativeEngineSession
from vflash.planner import resolve_plan

plan = resolve_plan(ProfileCatalog.bundled(), profile_id="ref2va-turbo4-exact-sm89", device=devices[0])
with NativeEngineSession(plan, artifact=Path("artifact"), schedule_overlay=Path("schedule"),
                         auxiliary_tensor=Path("auxiliary.safetensors")) as session:
    result = session.generate(Path("bundles/a"), Path("out/a.safetensors"),
                              progress_callback=lambda done, total: print(done, total))
```

两张 RTX 3080 20 GB 时，用 `profile_id="ref2va-turbo4-exact-sm86"`、`peer_device=devices[1]` 和
`strategy="sequence-head"` 解析 plan，两张卡服务同一个请求。4090 会话默认常驻，可传
`weight_residency="block-ring"`；SM86 始终流式加载。步进回调只报告去噪评估，并在每步增加一次同步。资源契约见
[运行资源](../reference/runtime-assets)；可选的[注意力 LoRA](../reference/attention-lora) 适用于独占的单 SM89
Base16 会话。
