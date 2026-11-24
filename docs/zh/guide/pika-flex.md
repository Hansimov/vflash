# Pika 视频生成与音轨选择

对完整 96 GiB SM120 设备上的文字或首帧任务，可复用准备权重的运行时。先生成一条五秒样片，再分别修改时长、画幅或声音配方。

## 准备权重的 Pika 音画运行时 {#prepared-pika-av}

Pika 适配器执行六次画面推理。联合音轨来自音画推理，Larry10 和 Larry12 使用独立的原生音频补采。四秒请求已经验证 Larry10 路径。Python 构造函数接收准备好的画面与音频组件、清单和文本加载器；它们应属于同一配方。

```python
from vflash.pipeline import VideoRequest
from vflash.pipeline.pika_av import PikaAVRecipe
from vflash.pipeline.pika_av_runtime import PikaAVRuntime

# All paths, manifests and the text loader are explicit caller-owned components.
runtime = PikaAVRuntime(**prepared_components, **pika_components, trust_local_code=True)
try:
    result = runtime.generate(
        VideoRequest("A ceramic bird unfolds its wings.", width=960, height=544),
        output_directory,
        recipe=PikaAVRecipe(audio_steps=10, visual_attention="veda", overlap_decode=True),
    )
finally:
    runtime.close()
```

## 确定参考图的作用

固定开场时传入首帧。由外观照片生成新镜头时，选用独立 Ref 主干的 [Pika 参考图运行时](./pika-reference-weights)。已经有动作轨迹的源片可使用 [Remake](./remake)。

## 原版 BF16 执行 {#original-bf16-reference-execution}

对照原版 Pika 输出，或接入上游 FL/Ref 模型时，采用原版环境并固定其源码及权重。最后一次请求结束后释放运行时。

```python
from pathlib import Path
from vflash.pipeline.pika_flex import PikaFlexRequest
from vflash.pipeline.pika_flex_runtime import PikaFlexRuntime

runtime = PikaFlexRuntime(
    source_root=Path("Pika-H3-Flex"),
    base_path=Path("models/MiniMax-H3"),
    adapters_root=Path("models/Pika-H3-Flex"),
    trust_local_code=True,
)
try:
    result = runtime.generate(
        PikaFlexRequest(prompt="integrated_multimodal_description: [Shot 1] ...\n"
                               "overall_soundscape: ...\nnon_diegetic_music: N/A"),
        Path("outputs/take.mp4"),
    )
finally:
    runtime.close()
```
