# 分段画面与原生声音

`ReferenceAVRuntime`在完整96 GiB SM120显卡上持有原始FL参考模型。默认前4步使用Larry v4 EMA，
后6步使用原始模型；随后在保持完整视频潜变量的条件下，以Larry执行12步H3原生声音生成。
声音阶段保留原画面，使用H3自身的音频模型。

视觉默认Veda：44层稀疏、首尾各3层SageAttention2密集计算，INT8 QK／FP8 PV。
完整Sage是另一项显式配置，不会自动回退。回执核验实际440次稀疏与60次密集块调用。
声音保留实时文本／音频行，复用视频KV。

首批范围是5秒I2VA：一张已按比例处理至目标画布的首图，宽高32对齐、至多2×1024²像素、
最长边2048。实际模型124帧，交付24fps／120帧；应用可将1920×1088模型画布裁切至1920×1080。
其他模式、时长、小显存卡和人脸修复继续使用各自明确配置的流水线。

声音条件保留全部时间位置和完整首帧；不超过1376×768时空间步幅为2，更大画布为4。
这属于近似声音条件处理，完整画面保持。小画布的视频解码与声音采样重叠，大画布顺序解码以保留显存。
同时采用QKV融合、严格BF16执行、单次提示编码和最终编码。请求结束或异常后恢复钩子、调度器和LoRA状态。

调用方显式提供固定版本资产和文本权重加载器，启用`trust_local_code=True`：原始FL行FP8基础矩阵、
原始十／十二步时间调制表、视频／音频shift12／3、Larry v4 EMA、官方解码器和Veda预测器。
基础矩阵不合并其他加速LoRA。每次模型调用核对实际时间网格，时间嵌入保留原始FP32 SiLU精度。

上游来源：[Larry固定版本](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/tree/43a74557ac3f6539db8e0f2a959d03feb7a81480)
和[VideoX-Fun固定版本](https://github.com/aigc-apps/VideoX-Fun/tree/13bb3c625e8b1e457d7124d92207b49e00e4a6fe)。
Larry十二步声音是下游验证的运行点；作者推荐四至八步。资产分发、许可、准入和结果发布由调用方负责，
Vflash不携带私有资产或应用状态。完整资产合同见[英文页](../../reference/segmented-av)。

`runtime.generate(VideoRequest(...), output_directory)`返回MP4、原生潜变量、各阶段步数／CUDA耗时、
注意力覆盖、条件复用、媒体阶段、解码窗口显存峰值和完整调用耗时。加载、采购、上传与浏览器可播放时间另计。
原`H3Pipeline`保持既有合同。运动模糊、远景人脸和时序身份仍需按实际创作要求审看。

## Python集成

从[0.22.0发行包](https://github.com/Hansimov/vflash/releases/tag/v0.22.0)安装 `pipeline` extra，
并安装本运行时对应的固定Veda／Sage依赖。下例是显式资产集成函数，不自动下载资产，也不是CLI profile。
本例首图必须预先处理为1024×1024，每次输出目录必须尚不存在。

```python
from pathlib import Path
from vflash.pipeline import VideoRequest
from vflash.pipeline.reference_av_runtime import ReferenceAVRuntime


def generate_clip(*, cache_root, source_root, tables, adapter, decoder_directory,
                  predictor, text_loader, first_frame, prompt, output_directory):
    runtime = ReferenceAVRuntime(
        cache_root=Path(cache_root), source_root=Path(source_root),
        tables=Path(tables), adapter=Path(adapter),
        decoder_directory=Path(decoder_directory), predictor=Path(predictor),
        text_loader=text_loader, trust_local_code=True,
    )
    try:
        request = VideoRequest(
            prompt=prompt, first_frame=Path(first_frame), width=1024, height=1024,
            duration_seconds=5, seed=42, audio_delivery_profile="web-v1",
        )
        return runtime.generate(request, Path(output_directory))
    finally:
        runtime.close()
```

`text_loader`是无参数回调，返回已加载的Qwen3-VL文本编码器，并负责其量化布局与来源。
可复用 `vflash.native.h3_fp8_text.load_row_fp8_text_encoder`，显式提供 `RowFP8TextShard`、
匹配的 `RowFP8Execution` 和原始未量化分片。原始FL缓存不能替换为已合并LightX的主干。

运行时不实现上下文管理器，必须在 `finally` 中调用 `close()`，同一实例串行服务。
`result["path"]`为MP4，输出目录保留 `visual.safetensors` 与 `native.safetensors`。
`seconds`不含构造加载和应用交付；`peak_cuda_bytes`在媒体解码器重置峰值后读取，
**不是整次请求显存峰值**，部署容量应另测完整窗口。默认调用执行22次评估。
