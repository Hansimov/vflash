# 使用外观参考图生成

PikaReferenceAVRuntime 使用有序照片作为外观参考，适合生成同一主体的新镜头。

## 选择时长

先从五秒配方开始。八秒和十二秒配方使用十步音频补采，请求时长需与配方一致。24 fps 下八秒交付 192 帧，十二秒交付 288 帧。

```python
from vflash.pipeline.pika_reference_recipe import PikaReferenceRecipe

recipe = PikaReferenceRecipe(duration_seconds=8, audio_steps=10)
```

## 安排参考图预算

在运行时的 1 MP 支持范围内处理，提示中的图片顺序与上传顺序一致。可从条件编码记录核对源图和实际编码尺寸。调整预算可能影响小字、手部和商品细节，应以原尺寸成片检查。

## 量化画面权重

QAD 画面权重另有清单和 SM120 内核；已有样片出现清晰度与动作幅度下降。需要复现该设置时，可参阅[固定版本的技术记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/guide/pika-reference-weights.md)。
