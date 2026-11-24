---
search: false
---

# 原版 FL 画面与 Larry 音轨

ReferenceAVRuntime 的十步画面采样分为四步 Larry 和六步原模型，再对冻结的画面运行十二步 Larry 音频采样。实测配置为单张 96 GiB SM120，一张与 32 对齐画幅一致的首图，5 秒，总面积最多为 2 × 1024²，交付 24 fps 共 120 帧。使用单独准备的 Pika 模型见 [Pika 画面与声音](../guide/pika-flex)。

[查看此实现的详细配方与测量记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/reference/segmented-av.md)
