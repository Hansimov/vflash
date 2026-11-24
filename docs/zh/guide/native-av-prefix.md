---
search: false
---

# 保留与续接原生潜变量

给 H3Pipeline.generate 的 latent_output 传入全新路径，可在媒体解码前保存完整音视频张量。该路径须与 MP4 分开。NativeAVPrefix 还要求匹配的模型表、源帧数及上下文。拼接时只保留一次重叠区，试听声音接缝。修改已编码的视频可使用[源片编辑](./remake)。

[查看此实现的详细配方与测量记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/guide/native-av-prefix.md)
