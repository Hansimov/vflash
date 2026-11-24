---
search: false
---

# 输入条件与权重放置

<span id="extended-hybrid" aria-hidden="true"></span>


FL/Ref 混合模型共用 LightX 主干，另载入官方 Ref 的第 25–49 块 AdaLN 表。准备模型时需匹配主干版本。多图参考按原始顺序绑定。比较推理路径时固定参考图像素、种子与画幅；可用设备和显存组合见[硬件要求](./profiles)。

[查看此实现的详细配方与测量记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/guide/pipeline-conditioning.md)
