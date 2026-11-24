---
search: false
---

# Sol 内核兼容性

双卡重排采用目标设备优先的布局。先比较代表尺寸的张量是否相同，再运行完整请求，分别记录通信耗时和生成总耗时。更改注意力精度、采样时序或权重需独立验证。下方版本记录列出上游机制、实测尺寸与暂缓采用的内核。

[查看此实现的详细配方与测量记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/reference/sol-engine-alignment.md)
