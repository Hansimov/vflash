---
search: false
---

# VDN 关键帧兼容性

VDNEngineSession 和 VDNKeyframePipeline 使用单独训练的八步模型。已有 SM89 集成需继续使用对应权重、逐行 FP8 加载器及原配方。实测范围为单张 48 GB 卡的 5–10 秒 I2VA 与 FL2VA。更改 H3Pipeline 的注意力后端无法切换到 VDN。新项目可从[模型配置](../reference/pipeline-profiles)选择关键帧路径。

[查看此实现的详细配方与测量记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/guide/community-vdn.md)
