---
search: false
---

# 加载注意力适配器

AttentionAdapter 在单张 SM89 的 Base16 关键帧会话中串行应用本地 FP32 DiT 注意力 LoRA。同时传入 safetensors 文件、秩和缩放值，在结果记录中保留适配器身份。退出上下文后恢复基础权重。其他配置与协作双卡需分别验证。

[查看此实现的详细配方与测量记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/reference/attention-lora.md)
