# 许可证与致谢

## Vflash 源码 {#source}

Vflash 源码采用 [Apache License 2.0](https://github.com/Hansimov/vflash/blob/main/LICENSE)。源码仓库不包含模型权重。

## 模型与 LoRA {#models}

MiniMax H3 和可选 LoRA 权重各自遵循其许可证、使用规则和地域条款。Vflash 的源码许可证不会授予这些权重的使用权，也不会改变上游限制。请阅读所用模型和 LoRA 确切版本的条款。

- [MiniMax H3 模型、文档与许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [LightX2V MiniMax H3 Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo)

## 致谢 {#acknowledgements}

Vflash 基于 MiniMax 公开的 H3 架构和模型权重开发。LightX2V 的开放推理实现及 Turbo LoRA 为兼容性开发和性能对比提供了重要参考。

运行时使用 [PyTorch](https://github.com/pytorch/pytorch)、[Triton](https://github.com/triton-lang/triton) 和 NVIDIA CUDA。同时感谢 [FlashAttention](https://github.com/Dao-AILab/flash-attention)、[CUTLASS](https://github.com/NVIDIA/cutlass)、[SageAttention](https://github.com/thu-ml/SageAttention)、[FastVideo](https://github.com/hao-ai-lab/FastVideo)、[Alibaba PAI](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) 和 [OpenVDN](https://huggingface.co/OpenVDN/vdn-minimax-h3) 社区为 H3 加速研究提供的工作。

致谢不表示当前版本包含所有被提及的实现或 LoRA。实际可用范围见[配置与硬件](../guide/profiles)。
