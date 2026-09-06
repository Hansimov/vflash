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

完整链路的开发版适配器使用 [Diffusers](https://github.com/huggingface/diffusers)、[Transformers](https://github.com/huggingface/transformers)、[Accelerate](https://github.com/huggingface/accelerate)、[PEFT](https://github.com/huggingface/peft)，以及固定版本 MiniMax H3 模型附带的官方解码器代码。本地 MP4 输出使用 [FFmpeg](https://ffmpeg.org/legal.html)。这些依赖保留各自的许可证；Vflash 仓库不重新分发它们的模型权重。

## 完整链路依赖

完整镜像包含固定版本的 [Diffusers](https://github.com/huggingface/diffusers)、[Transformers](https://github.com/huggingface/transformers)、[Accelerate](https://github.com/huggingface/accelerate) 和 [PEFT](https://github.com/huggingface/peft) 适配器，其许可证与声明保留在各自安装包中。

MP4 编码使用 Debian 的 [FFmpeg 软件包](https://packages.debian.org/trixie/ffmpeg)。镜像内的 `/usr/share/doc/ffmpeg/copyright` 保留版权和许可声明，对应源码可通过 [Debian Sources](https://sources.debian.org/src/ffmpeg/) 获取。依赖仍按各自条款使用；H3 官方解码器代码来自单独核验的模型快照。
