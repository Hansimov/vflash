# License and acknowledgements

## Vflash source code {#source}

Vflash is released under [Apache License 2.0](https://github.com/Hansimov/vflash/blob/main/LICENSE). The source repository contains no model weights.

## Models and adapters {#models}

MiniMax H3 and optional LoRA checkpoints retain their own licenses, usage rules, and geographic terms. The Vflash code license does not grant rights to those weights or override their restrictions. Read the terms for the exact model and adapter revision you use.

- [MiniMax H3 model, documentation, and license](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [LightX2V MiniMax H3 Turbo adapters](https://huggingface.co/lightx2v/Minimax-h3-Turbo)

## Thanks {#acknowledgements}

Vflash builds on the published H3 architecture and checkpoints from MiniMax. LightX2V's open inference work and Turbo adapters provide a valuable compatibility source and comparison reference.

The runtime uses [PyTorch](https://github.com/pytorch/pytorch) and [Triton](https://github.com/triton-lang/triton), with NVIDIA CUDA. We also acknowledge the work of the [FlashAttention](https://github.com/Dao-AILab/flash-attention), [CUTLASS](https://github.com/NVIDIA/cutlass), [SageAttention](https://github.com/thu-ml/SageAttention), [FastVideo](https://github.com/hao-ai-lab/FastVideo), [Alibaba PAI](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs), and [OpenVDN](https://huggingface.co/OpenVDN/vdn-minimax-h3) communities that informed H3 acceleration research.

Acknowledgement does not imply that every referenced implementation or adapter is included in this release. The supported runtime configurations are listed under [profiles and hardware](../guide/profiles).
