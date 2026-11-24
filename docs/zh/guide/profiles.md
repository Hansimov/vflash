# 硬件要求

下表按参考型号的**单卡 FP32 峰值算力**降序排列。显存列为 Vflash 实测配置。实际出片时间还取决于张量精度、内存带宽、权重如何加载以及双卡互联。

<span id="heterogeneous" aria-hidden="true"></span>

| 显卡 | 参考 FP32 算力 | 实测显存 | 适用范围 |
| --- | ---: | --- | --- |
| RTX PRO 6000 Blackwell Server | 120 TFLOPS | 96 GiB | 准备权重的 Pika、原版 Remake 及常驻混合主干。Pika 音画配方使用完整显卡。 |
| RTX 4090 | 82.6 TFLOPS | 48 GiB | SM89 文字、参考图与关键帧固定配置。使用高清或常驻主干前检查可用显存。 |
| H100 SXM | 67 TFLOPS | 80 GiB | SM90 混合首帧生成，常驻主干要求至少 75 GiB。 |
| 双 RTX 3080 | 每张参考卡 29.8 TFLOPS | 2 × 20 GiB | SM86 分块加载，两卡共同完成一个请求。实测原生工作负载每个 worker 需要至少 64 GiB 可用主机内存。 |

4090 与 3080 的算力采用 NVIDIA 参考型号规格，Vflash 公开实测使用了扩展显存配置。运行 `vflash doctor` 查看自己的显卡。双卡各自受本卡显存容量限制。

## 选择配置 {#available}

用 `vflash profiles` 列出配置标识，再以 `vflash plan PROFILE --gpu 0` 检查设备匹配。双卡协作可加 `--peer-gpu 1 --strategy sequence-head`。在 H100 或 Blackwell 上常驻混合主干时，应先满足[配置和分配器要求](./pipeline-conditioning)。

## 规格来源

峰值规格来自 [RTX PRO 6000 Server](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)、[Ada 架构白皮书附录 A](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-V2.02.pdf)、[H100 SXM 规格](https://www.nvidia.com/en-us/data-center/h100/)和 [Ampere GA102 白皮书](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)。某一具体请求的耗时请参见[实测记录](../reference/benchmarks)。

## 常驻与受限显存配置

<span id="h100-resident" aria-hidden="true"></span>
<span id="sm120-resident" aria-hidden="true"></span>
<span id="memory" aria-hidden="true"></span>
<span id="small-blackwell" aria-hidden="true"></span>
<span id="lora" aria-hidden="true"></span>

常驻主干需要为输入编码与媒体解码预留显存。H100 NVL 和 RTX 5090 的有界试验、适配器选择及放置参数可在[对应版本的硬件记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/guide/profiles.md)中核对。
