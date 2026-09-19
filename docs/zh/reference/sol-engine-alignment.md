# SM86 与 SM89 上的 Sol-Engine 对齐

Vflash 把 NVIDIA Sol-Engine 当作优化机制的上游参考，而不是整套照搬的配置。当前实际部署目标是
RTX 3080 20 GB（SM86）和 RTX 4090 48 GB（SM89），通常为单卡或通过 PCIe 协作的双卡。
Sol-H3 最新的主要结果使用了不同的加速器、互联、步数、适配器、精度和注意力语义。

本页把对照源码固定为 2026-09-19 的 Sol-Engine
[`ca26dbd`](https://github.com/NVlabs/Sana/tree/ca26dbd2b7034cc90a64c093d715d99b0bfa5b7f)。
上游版本变化后，需要重新复核，不能自动继承结论。

## 当前启用的优化

双卡 `sequence-head` 路径现在用直接 Triton 重排完成 QKV 打包和返回 attention head 的合并。
collective 的线上布局和 BF16 数值不变；内核直接按照源 stride 读取，并一次写入 destination-major
存储，取代 `stack/permute/contiguous` 或 `cat/permute/contiguous` 的多次物化。机制来自
Sol-Engine 的 Ulysses 直接重排，但 Vflash 保留自己的四分块重叠通信布局和资源所有权。

该实现默认用于两种受支持架构上的 CUDA 张量。仅在受控诊断时设置
`VFLASH_H3_DIRECT_RELAYOUT=0`；其他值会直接报错。CPU 或不符合布局合同的张量继续使用参考实现。

在 Base16 的代表性 55,413-token 形状下，A/直接重排/A 的热态拷贝中位数如下。所有候选均逐元素
等于参考输出。

| 操作 | SM86 参考 → 直接重排 | SM86 倍率 | SM89 参考 → 直接重排 | SM89 倍率 |
| --- | ---: | ---: | ---: | ---: |
| QKV destination 打包 | 8.627 → 3.551 ms | 2.429× | 5.152 → 2.639 ms | 1.952× |
| attention 结果分片 | 0.453 → 0.345 ms | 1.312× | 0.334 → 0.266 ms | 1.254× |
| 返回 head 合并 | 2.501 → 1.234 ms | 2.027× | 1.762 → 0.912 ms | 1.932× |

两种架构的分配 screen 还把 QKV 打包的增量峰值从 2,383,245,312 降到 1,191,622,656 字节，
节省 1,136.4 MiB（50%）；返回 head 合并从 794,415,104 降到 397,207,552 字节，节省
378.8 MiB（50%）。attention 结果分片仍只保留 94.7 MiB 输出，峰值不变。这些 buffer 位于不同
执行点，不能相加后冒充完整 pipeline 峰值。

最终未开启 profile 的完整请求 A/直接重排/A 在同一常驻 pipeline 中执行，并以最后一个控制臂作为热态分母：

| 硬件与固定负载 | 热态控制 → 直接重排 | 请求降低 | 去噪降低 |
| --- | ---: | ---: | ---: |
| 双 SM86 · 5 秒 · 512² I2VA · 16 次评估 | 174.776 → 172.149 秒 | 1.503% | 1.995% |
| 双 SM89 · 10 秒 · 736 × 992 L2VA · 16 次评估 | 474.669 → 472.429 秒 | 0.472% | 0.623% |

三个臂的压缩视频 elementary stream 一致。官方音频 VAE 仍有已知的重复解码波动，与本次仅修改去噪
布局无关；两轮均未出现热降频样本。因此直接重排因精确布局和瞬时显存改善而默认启用，不把它宣传为
完整视频的大幅加速。完整归因和测量边界见[性能指南](./performance)。

## 与上游的差距和处理决定

| Sol-H3 机制 | Vflash 当前状态 | 当前 SM86/SM89 决定 |
| --- | --- | --- |
| destination-major QKV 打包和 head 合并 | 同一机制，Vflash 自有四分块布局 | **已采用。** 仅移动元素，且两种架构均已实测。 |
| AdaLN 预计算 | schedule overlay 已保存固定步数的 AdaLN 表 | **已经存在。** 不重复移植。 |
| modulation、gate、rotary 和 SwiGLU 融合 | 严格 Triton 内核保留既有 BF16 舍入边界 | **已经存在。** 保持精确默认。 |
| 合并 QKV 投影 | 编译器把 Q/K/V 保存为一个 `attn.qkv` 权重，运行时先执行一次宽 GEMM 再拆分 view | **已经存在。** 不再增加第二层投影包装。 |
| regional `torch.compile` | 上游 4090 路径只编译选定区域，但把收益包含在整体 “lossless opt” 中，没有单独归因 | **暂缓。** Vflash 已显式拥有固定 block 循环和热点内核；只有在两种目标架构分别完成纯编译变量、持久 worker 的 A/B/A 后，才考虑增加编译层。 |
| 分层组件 offload 与临时 VAE 常驻 | Vflash 使用显式 CPU master、双槽 block ring，以及 encoder/core/VAE 串行所有权 | **已用不同生命周期实现。** 不叠加第二套 offload manager；只有阶段耗时和峰值显存证明有收益时才修改现有生命周期。 |
| LoRA consumer fusion | Base16 没有 adapter；已准入的 Turbo 路径已融合部分 QKV merge 和 FFN adapter/activation consumer | **部分已有且限定 profile。** 上游 FastH3 adapter 不能互换，未准入 adapter 继续走显式 residual 路径。 |
| RMSNorm/AdaLN 与 QKNorm/RoPE/pack 组合融合 | 上游融合改变 reduction 或 rotary 的算术边界 | **不直接复制。** 需同架构数值和完整请求独立证明。 |
| SOL/BSA 稀疏注意力 | 上游的可选近似 attention | **不是生产后端。** 保守 SM89 候选改变了生成轨迹，并且没有越过预注册完整请求门槛。 |
| TeaCache / FirstBlockCache | 上游 4090 结果为 49 次 DiT forward，复用其中 35 次 | **暂缓。** Vflash Base16 只有 16 次评估，收益空间与质量风险不同。 |
| INT8 QKV 与 FP8 输出传输 | 八张 B300 快速配置的默认项 | **不进入精确默认。** 有损传输和不同拓扑需要独立 profile 与质量门。 |
| MXFP8 融合线性层 | SM100 系列路径；上游报告了明显输出差异 | **不在范围内。** 不面向 SM86/SM89，也不能继承 exact 名称。 |
| 音视频 VAE 并行和 tile sharding | 围绕八 rank Sol-H3 decoder 生命周期设计 | **暂缓。** Vflash 双卡在去噪后由主卡持有媒体阶段；应先改进生命周期，再证明完整媒体等价。 |
| 四步 FastH3 adapter | 主要对照把 49 次 forward 改成四次 | **不算运行时优化。** adapter 质量和模型语义需要独立 profile。 |
| H3 草稿 → LTX 精修与 TAEH 解码 | 单独的粗到细产品，同时改变模型、adapter、解码器、分辨率链和采样调度 | **不进入当前 exact profile。** 只有另建具名链路并通过质量门后才可采用，不能写成 SM86/SM89 运行时提速。 |

## 精确与近似是两类产品

已发布的 Vflash profile 使用 dense PyTorch Flash SDPA 和精确 BF16 传输。这里的“精确”表示所选实现
保持声明的算术与注意力合同；它不表示不同 GPU 架构、不同并行拆分或蒸馏 adapter 必须输出相同张量。

近似方法可能有价值，但必须具名、默认关闭、可测量且可移除。它们需要固定的提示词/参考/seed 套件、
完整解码视频与音频复核、实际播放与试听、重复计时、温度/稳定性证据和清晰回滚。仅有张量相似度或上游
样片不能完成资格验证。

Sol-H3 的 1/4/8 张 B300 表格组合了四次 forward 的 adapter、稀疏 attention、量化通信和并行解码；
可选 MXFP8 行也明确是有损模式。这些数字都不是 Vflash dense Base16、16 次评估的 SM86/SM89
运行时倍率。

## 如何复现实验

固定 artifact、请求和设备组，在同一个持久化进程中预热两臂并执行 A/B/A。初始化与成功请求边界要分开
报告；双卡路径还要分别报告单请求延迟和双 worker 吞吐。`--profile-denoise` 只用于归因，最终吞吐应使用
未开启 profile 的运行。

不能因为局部 kernel 更快就直接采用。候选只有在完整请求、解码媒体、资源清理和目标硬件重复测量都通过后，
才能进入受支持 profile。
