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
| Sol 稀疏注意力 | 0.5.0的`auto`选择`sol-sm89` | **仅单SM89官方Base16默认启用。** 仍是近似，显式dense可选；真实CuTe执行和完整媒体不等于质量资格。 |
| TeaCache / FirstBlockCache | 上游 4090 结果为 49 次 DiT forward，复用其中 35 次 | **暂缓。** Vflash Base16 只有 16 次评估，收益空间与质量风险不同。 |
| INT8 QKV 与 FP8 输出传输 | 八张 B300 快速配置的默认项 | **不进入精确默认。** 有损传输和不同拓扑需要独立 profile 与质量门。 |
| MXFP8 融合线性层 | SM100 系列路径；上游报告了明显输出差异 | **不在范围内。** 不面向 SM86/SM89，也不能继承 exact 名称。 |
| 音视频 VAE 并行和 tile sharding | 围绕八 rank Sol-H3 decoder 生命周期设计 | **暂缓。** Vflash 双卡在去噪后由主卡持有媒体阶段；应先改进生命周期，再证明完整媒体等价。 |
| 四步 FastH3 adapter | 主要对照把 49 次 forward 改成四次 | **不算运行时优化。** adapter 质量和模型语义需要独立 profile。 |
| H3 草稿 → LTX 精修与 TAEH 解码 | 单独的粗到细产品，同时改变模型、adapter、解码器、分辨率链和采样调度 | **不进入当前 exact profile。** 只有另建具名链路并通过质量门后才可采用，不能写成 SM86/SM89 运行时提速。 |

## 精确与近似是两类产品

dense执行使用PyTorch Flash SDPA和精确BF16传输。这里的“精确”表示所选实现
保持声明的算术与注意力合同；它不表示不同 GPU 架构、不同并行拆分或蒸馏 adapter 必须输出相同张量。

### 0.5.0的SM89 Sol默认 {#sol-default}

共享`auto`默认在单SM89官方Base16选择近似`sol-sm89`，其他配置为`torch-flash`。
覆盖`H3Pipeline`、`NativeEngineSession`、`generate`、`denoise`及HTTP的`VFLASH_ATTENTION_BACKEND=auto`。
用`attention_backend="torch-flash"`、`--attention-backend torch-flash`或对应HTTP环境值显式保留dense。
Sol需要串行block-ring；原生会话的`default`在Sol下解析为block-ring。显式选择不支持的Sol会报错，
不会在SM86、双卡或Turbo上启用。prepared模型/调度身份不变，结果报告实际非精确策略。
0.4.0和旧预构建镜像不含该能力。

适配器调用[NVIDIA Sol-Attn](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/sparse/sol_attn/)，
固定源码`d0c0a4685ab5dc2336d18b7213d85f13def92418`、版本0.5.0。
实际后端必须为`cute_sm89`；依赖缺失或布局不兼容会报错，不回退dense。
BF16 BTHD Q/K/V的head dimension为128；固定`tau=0`、对角阈值及文字/条件图像/音频KV sink。
前缀query另跑dense Torch Flash并覆盖对应行，仅目标视频query保留Sol输出。QKV物化及前缀attention
全部计入请求耗时。

该保护仅针对这些行的attention计算，**不保证端到端音频或身份等价**：后续层仍读取已经变化的隐藏状态，
视频轨迹和声音都可能改变。执行metadata报告`exact=false`、真实后端、保护前缀长度和调用次数；
首个请求前只报告零次调用及空执行后端，不能把配置声明当作执行证据。

在0.5.0源码checkout中构建标准完整镜像：

```bash
docker build --target pipeline -t vflash:0.5.0-pipeline .
```

该target基于原Torch 2.11/cu130 pipeline，固定CUTLASS DSL 4.5.0、cuda-python 13.2.0及
TVM-FFI 0.1.11，并在构建时对上游四处interface调用应用已测的位置stream参数ABI补丁。
普通`runtime`和`pipeline`都包含Sol，移除独立实验overlay。Python安装GPU/pipeline依赖后，在同一
环境运行`python -m vflash.install_sol`（需要Git及网络，不下载模型）。模型仍外部挂载，复用既有
prepared assets及完整pipeline指南的挂载方式即可。
首用JIT时延不能外推为热态吞吐，MP4成功也不等于同质量成立。

### 完整请求探索性对照

源码main的adapter已在单张RTX 4090 48 GB（SM89）、450 W功率限制下实跑；同一个持久化
Torch 2.11/cu130 pipeline使用官方Base16、BF16、16次评估及`exact-v1`端点交付。
每个场景冻结提示、参考和seed。初始化72.311秒单列，服务排队/传输不在以下边界内。
**本轮启用了去噪profile**：属于完整本地MP4请求的归因测量，不是无profile的服务吞吐资格。

| 场景 / 执行顺序 | Dense A | Sol | Dense A2 |
| --- | ---: | ---: | ---: |
| 5秒 · 512×672 · I2VA · 完整请求 | 122.910秒 | 131.464秒 | 未跑 |
| 同一小场景 · 去噪阶段 | 94.408秒 | 107.723秒 | 未跑 |
| 10秒 · 736×992 · L2VA · 完整请求 | 767.342秒 | 617.283秒 | 767.883秒 |
| 同一长场景 · 去噪阶段 | 715.365秒 | 564.808秒 | 715.938秒 |

以长场景两次control均值为分母，完整请求缩短19.584%（1.244×），去噪缩短21.078%；
control漂移0.070%。小场景Sol包含首次编译，其独立成本未分离，不能据此认定热态小场景收益或回退。
两条Sol均记录800次实际`cute_sm89`调用，同时执行保护前缀的dense计算。
小场景两臂的去噪allocated峰值均为5,421,633,536字节，长场景三臂均为17,186,832,384字节；
设备采样总峰值18,359 MiB。忙时滚动十分钟最高均温65.119°C、瞬时峰值67°C，热降频标志样本为零。

五条MP4全部完整解码，120/240帧、音频有限值、零黑帧。每条12个时间点的可见主体与端点关系保持，
但Sol改变了中间进展和音频波形。端点PSNR约33.14 dB（I2VA）及31.28 dB（L2VA），
参考为此次直接引擎合同的Lanczos resize，不是应用层另行规定的裁切；端点指标不代表运动帧合格。
全速动态、声音内容和实际试听仍未关闭，尤其小场景两臂声音能量都较低，不能记为质量通过。
**同质量和合格视频吞吐尚未成立。** 0.5.0将有界Sol路径作为单SM89 Base16默认是明确的版本决策，
不是同质量认证。两场景不公开私有参考/提示，仅作为聚合工程
证据，不能冒充可公开复现的完整benchmark套件。

近似方法必须具名、可测量且可移除，默认变更必须明确披露。它们需要固定的提示词/参考/seed套件、
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
