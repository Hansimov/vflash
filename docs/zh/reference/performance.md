# 速度、显存与质量

测量应覆盖应用真正关心的过程，并明确计时范围：完整流水线输出 MP4，原生去噪器输出 latent，两者计时边界不同。
当前参考数字见[实测数据](./benchmarks#current)。

## 提速手段 {#levers}

| 手段 | 效果 | 代价或限制 |
| --- | --- | --- |
| 保持流水线常驻 | 首个请求之后不再加载模型 | 每张卡（组）一个进程，请求串行执行 |
| `weight_residency="resident"`（SM90 ≥75 GiB、SM120 ≥90 GiB） | 暖态请求更快，主机内存更低 | 启动更慢；仅 I2VA，且在常驻范围内 |
| Veda 注意力，`veda_dense_edge_layers=3`（44/6） | 50 层中 44 层使用稀疏注意力 | 近似计算；需要混合模型与预测器 |
| FP8 行缩放加载器（SM90/SM120） | 常驻主干更小、线性层更快 | 近似计算；需应用显式回调；默认仍为 BF16 |
| 六步音频修复 | 改善混合 I2VA 的音频 | 增加六次音频评估（如 RTX 4090 上 1024²/5 秒约 55 秒） |
| 隐藏上下文音频复用 | 实测案例中音频耗时约减少 34% | 近似后续非音频状态；按请求显式启用 |
| 双卡处理一个请求 | 降低该请求延迟 | 消耗更多 GPU 秒；有两个待处理请求时，两个单卡 worker 完成更多工作 |

每次只改变一个手段，并固定提示词、输入、种子、画布和时长，采用前先检查输出（见[质量](#quality)）。

## 区分首次加载与后续请求 {#timing}

每次命令行调用都会初始化模型。常驻 `H3Pipeline` 或原生 HTTP worker 可以跨请求复用模型，因此启动、首次使用和后续请求的成本不同。

完整生成应单列 `H3Pipeline.initialization_seconds` 和 `VideoResult.elapsed_seconds`。后者覆盖输入准备、编码、去噪、媒体输出及请求清理。`stages` 同时包含外层时长和嵌套明细，不应重复相加，详见[链路计时合同](../guide/complete-pipeline)。

原生去噪结果的 `session` 包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `request_index` | 当前模型会话中的请求序号，从 1 开始 |
| `initialization_seconds` | 该会话的初始化耗时 |
| `initialization_charged_seconds` | 首次请求计入初始化耗时，后续请求为零 |
| `request_wall_seconds` | 在已初始化会话中执行本次请求的耗时，包含写入 latent 输出文件 |

估算引擎侧首次请求成本时，可将 `initialization_charged_seconds` 与 `request_wall_seconds` 相加。排队、进程启动、HTTP 传输、输入编码和视频解码不在这两个字段的合计范围内；如果应用包含这些环节，应分别测量。

首个任务之前，`/readyz` 通过只表示配置文件和显卡检查通过，不代表模型已经加载。

## 剖析单次去噪请求 {#denoise-profile}

`vflash generate` 与 `vflash denoise` 均支持 `--profile-denoise`。该开关刻意保持为显式诊断选项：它会为每个 evaluation 和每个流式 block 记录 CUDA timing event，因此最终吞吐比较应使用未开启诊断的独立运行。诊断本身不增加 evaluation 围栏；完整 pipeline 原本就会在发布每次进度前等待所有协作设备，报告会把这些既有围栏单列。

返回的 `generation.denoise_profile` 会区分：

- 主卡的行时间步准备、输入打包、invocation 准备、denoiser、final layer 与 latent 更新；
- 每个 rank 的 H2D 活跃时间、计算流等待权重时间、block 执行以及复制/计算跨度；
- 每个 evaluation 与 rank 汇总的 AdaLN、attention norm/modulation、QKV projection、Q/K norm 与
  rotary、attention、attention output、FFN norm/modulation、FFN input 和 FFN output；
- 输出投影与 gate/residual、FFN 输入投影与 SiLU-times-gate 的 block 内部明细；adapter 融合和
  tensor-parallel 实现保留各自明确的实现标签；
- sequence/head collective 的调用次数、传输字节、主机提交时间和主机 `Work.wait()` 时间；
- sequence/head attention 在计算流上的 QKV pack、入站依赖等待、QKV unpack、Flash-SDPA、
  attention 结果 pack、出站依赖等待和最终 unpack；
- 每个 evaluation 的引擎提交、既有进度围栏、回调、最终围栏和 profile 汇总耗时。

H2D、ready wait、block compute 与 rank span 描述的是相互重叠的 CUDA stream，**不能相加**。block compute 包含 collective 临界路径；主机 collective wait 只是提交/同步诊断，不能单独当作 GPU 通信耗时。需要按 kernel 归因 NCCL 时仍应使用外部 CUDA profiler。诊断输出只含设备序号、时间和字节计数，不包含模型张量、提示词、路径或 GPU UUID。

sequence/head 的 `inbound_ready_wait` 与 `outbound_ready_wait` 只测量实际阻塞计算流的依赖，
不代表 NCCL kernel 的完整生命周期；后者可能在另一条 stream 上与 QKV pack 或 Flash-SDPA 重叠。
这些 attention 子阶段已经包含在外层 `attention` 中，不能再与 attention 或 block 执行重复相加。
同样，`block_detail_seconds` 已经包含在 `block_phase_seconds` 内，只用于估算融合收益上限，不能再与
外层阶段合计。

## 让比较有意义 {#comparisons}

使用相同的条件包、模型与 LoRA 版本、运行配置、显卡和输出边界。分别报告首次使用与后续请求耗时，并重复足够次数以观察波动。

比较内存时，区分 GPU 已分配内存、GPU 预留内存、整张显卡占用和系统 RAM。尤其是 3080 的分块加载权重会占用系统内存，这部分不会出现在只统计显存的图表中。

从 Turbo8 切换到 Turbo4 时，变化的不仅是计算量，也包括蒸馏调度方式。报告速度比较时，应同时说明这个差别。

## 双卡执行 {#parallel}

双卡协作可以缩短单个请求的等待时间，独立单卡 worker 则服务于不同的吞吐需求。使用相同主卡、输入和计时边界评估两者，别把双卡的一次请求与两张卡的总吞吐混为一谈。

已公开的 [a3 双 3080 对照](./benchmarks#sm86-parallel)在一个固定负载上测得 1.725× 加速。它保留原始软件版本、热状态和质量限制，不代表当前版本或其他负载的速度保证。

0.4.0 还有一个受控的 SM89 Base16 结果：一个固定的十秒、736 × 992 L2VA 请求在前后两个单卡对照中分别耗时 774.153 和 773.515 秒，双卡候选为 484.232 秒。相对 773.834 秒的单卡均值，双卡延迟降低 37.4%。三个 MP4 输出字节一致。双卡运行中两张 450 W 显卡均不高于 78°C，没有热降频样本；后一个单卡对照的十分钟滚动最高均温为 65.077°C。双卡串行执行两个请求将消耗 968.464 设备秒，而两个独立单卡请求的平均 makespan 为 773.834 秒，fleet 代价为 25.2%。因此该证据只资格化“同类显卡原本空闲”时的可选低延迟路径，而不是默认吞吐策略。

### 精确直接重排的边界 {#direct-relayout}

0.4.0 删除了双卡 `sequence-head` 路径中的多次布局物化。最终未开启 profile 的 A/直接重排/A
实测中，双 SM86 的五秒 512² I2VA 请求为 **174.776 → 172.149 秒（1.503%）**，双 SM89 的
十秒 736 × 992 L2VA 请求为 **474.669 → 472.429 秒（0.472%）**；去噪阶段分别降低
1.995% 和 0.623%。

另一次开启归因的 SM89 A/直接重排/A 中，完整请求降低 0.662%，去噪降低 0.801%。800 个被记录的
block 内，每个 rank 的 QKV 打包约从 4.51 降到 2.13 秒，返回 head 合并约从 1.38 降到
0.67 秒；Flash-SDPA 仍约为每 rank 223–235 秒且没有变化，说明布局内核不是主瓶颈。

在代表性的 55,413-token shape 上，直接 QKV 打包在 SM86、SM89 上都少分配 1,136.4 MiB 瞬时
显存，返回 head 合并少分配 378.8 MiB，各自的操作峰值均降低 50%；二者发生在不同执行点，不能相加
成完整 pipeline 显存。所有算子输出逐元素一致，完整 A/B/A 的视频 elementary stream 也一致；官方
音频重复解码在控制和候选中仍会波动。本轮没有对大模型或媒体文件做哈希。详见
[Sol-Engine 对齐矩阵](./sol-engine-alignment)。

## 注意力激活的生命周期

普通原生block分别展开注意力和FFN调制，在进入FFN前释放已无用途的QKV、归一化和注意力张量。
GEMM形状、采样和权重不变；可选分阶段profiler仍保持原测量路径。

单张48GiB SM89上的2048×1152、五秒、原始v0.1 BF16 hybrid/Veda探索性配对控制中，峰值allocated
从34,914,620,416降至24,200,463,872字节（30.69%）。去噪221.633对223.787秒，证明内存收益，
不声称提速。120张解码RGB帧逐像素一致；音频解码非逐位一致，与此前解码器波动一致，但未确立
音频内容等价。这是单配对案例，不是通用画质保证或扩大已发布画布边界。CPU float32/BF16检查
覆盖block输出和进入FFN时的临时对象归属。

## 检查输出质量 {#quality}

更快的结果仍需满足实际任务。请根据用户原始要求与参考素材，检查指令遵循、主体与细节一致性、运动、画面瑕疵和音画关系。即使候选结果接近基线，基线本身也可能没有满足要求。

解码后按时间查看多帧，并以正常速度观看视频、试听音频。静帧无法证明运动自然，音频波形也不能证明语义正确。分别记录已确认的发现、无法判断的维度、样本不足和评审分歧。张量误差仅用于排查实现差异，不能用作生成质量分数。

精确注意力不代表蒸馏 LoRA 与基础模型效果等价，也不保证不同 GPU 架构上的浮点结果完全一致。每个配置当前完成的检查范围见[支持说明](../guide/profiles#lora)。

完整链路已支持[列出的配置](./pipeline-profiles)。集成检查证明已测请求能够正确完成，不代表通用的提示词到 MP4 速度或质量保证。

## 扩展hybrid交付（0.6.12） {#extended-hybrid}

五条全新条件完整请求在单张RTX 4090 48 GB／450 W串行完成，原始v0.1 BF16 hybrid、四次计算、显式Veda。首条含冷加载，后续复用pipeline；表中是观察到的完整请求延迟，不是隔离提速对照。实际尺寸/帧数及32kHz立体声音视频完整解码通过。主存上限260GiB不是最低需求测量。显存数值为去噪/媒体阶段记录的最高allocated，并非进程总显存或最低设备容量保证。

| 模式 | 图像数 | 画布 | 秒 | 完整请求秒 | 已记录阶段峰值GiB |
| --- | --- | --- | --- | --- | --- |
| I2VA | 1 | 1024×1024 | 15 | 484.13 | 27.74 |
| I2VA HD | 1 | 1440×1440 | 10 | 609.89 | 36.53 |
| FL2VA | 2 | 1344×768 | 15 | 405.50 | 27.81 |
| Ref2VA | 1 | 960×544 | 15 | 172.07 | 14.83 |
| Ref2VA | 9 | 960×544 | 5 | 93.15 | 9.44 |

容量证据与画质分开：多时间点检查仍发现部分场景的机位/构图漂移及细节变化，复杂参考与动作遵循尚非全面合格。音频有限且可解码，AAC存在不证明静音或声音语义；数字静音请显式使用silent交付。没有H100、普通24GiB4090或任意4MP/15秒资格；入口执行组合资源约束。集成来源：`90e19b3b`。


## 可选的有界RGB流式编码 {#media-streaming}

源码构建支持 `H3Pipeline(..., media_video_input="pipe")` 或
`vflash generate --media-video-input pipe`；底层 `encode_mp4` 使用 `video_input="pipe"`。
默认仍为 `file`。两者采用相同的八帧量化、H.264/AAC、音频策略、精确时间锚点和原子非覆盖发布。
流式省去RGB临时文件：2048²的120帧为1,509,949,440字节；不会省去已解码CPU张量或扩大GPU请求容量。

同卡、已初始化、四步hybrid/Veda、2048²五秒I2VA完整请求对照：

| GPU／整机范围 | 文件A1 | 流式B | 文件A2 | 完整请求结论 |
| --- | ---: | ---: | ---: | --- |
| H100 SXM，单个远端宿主 | 282.452秒 | 294.185秒 | 283.594秒 | 流式比原路径均值慢3.94% |
| RTX PRO 6000 Server，另一个远端宿主 | 498.368秒 | 428.771秒 | 359.039秒 | 处于对照波动内，无已证实提速 |

每组三片字节相同，仅证明受测编码等价，不代表通用画质或提示遵循达标。PRO原路径存在明显内存等待漂移，
不能只比较第一次原路径并宣称约14%提速。按实际工作负载的CPU/I/O、临时盘收益选择显式选项，
H100 SXM保留file。表中不包含资源申请、模型加载、传输和排队。流式超时覆盖喂入与FFmpeg执行，
失败会终止并回收子进程、清理临时输出。


### 显式CPU编码线程预算（源码构建）

`vflash generate --media-video-threads 16`、`H3Pipeline(media_video_threads=16)`及
`encode_mp4(..., video_threads=16)`可指定H.264编码线程数；默认None保留FFmpeg auto。
合法显式范围1–256，该值不限制所有滤镜/音频线程。它既可配合file，也可配合pipe输入。
线程配置可能改变有损码流，不承诺逐字节一致。

固定2048²/120帧RGB、相同libx264 medium/CRF18、8 CPU配额/8GiB内存的CPU对照中，
auto为7.127/7.432秒，16线程6.648秒，8线程9.847秒。16线程比两侧均值快8.68%，
匿名峰值内存减少50.54%；三者不是GPU完整请求或共享宿主压力复现。
默认与16线程的全帧luma SSIM均为0.994817（相对同一解码输入），音轨解码一致；
码流不同，稀疏视觉抽查及该压缩指标不代表模型画质资格。保留此显式CPU选项，
不依据它修改GPU默认、宣称远端超时已修复或自动按CPU配额选线程数。

[FFmpeg codec options](https://ffmpeg.org/ffmpeg-codecs.html#Codec-Options).


## 显式缓冲读取与无损压缩权重

`vflash[weight-io]` 提供可选的 Zstandard codec。在已安装 Torch 的运行环境中，
`SequentialWeights` 将不可变 safetensors 读入独占的 CPU 分片缓冲区，支持
1/2/4/8 个有界读取线程。离开读取作用域后，仍被使用的张量会保留整个分片缓冲区。
其他路径仍默认映射读取；不会自动改变精度、显卡配置或公共 pipeline 准入。

```python
from pathlib import Path
from vflash.native.h3_weight_codec import prepare
from vflash.native.h3_buffered_weights import SequentialWeights

source = Path("weights.safetensors")
sidecar = Path("prepared-sidecar")  # 新目录，保留原文件
prepare(source, sidecar)             # 每个固定源只需准备一次，可在 CPU 上完成
with SequentialWeights(source, read_workers=4, compressed_directory=sidecar) as weights:
    tensor = weights.get_tensor("weight")
# tensor 在作用域结束后继续持有 CPU 存储
```

调用者必须使用可信产物，并在加载前将源和 sidecar 绑定到同一准备版本；布局和文件
不变性检查不等于密码学认证。准备时逐块比较解压字节与原文，不计算大模型哈希。
清单最后发布，未完成的准备不能被消费。原文件用于读取头部和回滚，不能删除。
额外存储与 CPU 准备时间均应计入完整成本。

单台 H20/150GiB 慢盘实测中，native 无损压缩由 20.73→17.08GB，A/B/A 初始化
快 14.25%，初始化加首片快 5.54%。固定其他优化后，文本由 29.19→23.88GB 的
两组配对完整视频中，初始化加首片 286.93→270.07 秒（快 5.88%），实际登记
301.02→287.56 秒；暖态第二片 90.49/90.62 秒。完整解码视频与 PCM 逐样本一致，
只证明这些复制型改动的等价，不代表通用语义画质接受、物理存储空缓存或跨宿主均值。
匿名内存峰值 7.89→37.38GB、cgroup 总峰值 65.10→90.86GB，容量代价不可忽略。
本地快速页缓存下压缩反而更慢，须按实际机器的完整启动、交付和成本显式选择。

通用核心源自已验证应用版本 `0ae285da`，补充现有头部验证及失败读取的缓冲区释放。
本版本不公开应用 FP8 runner，也不会自动升级运行中的服务。

## 显式近似 FP8 线性算子

0.6.29 提供 `vflash.native.h3_fp8_linear.RowFP8Execution` 和 `RowFP8Linear`，
消费 `[out, in]` 的 E4M3 编码与 `[out, 1]` 的 FP32 逐行尺度。这是数值算子接口，
**不是**完整 FP8 资产加载器、流水线精度开关，也不改变现有 BF16 profile。
Torch 来自 GPU 可选依赖；仅选择融合量化核时才导入 Triton。

```python
import torch
from vflash.native.h3_fp8_linear import RowFP8Execution

codes = torch.zeros((16, 32), dtype=torch.float8_e4m3fn)
scales = torch.ones((16, 1), dtype=torch.float32)
values = torch.ones((2, 32), dtype=torch.bfloat16)
reference = RowFP8Execution("weight-only").linear(values, codes, scales)
# 仅用于已明确准入的 SM90/SM120 CUDA 张量：
execution = RowFP8Execution("w8a8", activation_quantizer="triton", fast_accum=False)
# result = execution.linear(values.cuda(), codes.cuda(), scales.cuda())
```

weight-only 恢复近似权重后使用普通 GEMM；W8A8 还会逐行量化激活，调用 Torch scaled
GEMM 并输出 BF16。Torch 和融合 Triton 量化均须显式选择，硬件不支持时失败，不偷偷回退
精度。计数与可选异步测量作用域属于各自执行对象；算子不修改全局类、不读取模型、不选择设备，
不插入 GPU 同步栅栏。`RowFP8Linear` 的权重/尺度初始位于 meta，保留原 bias；可信加载器
必须在执行前赋齐张量。不可变资产、有限的预制尺度、转换和完整流水线准入仍由调用方负责。

融合核保留“先拓宽到 64 位再乘步长”的地址计算，覆盖物理行跨度跨过 2³¹ 元素的 FFN 半幅视图。
实现来自已实测应用源码 `3bbaa52d`；私有 H100、RTX PRO 6000、H20 集成曾执行这些计算，
H20 九例包含四模式、0.25MP/五秒到4MP/十五秒。本次提取完成 CPU 数值/所有权检查与
SM90/SM120 离线编译，**没有**用新包重跑上述全流水线。不据此宣称普遍提速、等同基线的时序
画质、小显存容量或公开量化流水线资格。FP8 仍是近似计算，提取后计算一致不消除既有画质未知项。
生产默认不变。

### 每条流水线独立加载文本编码器

`H3Pipeline(..., text_encoder_loader=loader)` 接受显式可信 Python callable：输入为
已准备模型根目录 `Path`，返回匹配固定架构、位于 CPU 的
`Qwen3VLForConditionalGeneration`。编码器仍由对应 conditioner 裁剪所需前缀、安装
卸载机制并回收；仍到 `prepare()` 或首个请求才加载。省略该选项时保留原本固定的离线
BF16 加载。不把字符串或模块路径解析成可执行代码，也不提供 HTTP/CLI 插件入口。

加载器可绑定独立准备的 FP8 资产与 `RowFP8Execution`，无需替换 Transformers 全局
`from_pretrained`。调用方仍须保证不可变资产身份和精度资格；此接口本身不会把 BF16 收据
变成 FP8 资产。native FP8 装载及完整量化流水线准入仍是独立工作。
CPU 测试覆盖独立所有权和异常隔离。微型51层固定 Qwen 实例使用372个真实 FP8 Linear，
通过真实 H3 的文本、单图、多图、视频条件 helper，四项输出与恢复权重对照逐值一致。
这是接口和数值证据，不是完整模型、GPU 或视频画质资格。构建此类对照时应保留旋转位置
编码的非持久 FP32 缓冲区；把整个模块转成 BF16 也会改变这些缓冲区。


可信 Python 集成可通过 `native_denoiser_loader` 将独立去噪器工厂从 `H3Pipeline`
传给 `NativeEngineSession` 及其 native runtime。工厂接收已验证的 artifact，以及
设备、AdaLN 加载器、注意力和融合设置等常驻加载参数，返回生命周期兼容的去噪器。
必须显式使用 `weight_residency="resident"` 和单设备执行，原有容量与模型限制保持。
这不是可序列化插件名称，也不提供 CLI/HTTP 入口。常驻块栈的 `load` 另接受
`block_loader`，块子类可覆盖 `_resident_weight`，无需修改模块全局对象。
默认路径仍严格搬运 BF16；这些所有权接口不替代其他资产的身份验证、精度声明、性能或画质资格。
