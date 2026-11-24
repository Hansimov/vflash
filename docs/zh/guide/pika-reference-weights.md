# Pika 参考图配方与 QAD 画面权重

运行环境配置见 [Pika H3 Flex](./pika-flex)。按参考输入、时长、权重与设备选择以下准备权重的路径。

## 实验性 Pika 图片参考运行时

`PikaReferenceAVRuntime` 是独立的准备权重候选，固定配置 CLI／server 不会选择它。
它用原始 Ref 适配器执行六次画面前向，用独立 FL 模型生成可选的 Larry 原生声音。
支持 1 MP 内的有序图片参考，保留原始 FP32 图片条件，两模型依次驻留 GPU。默认配方为五秒。
十二秒使用 `PikaReferenceRecipe(duration_seconds=12, audio_steps=10)` 与同长度的 `VideoRequest`。
参考编码、Veda 注意力、FL 音频和解码共享 294 帧模型时钟，以 24 fps 交付 288 帧。
八秒可选择 `PikaReferenceRecipe(duration_seconds=8, audio_steps=10)` 并传入相同时长的请求。192 个模型帧对应 57 个潜帧，按 24 fps 交付 192 帧。

设备空间不足时，八或十二秒音频可将最多 12 GiB 的冻结注意力键值保存在主存。每一步继续更新
文字与音频行，并保留 CUDA 解码余量。音频回执包含显存驻留层数和主存缓存字节数。
已在单张 96 GiB SM120 上实测 720p。其他时长、设备与逐字对白质量仍需单独验证。

`reference_short_edge=None` 保留原始按档位选择的参考几何。显式传入 `1024` 或 `2048`
只改变参考图片编码尺寸，允许放大；输出画布保持请求的尺寸。条件回执记录实际策略、
原图／编码尺寸和参考行数。覆盖参数是待验的质量／性能候选，不代表原始等价或合格默认；
应保留同输入控制，并检查完整成片的身份、细节、动作接触及声音。


实验会话接收冻结的 `PikaReferenceRecipe`，身份同时包含 `ref6`、原生声音和参考编码预算，
例如 `pika-ref6-larry10-r1024-v1`。它不会与仅使用 FL 分区的 `pika6-larry10-v1` 混淆；
参考预算必须与初始化运行时一致。

## 实验性 QAD 画面权重（0.26.0a2）

Python `PikaAVRuntime` 可显式接收 `qad_packed_weights`、`qad_dense_weights` 和
`qad_manifest`。通过 `PikaAVRecipe(visual_weights="nvfp4-qad", audio_steps=10)`
选择固定 FastVideo Pika QAD FL 权重执行六步画面，然后恢复独立保留的原始 FL 模型
进行 Larry 声音采样。普通配方保持原样。这是实验性 SM120 能力；CPU 检查只证明
资源归属和数学调用合同。另外已有六条五秒T2VA／I2VA完整音画在RTX PRO 6000执行，
使用固定SM120后端、原生VAE及独立Larry十步声音。实际前向次数、原模型恢复、
声音KV建立前的画面CUDA副本释放均核对通过。这是执行证据，不是普遍质量验收或
同卡匹配速度基准。人工对照发现壁球动作较弱、锐度略低于原Pika画面路径；保持
实验性选项，应按原始需求比较完整成片后选择速度与质量的取舍。

模型来源为 `FastVideo/Pika-H3-Flex-NVFP4-QAD` 的
`0434eb89466c15ab451cb92fdd20995b4a2d1f1a`。300 个主块线性层和密集文本 refiner
已合入 Pika 适配器，不能重复叠加。准备时必须保留 238 个密集张量描述，并核实全部
106 个投影／时间／输出张量与原始基础模型逐字节相同，才能复用原始六步时间表。
清单合同见 `vflash.adapters.h3_qad.validate_manifest`；模型沿用各原始来源许可证。

在完整流水线依赖之外安装可选 `qad` 依赖。执行采用 FlashInfer 0.6.18 CUTLASS W4A4、
16 值分块及 128×4 缩放布局；在匹配的 CUDA／Torch 环境预编译 SM120 量化和矩阵模块，
或为首次执行提供 CUDA 编译器。不支持的设备或数学路径会明确失败。原始声音模型保持驻留；QAD保留主机侧不可变源张量映射，
只在画面阶段创建CUDA副本，并在建立声音KV缓存之前释放。复制耗时计入完整请求。注意力、原生 VAE 和声音配方保持各自显式选择。
执行回执区分画面 NVFP4 与声音原始 row-FP8，记录完整主块／refiner 调用和权重恢复，
异常清理同样恢复原模型。

在较大画布上编辑较低分辨率原片时，0.28.0a5 预览版可显式使用 `RemakeRequest(…, half_size_source_reference=True)`。模型在源视频的两个空间轴上分别做二倍区域平均，再执行 VAE 编码。参考图及钉住的关键帧沿用各自的正常编码，生成画布、帧时钟与采样步骤保持请求值。宽高须为 64 的倍数，短边至少 960。源掩码依赖全尺寸条件。默认使用全尺寸源条件，新场景需核对商品细节、运动和源画面对应关系。
