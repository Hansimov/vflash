# 运行资源

latent 命令行从编译后的输入开始执行。SM89 的 Ref4 和 T2VA Base4 还提供[完整 Python 视频链路](../guide/complete-pipeline)和[官方权重编译器](../guide/compile-weights)。安装 Python 包不会自动下载权重或示例条件包。

下列命令行合同面向已经提供兼容编译资源的开发者。

## 四类输入 {#inputs}

| 输入 | 内容 | 命令行参数 |
| --- | --- | --- |
| 权重包 | `artifact.json` 和已编译的 Transformer 分块权重 | `--artifact` |
| 调度包 | `overlay.json`、`schedule.safetensors` 及匹配的调度数据 | `--schedule-overlay` |
| 辅助张量 | 一个 `.safetensors` 文件，包含投影层等 Transformer 分块之外的张量 | `--auxiliary-tensor` |
| 条件包 | `bundle.json`、`conditioning.safetensors` 和该条件包声明的其他文件 | `--bundle` |

前三类输入对应固定模型和运行配置。每个条件包对应一次请求，保存已编码的条件信息、初始 latent 状态等输入张量。

将它们分开后，服务可以只加载一次模型，再连续处理多个条件包。

## 版本必须匹配 {#versions}

使用为所选 GPU、模型版本、LoRA 版本和调度方式编译的资源。Turbo4 与 Turbo8 需要不同的调度数据；SM86 与 SM89 也需要各自对应的权重包。

当前配置固定使用以下来源版本：

| 来源 | 版本 |
| --- | --- |
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| [LightX2V Turbo4](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/83b617309219e859c1c264520eba07492d22e958) | `83b617309219e859c1c264520eba07492d22e958` |
| [LightX2V Turbo8](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/0eebcc7e79f9cb200927c80b8e7595265b770e34) | `0eebcc7e79f9cb200927c80b8e7595265b770e34` |
| [LightX2V Base4 v1.0](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/ec01fa4c86263832faa0bd1d6d8f36a281eaabb2) | `ec01fa4c86263832faa0bd1d6d8f36a281eaabb2` |

Vflash 会检查资源声明的元数据。文件名正确并不代表兼容；重命名目录或修改清单不能转换不匹配的权重。

条件采集显卡属于来源记录：SM86 与 SM89 采集的条件，在模型身份、编码器版本、计算配置和张量布局匹配时，可以使用同一目标的模型资源。这不承诺跨卡条件张量一致；去噪权重资源仍须匹配实际执行目标。

## 存放输入与输出 {#storage}

把模型资源和条件包保存在源码目录之外。使用 Docker 时，将它们以只读方式挂载，并为服务准备单独的可写输出目录。具体设置见 [Docker 部署](../guide/docker)。

latent 命令行输出包含音视频张量的 safetensors 文件；完整 Python 链路在内部处理编码、官方 VAE 解码，再发布 MP4。

模型和 LoRA 文件各自遵循相应的[许可证及使用条款](./license)，与 Vflash 源码许可证分开。

## LoRA 文件核对 {#adapter-files}

Turbo4 使用 `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`，Turbo8 使用 `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`。这两个文件的摘要与 2026-09-05 核对的上游修订 `2f015e66b37c585cea9dc4ae6f1850ea8788e742` 一致。这是一次固定来源核对，不保证未来修订自动兼容。

T2VA 配置使用独立的 Base4 v1.0 文件及上表对应修订。SHA-256 为 `1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c`，alpha 128 / rank 128。不能与 Ref4 或更新的 Base4 版本互换。

## 视频条件包 {#video-conditioning}

视频 adapter 使用 **schema 2 条件包**，实际输入是 `VideoReference`，编码后的提示词使用 `<Video 1>`。已有图片和 T2VA 条件包保留 schema 1；不会把视频帧改名为多张参考图片。

原生读取接口限定单段 2–5 秒视频，归一化为 24 fps，使用单张 SM89 显卡的 Ref4。输出为 5 秒、24 fps，总像素不超过 928×512。源音轨、图视频混合、Ref8 和 SM86 视频参考不在此范围内。这是依据参考重新生成，不保证精确编辑或保留每一源帧。

adapter 必须记录源文件及解码 RGB 的摘要、源尺寸、归一化帧数、实际官方画布、VAE 输入与 latent 帧数及条件视频行数。`official-video-cfr24-v1` 规则要求先按源显示尺寸解码，再由固定官方 setup 缩放一次。最终画布长边不超过 1376、面积不超过 1376×768，条件视频行不超过 33,024。VAE 只消费完整时序块，文本编码器则从整段归一化视频采样，两者不能混称。

`H3ConditioningCaptureSession.finish(..., schema_version=2)` 将这些元数据与真实捕获的 14 个张量绑定。读取时检查来源、精确 Ref4 调度、张量宽度、模态索引和时序前缀；修改 schema 不能把图片捕获转换成视频条件。0.3.1 的 `H3Pipeline` 接受本地 `VideoRequest(reference_video=Path(...))`，通过官方视频 setup 完成捕获，见[输入限制与示例](../guide/complete-pipeline#reference-video)。实际安装的公开流水线已完成图片、视频、图片的连续请求，并核验 14 个条件张量、最终 FP32 音视频 latent、全部交付 RGB 帧及取消清理。[发布说明](./releases#v0-3-0)区分实现验证、内容质量与音频重复性。
