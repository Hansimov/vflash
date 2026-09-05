# 运行资源

Vflash 当前从已编译的输入开始执行。安装 Python 包不会自动准备模型权重、编码提示词或下载示例条件包。面向用户的资源准备命令和公开运行资源包尚未提供。

这个预览版适合已经持有兼容编译资源的开发者。

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

Vflash 会检查资源声明的元数据。文件名正确并不代表兼容；重命名目录或修改清单不能转换不匹配的权重。

条件采集显卡属于来源记录：SM86 与 SM89 采集的条件，在模型身份、编码器版本、计算配置和张量布局匹配时，可以使用同一目标的模型资源。这不承诺跨卡条件张量一致；去噪权重资源仍须匹配实际执行目标。

## 存放输入与输出 {#storage}

把模型资源和条件包保存在源码目录之外。使用 Docker 时，将它们以只读方式挂载，并为服务准备单独的可写输出目录。具体设置见 [Docker 部署](../guide/docker)。

输出为 safetensors 文件，包含视频和音频 latent 张量。它需要交给兼容的解码器，而解码器尚未包含在当前公开运行时中。

模型和 LoRA 文件各自遵循相应的[许可证及使用条款](./license)，与 Vflash 源码许可证分开。
