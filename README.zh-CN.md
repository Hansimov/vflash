# Vflash

面向 RTX 3080 20 GB 和 RTX 4090 48 GB 的原生 **MiniMax H3 推理引擎**。Vflash 使用 PyTorch 与 Triton 实现自己的去噪运行时，支持固定版本的 LightX2V Turbo LoRA。

[文档](https://hansimov.github.io/vflash/zh/) · [开始使用](https://hansimov.github.io/vflash/zh/guide/getting-started) · [版本更新](https://hansimov.github.io/vflash/zh/reference/releases) · [English](README.md)

**0.2.1 正式版。** 支持纯文字生成，也支持提示词加一至三张有序参考图。通过 Python 或容器命令行的[完整链路](https://hansimov.github.io/vflash/zh/guide/complete-pipeline)生成五秒 MP4：Ref4 支持单张 RTX 4090 48 GB 或双张 RTX 3080 20 GB，纯文字 T2VA 使用单张 4090；运行资产可以[从固定官方权重直接编译](https://hansimov.github.io/vflash/zh/guide/compile-weights)。原生 Python、CLI 和 HTTP 接口也支持在下列显卡上接收条件包、输出音视频 latent。

## 检查运行环境

需要 Python 3.11 或更新版本。基础安装不会下载模型权重或 PyTorch。

```bash
git clone --branch v0.2.1 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

vflash doctor
vflash profiles
vflash plan ref2va-turbo4-exact-sm89 --gpu 0
```

| 显卡配置 | 已发布配置 | 权重放置 |
| --- | --- | --- |
| 单 RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8；T2VA Turbo4 | 默认常驻显存，也可选择分块加载 |
| 单 RTX 3080 20 GB | Ref2VA Turbo4 | 从系统内存分块加载 |
| 双 RTX 3080 20 GB | Ref2VA Turbo4 | 共享主机权重，两卡协作执行 |

使用双 3080 时，通过 `--peer-gpu 1` 显式选择第二张卡。默认并行策略为 `sequence-head`，也可选择 `tensor`。引擎不会自动占用其他显卡。

对于原生去噪器的已测负载，建议**每个 worker 预留至少 64 GiB 可用系统内存**，并保留额外余量。完整链路还需在主机上保存编码器和解码器，需要更大的内存预算。更大输入需要重新检查容量，详情见[硬件、LoRA 与质量限制](https://hansimov.github.io/vflash/zh/guide/profiles)。

## 接入应用

- [生成 MP4](https://hansimov.github.io/vflash/zh/guide/complete-pipeline)，使用纯文字或一至三张参考图，并在连续请求间复用模型。
- [编译官方权重](https://hansimov.github.io/vflash/zh/guide/compile-weights)，从固定版本的模型与 LoRA 创建 Ref4 或 Base4 运行资源。
- [运行条件包](https://hansimov.github.io/vflash/zh/guide/getting-started#run-a-bundle)，使用匹配的编译资源执行推理。
- [通过 Python 接入](https://hansimov.github.io/vflash/zh/guide/python)，在多个请求之间复用模型。
- [启动 Docker 与 HTTP 服务](https://hansimov.github.io/vflash/zh/guide/docker)，使用独立 worker 和有容量限制的任务队列。
- [测量性能与质量](https://hansimov.github.io/vflash/zh/reference/performance)，区分加载、推理和端到端成本。

Turbo4 和 Turbo8 使用蒸馏 LoRA。精确注意力不保证基础模型的质量；不同显卡或并行策略也不保证逐位一致的结果。完整生成使用固定的 Ref4 或 T2VA Base4 v1.0 资产和会话。双 SM86 的 Ref4 必须使用 `sequence-head`；单 SM86 与 Turbo8 使用原生条件包接口。W8 以及任意适配器或模式切换不在支持范围内。请查看[验证范围](https://hansimov.github.io/vflash/zh/guide/profiles)。

## 参与开发

```bash
python -m pip install -e '.[dev,server]'
pre-commit install
pre-commit run --all-files
pytest
```

使用 `npm ci` 和 `npm run docs:build` 构建双语文档。代码职责见[贡献者地图](AGENTS.md)。

## 许可证

Vflash 源码使用 [Apache-2.0](LICENSE)。模型和 LoRA 各自遵循相应许可证和使用条款，本仓库不含模型权重。运行时不依赖 LightX2V 推理框架。

感谢 [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[LightX2V](https://github.com/ModelTC/LightX2V)，以及 PyTorch、Triton 和 CUDA 社区。[来源与致谢](https://hansimov.github.io/vflash/zh/reference/license)。
