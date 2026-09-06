# Vflash

面向 RTX 3080 20 GB 和 RTX 4090 48 GB 的原生 **MiniMax H3 推理引擎**。Vflash 使用 PyTorch 与 Triton 实现自己的去噪运行时，支持固定版本的 LightX2V Turbo LoRA。

[文档](https://hansimov.github.io/vflash/zh/) · [开始使用](https://hansimov.github.io/vflash/zh/guide/getting-started) · [版本更新](https://hansimov.github.io/vflash/zh/reference/releases) · [English](README.md)

**0.1.0a5 · 开发者预览版。** 公开接口接收预编译条件包，输出供解码器使用的视频与音频 latent 张量。实际运行需要兼容资源，资源包尚未公开。提示词处理、参考素材上传和 MP4 输出不在本次发布范围内。

## 检查运行环境

需要 Python 3.11 或更新版本。基础安装不会下载模型权重或 PyTorch。

```bash
git clone --branch v0.1.0a5 --depth 1 https://github.com/Hansimov/vflash.git
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
| 单 RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8 | 默认常驻显存，也可选择分块加载 |
| 单 RTX 3080 20 GB | Ref2VA Turbo4 | 从系统内存分块加载 |
| 双 RTX 3080 20 GB | Ref2VA Turbo4 | 共享主机权重，两卡协作执行 |

使用双 3080 时，通过 `--peer-gpu 1` 显式选择第二张卡。默认并行策略为 `sequence-head`，也可选择 `tensor`。引擎不会自动占用其他显卡。

对于已测负载，建议**每个 worker 预留至少 64 GiB 可用系统内存**，并保留额外余量。更大输入需要重新检查容量，当前配置仅覆盖表中的显存版本。详情见[硬件、LoRA 与质量限制](https://hansimov.github.io/vflash/zh/guide/profiles)。

## 接入应用

- [运行条件包](https://hansimov.github.io/vflash/zh/guide/getting-started#run-a-bundle)，使用匹配的编译资源执行推理。
- [通过 Python 接入](https://hansimov.github.io/vflash/zh/guide/python)，在多个请求之间复用模型。
- [启动 Docker 与 HTTP 服务](https://hansimov.github.io/vflash/zh/guide/docker)，使用独立 worker 和有容量限制的任务队列。
- [测量性能与质量](https://hansimov.github.io/vflash/zh/reference/performance)，区分加载、推理和端到端成本。

Turbo4 和 Turbo8 使用蒸馏 LoRA。精确注意力不保证基础模型的质量；不同显卡或并行策略也不保证逐位一致的结果。W8 与 T2VA 尚不是已发布配置。

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
