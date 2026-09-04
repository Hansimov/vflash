# Vflash

面向 **RTX 3080 20 GB** 和 **RTX 4090 48 GB** 的原生 MiniMax H3 推理引擎，支持已适配的 LightX2V Turbo LoRA。Vflash 使用 PyTorch 和 Triton 实现自己的去噪运行时，不依赖 LightX2V 推理框架。

[文档](https://hansimov.github.io/vflash/zh/) · [开始使用](https://hansimov.github.io/vflash/zh/guide/getting-started) · [English](README.md)

> **Alpha 预览版：** 接收预编译条件包，输出视频和音频潜变量（latents，即解码前的张量）。运行需要兼容的资源文件，资源包尚未公开。提示词处理、参考素材上传和 MP4 输出仍在开发中。

## 开始使用

先安装轻量命令行工具，检查硬件和可用配置。这个过程不会下载模型权重或 PyTorch。

```bash
git clone https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

vflash doctor
vflash profiles
vflash plan ref2va-turbo4-exact-sm89 --gpu 0
```

需要 Python 3.11 或更新版本。实际 GPU 推理的准备步骤见[安装指南](https://hansimov.github.io/vflash/zh/guide/getting-started)或 [Docker 与 API 部署](docker/README.zh-CN.md)。

## 支持的配置

| 显卡 | 配置 | 内存策略 |
| --- | --- | --- |
| RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8 | 权重常驻显存 |
| RTX 3080 20 GB | Ref2VA Turbo4 | 从系统内存分块加载 |

已测 3080 负载的进程占用约 60 GiB 系统内存；建议为**每个 worker 至少预留 64 GiB 可用系统内存**，并为其他进程保留额外余量。更大输入需要重新检查容量。目前已检查容量，以及同卡串行加载与传输计算重叠时的结果一致性；更广泛的质量评测和独立参考对照仍待完成。这些配置仅覆盖表中所列显存容量。

Turbo4 和 Turbo8 使用蒸馏 LoRA。精确注意力不代表效果与基础模型等价，也不保证跨 GPU 的结果完全一致。完整范围见[配置与硬件](https://hansimov.github.io/vflash/zh/guide/profiles)。

## 接入应用

- [运行条件包](https://hansimov.github.io/vflash/zh/guide/getting-started#run-a-bundle)，保存 latent 张量。
- [启动 HTTP 服务](https://hansimov.github.io/vflash/zh/guide/docker)，在多个请求之间复用已加载的模型。
- [了解运行资源](https://hansimov.github.io/vflash/zh/reference/runtime-assets)，核对各项版本。
- [测量性能](https://hansimov.github.io/vflash/zh/reference/performance)，区分首次使用与后续请求成本。

## 参与开发

```bash
python -m pip install -e '.[dev,server]'
pre-commit install
pre-commit run --all-files
pytest
```

使用 `npm ci` 和 `npm run docs:build` 构建文档。

## 许可证

Vflash 源码采用 [Apache-2.0](LICENSE)。模型和 LoRA 权重各自遵循相应的许可证和使用条款；本仓库不包含模型权重。

感谢 [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[LightX2V](https://github.com/ModelTC/LightX2V)，以及 PyTorch、Triton 和 CUDA 社区。来源与更多致谢见[许可证与致谢](https://hansimov.github.io/vflash/zh/reference/license)。
