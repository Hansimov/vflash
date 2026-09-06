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
| RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8 | 权重常驻，或显式使用分块加载为中间结果留出显存 |
| 双 RTX 3080 20 GB | Ref2VA Turbo4 | 共享系统内存权重，按序列和注意力头协作 |

3080 服务建议先采用**双卡 `sequence-head`**，优先降低一次请求的等待时间；单卡执行仍可显式选择。第二张卡需要由调用方指定，Vflash 不会自动占用其他设备。具体命令见[双卡指南](https://hansimov.github.io/vflash/zh/guide/getting-started#parallel)。

为每个 worker 至少预留 **64 GiB 可用系统内存**，并为请求和其他进程保留余量；更大输入需要单独检查容量。

Turbo4 和 Turbo8 使用蒸馏 LoRA。精确注意力不等于基础模型质量，跨 GPU 的结果也不保证逐位一致。已验证范围与限制见[配置与硬件](https://hansimov.github.io/vflash/zh/guide/profiles)。

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
