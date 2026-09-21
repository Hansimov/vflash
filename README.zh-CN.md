# Vflash

面向 RTX 3080 20 GB 和 RTX 4090 48 GB 的原生 **MiniMax H3 推理引擎**。Vflash 使用 PyTorch 与 Triton 实现自己的去噪运行时，支持固定版本的 LightX2V Turbo LoRA。

[文档](https://hansimov.github.io/vflash/zh/) · [开始使用](https://hansimov.github.io/vflash/zh/guide/getting-started) · [版本更新](https://hansimov.github.io/vflash/zh/reference/releases) · [English](README.md)

**0.5.1 正式版。** 去掉未使用的H3条件编码尾层，保持实际消费的状态不变。
[SM86/SM89条件与整片对照](https://hansimov.github.io/vflash/zh/reference/benchmarks#encoder-prefix)
明确区分编码收益和很小的整片差值。

通过完整 Python 或容器 pipeline 生成同步视频和音频。Turbo profile 使用纯文字、
一至三张图片或[一段短参考视频](https://hansimov.github.io/vflash/zh/guide/complete-pipeline#reference-video)
生成五秒 MP4；官方 Base16 关键帧 profile 接受首帧、尾帧或两者，并支持五至十秒的整数时长。同一个
prepared keyframe pipeline 可以在 I2VA、L2VA 和 FL2VA 之间切换，不重新加载权重。

原生核心面向 RTX 3080 20 GB（SM86）和 RTX 4090 48 GB（SM89）。0.5.1 在**单SM89官方Base16默认
使用近似Sol attention**；SM86、协作双卡及Turbo仍为dense。Python、CLI与原生HTTP服务共享该策略，
可显式用`--attention-backend torch-flash`保留dense。标准Docker构建包含固定Sol依赖；Python安装
GPU依赖后执行`python -m vflash.install_sol`。缺依赖明确报错，不静默切换后端。
跨步cache与量化通信不进入默认；不承诺完整媒体质量等价。
详见[采用、暂缓和拒绝的机制](https://hansimov.github.io/vflash/zh/reference/sol-engine-alignment)。

十秒 Base16 边界在列出的硬件上已有有界完整请求证据，但不代表任意画布或提示词都得到保证。匹配的
SM89 双卡是显式的单请求低延迟选项；两个独立 worker 仍是吞吐默认。Turbo 和视频参考继续保持各自的
五秒合同。[查看资格边界](https://hansimov.github.io/vflash/zh/reference/releases#v0-4-0)。

## 检查运行环境

需要 Python 3.11 或更新版本。基础安装不会下载模型权重或 PyTorch。

```bash
git clone --branch v0.5.1 --depth 1 https://github.com/Hansimov/vflash.git
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
| 单 RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8；T2VA Turbo4；Base16 I2VA/L2VA/FL2VA | Base16 Sol使用分块加载，Turbo原生核心保持常驻 |
| 单 RTX 3080 20 GB | Ref2VA Turbo4 | 从系统内存分块加载 |
| 双 RTX 3080 20 GB | Ref2VA Turbo4；T2VA Turbo4；Base16 I2VA/L2VA/FL2VA | 共享主机权重，协作 `sequence-head` |
| 双 RTX 4090 48 GB | Base16 I2VA/L2VA/FL2VA | 可选的协作低延迟路径，不是吞吐默认 |

使用双 3080 时，通过 `--peer-gpu 1` 显式选择第二张卡。T2VA 必须使用 `sequence-head`；原生 Ref4 也可选择 `tensor`。引擎不会自动占用其他显卡。

对于原生去噪器的已测负载，建议**每个 worker 预留至少 64 GiB 可用系统内存**，并保留额外余量。完整链路还需在主机上保存编码器和解码器，需要更大的内存预算。更大输入需要重新检查容量，详情见[硬件、LoRA 与质量限制](https://hansimov.github.io/vflash/zh/guide/profiles)。

## 接入应用

- [生成 MP4](https://hansimov.github.io/vflash/zh/guide/complete-pipeline)，使用纯文字或一至三张参考图，并在连续请求间复用模型。
- [编译官方权重](https://hansimov.github.io/vflash/zh/guide/compile-weights)，从固定版本的模型与 LoRA 创建 Ref4 或 Base4 运行资源。
- [运行条件包](https://hansimov.github.io/vflash/zh/guide/getting-started#run-a-bundle)，使用匹配的编译资源执行推理。
- [通过 Python 接入](https://hansimov.github.io/vflash/zh/guide/python)，在多个请求之间复用模型。
- [启动 Docker 与 HTTP 服务](https://hansimov.github.io/vflash/zh/guide/docker)，使用独立 worker 和有容量限制的任务队列。
- [测量性能与质量](https://hansimov.github.io/vflash/zh/reference/performance)，区分加载、推理和端到端成本。
- [对照 Sol-Engine](https://hansimov.github.io/vflash/zh/reference/sol-engine-alignment)，只采用通过 SM86/SM89 实测的机制。
- [排查问题](https://hansimov.github.io/vflash/zh/guide/troubleshooting)，在不削弱资源所有权检查的情况下定位启动、内存、双卡与输出故障。

Turbo4 和 Turbo8 使用蒸馏 LoRA。精确注意力不保证基础模型的质量；不同显卡或并行策略也不保证逐位一致的结果。完整生成使用固定的 Ref4 或 T2VA Base4 v1.0 资产和会话。双 SM86 的 Ref4 必须使用 `sequence-head`；单 SM86 与 Turbo8 使用原生条件包接口。W8 以及任意适配器或模式切换不在支持范围内。请查看[验证范围](https://hansimov.github.io/vflash/zh/guide/profiles)。

## 参与开发

```bash
python -m pip install -e '.[dev,server]'
pre-commit install
pre-commit run --all-files
pytest
```

使用 `npm ci` 和 `npm run docs:build` 构建双语文档。证据、隐私与发布要求见
[参与开发](https://hansimov.github.io/vflash/zh/guide/contributing)和[贡献者地图](AGENTS.md)。

## 许可证

Vflash 源码使用 [Apache-2.0](LICENSE)。模型和 LoRA 各自遵循相应许可证和使用条款，本仓库不含模型权重。运行时不依赖 LightX2V 推理框架。

感谢 [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[LightX2V](https://github.com/ModelTC/LightX2V)，以及 PyTorch、Triton 和 CUDA 社区。[来源与致谢](https://hansimov.github.io/vflash/zh/reference/license)。
