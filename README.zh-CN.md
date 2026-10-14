# Vflash

采用显式GPU配置的原生 **MiniMax H3 推理引擎**。Vflash 使用 PyTorch 与 Triton 实现自己的去噪运行时，支持固定版本的 LightX2V Turbo LoRA。

[文档](https://hansimov.github.io/vflash/zh/) · [开始使用](https://hansimov.github.io/vflash/zh/guide/getting-started) · [版本更新](https://hansimov.github.io/vflash/zh/reference/releases) · [English](README.md)

**0.7.0：统一 Python、CLI 与服务配置。** 原生HTTP服务和`denoise`命令现在支持与
`H3Pipeline`、`generate`相同的显式Veda密集后端及层分配。默认首尾各五层密集（40稀疏/10密集），
显式首尾各三层得到44/6；就绪检查验证依赖与GPU架构。该选项属于近似注意力，不承诺输出相同或普遍质量/速度改善。

Vflash提供两类入口：

- **完整视频生成**：`H3Pipeline`与`vflash generate`管理条件编码、原生去噪、官方音视频解码和MP4编码。
  文本、关键帧、内容参考使用各自prepared配置；原始LightX v0.1及hybrid遵守已验证的
  [时长、参考数量与像素帧组合限制](https://hansimov.github.io/vflash/zh/guide/complete-pipeline#extended-hybrid)。
- **原生推理服务**：常驻GPU进程接收预编译条件包并返回音视频潜变量；有界HTTP队列不管理账号和持久任务，
  应用负责鉴权、调度与媒体交付。

显式配置覆盖单SM89、协作SM86/SM89双卡以及已验证的H100与Blackwell范围。准确显存、模式和几何必须符合
[当前配置指南](https://hansimov.github.io/vflash/zh/guide/profiles)，不能用显卡名称或旧发行记录推导容量。
设备和注意力后端不静默替换。

可复用的[人脸重绘与编码接口](https://hansimov.github.io/vflash/zh/guide/complete-pipeline)是可选基础模块；
人脸检测、轨迹选择、修复模型和最终合成由接入应用负责。单独安装引擎不会自动开启产品的完整人脸修复链路。
原模型、精度和注意力默认保持。

## 检查运行环境

需要 Python 3.11 或更新版本。基础安装不会下载模型权重或 PyTorch。

```bash
git clone --branch v0.7.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

vflash doctor
vflash profiles
vflash plan ref2va-turbo4-exact-sm89 --gpu 0
```

协作多卡须显式选择`--peer-gpu`，同一进程持有整个设备组直至关闭。主机内存需要覆盖权重、编码器、解码器及
当前请求，参见[硬件与内存限制](https://hansimov.github.io/vflash/zh/guide/profiles)。

## 接入应用

- [生成 MP4](https://hansimov.github.io/vflash/zh/guide/complete-pipeline)，使用纯文字或一至三张参考图，并在连续请求间复用模型。
- [编译官方权重](https://hansimov.github.io/vflash/zh/guide/compile-weights)，从固定版本的模型与 LoRA 创建 Ref4 或 Base4 运行资源。
- [运行条件包](https://hansimov.github.io/vflash/zh/guide/getting-started#run-a-bundle)，使用匹配的编译资源执行推理。
- [通过 Python 接入](https://hansimov.github.io/vflash/zh/guide/python)，在多个请求之间复用模型。
- [启动 Docker 与 HTTP 服务](https://hansimov.github.io/vflash/zh/guide/docker)，使用独立 worker 和有容量限制的任务队列。
- [测量性能与质量](https://hansimov.github.io/vflash/zh/reference/performance)，区分加载、推理和端到端成本。
- [对照 Sol-Engine](https://hansimov.github.io/vflash/zh/reference/sol-engine-alignment)，只采用通过 SM86/SM89 实测的机制。
- [排查问题](https://hansimov.github.io/vflash/zh/guide/troubleshooting)，在不削弱资源所有权检查的情况下定位启动、内存、双卡与输出故障。

Turbo4 和 Turbo8 使用蒸馏 LoRA。精确注意力不保证基础模型的质量；不同显卡或并行策略也不保证逐位一致的结果。完整生成使用明确固定配置的资产和会话。双 SM86 的 Ref4 必须使用 `sequence-head`；单 SM86 与其他仅原生Turbo配置使用条件包接口。W8 以及任意适配器或模式切换不在支持范围内。请查看[验证范围](https://hansimov.github.io/vflash/zh/guide/profiles)。

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
