# Vflash

基于 PyTorch 与 Triton 的 **MiniMax H3 音视频推理引擎**。提供完整视频流水线及常驻潜变量
生成服务，明确指定模型资产、采样配方和 GPU 分配。

[文档](https://hansimov.github.io/vflash/zh/) · [开始使用](https://hansimov.github.io/vflash/zh/guide/getting-started) · [版本更新](https://hansimov.github.io/vflash/zh/reference/releases) · [English](README.md)

## 选择入口

| 入口 | 输出 | 使用说明 |
| --- | --- | --- |
| `H3Pipeline` / `vflash generate` | 在所选配置范围内，从文本、关键帧或内容参考生成完整 MP4 | [完整流水线](https://hansimov.github.io/vflash/zh/guide/complete-pipeline) |
| `PikaAVRuntime` | 准备权重的 Pika H3 Flex 画面，明确选择联合声音或 Larry 声音配方 | [Pika 接入](https://hansimov.github.io/vflash/zh/guide/pika-flex) |
| 原生会话 / HTTP worker | 从编译条件包生成音视频潜变量 | [Python](https://hansimov.github.io/vflash/zh/guide/python) · [HTTP](https://hansimov.github.io/vflash/zh/reference/http-api) |

当前稳定版本为 **0.27.0**。准备权重的 Pika 文本／首帧路径包含已验证的四秒及精确768P交付。
其他模式、时长和画布分别遵守各自配置；[配置指南](https://hansimov.github.io/vflash/zh/guide/profiles)
记录硬件与实测范围。模型名称本身不能替代完整运行配置。

Vflash负责条件编码、采样、解码和本地执行资源。接入应用负责账户、持久任务调度、多步骤
工作流及媒体交付。原生HTTP的有界队列不是应用层持久任务队列。

**0.28.0a3 预览版**新增[原版 Remake 编辑客户端](https://hansimov.github.io/vflash/zh/guide/remake)，需显式选用；下方稳定安装仍为0.27.0。

## 安装与环境检查

需要 Python 3.11 或更新版本。基础安装不会下载模型权重或 PyTorch。

```bash
git clone --branch v0.27.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

vflash doctor
vflash profiles
vflash plan ref2va-turbo4-exact-sm89 --gpu 0
```

上述plan命令检查一个明确配置，不执行生成。按[安装指南](https://hansimov.github.io/vflash/zh/guide/getting-started)
为选定流水线安装依赖并取得固定资产。协作多卡须用`--peer-gpu`指定第二张卡，同一进程
持有整个分配；主机内存也需要覆盖所选模型和媒体组件。

## 使用与接入

- [生成视频](https://hansimov.github.io/vflash/zh/guide/complete-pipeline)：文本、首尾帧与内容参考分别表达。
- [准备权重](https://hansimov.github.io/vflash/zh/guide/compile-weights)及[运行资产](https://hansimov.github.io/vflash/zh/reference/runtime-assets)：使用准确的模型、适配器与组件清单。
- [复用Python会话](https://hansimov.github.io/vflash/zh/guide/python)，或[部署Docker](https://hansimov.github.io/vflash/zh/guide/docker)。
- [测量性能](https://hansimov.github.io/vflash/zh/reference/performance)，或[排查](https://hansimov.github.io/vflash/zh/guide/troubleshooting)启动、内存及输出故障。

各配方的质量、耗时和硬件证据保存在指南与[版本记录](https://hansimov.github.io/vflash/zh/reference/releases)。
精确注意力、蒸馏和CPU检查成功分别表达不同事实，应沿所选配方的实测范围使用。
重绘与人脸编码属于可选基础模块，不等于完整的人物跟踪或替换工作流。

开发环境和发布步骤见[贡献指南](https://hansimov.github.io/vflash/zh/guide/contributing)。

## 许可证

Vflash源码使用[Apache-2.0](LICENSE)。模型和适配器各自遵循相应许可证及使用条款，本仓库
不含模型权重，运行时不依赖LightX2V推理框架。参见[来源与致谢](https://hansimov.github.io/vflash/zh/reference/license)。
