# Vflash

使用 PyTorch 和 Triton 的 MiniMax H3 音视频推理引擎。可以从文字或参考图生成带声音的 MP4、编辑源视频，或将推理会话集成到自己的服务中。

[English](README.md) · [中文文档](https://hansimov.github.io/vflash/zh/) · [下载 0.31.0](https://github.com/Hansimov/vflash/releases/tag/v0.31.0)

## 从任务开始

| 目标 | 入口 | 结果 |
| --- | --- | --- |
| [文字或图像生成视频](https://hansimov.github.io/vflash/zh/guide/complete-pipeline) | H3Pipeline / CLI | MP4 |
| [编辑源片](https://hansimov.github.io/vflash/zh/guide/remake) | Remake | 编辑后的视频与恢复凭据 |
| [选取人物或场地参考](https://hansimov.github.io/vflash/zh/guide/remake#reference-view) | CPU 与 Pillow | 同一区域的图像编辑输入、视频外观输入及回执 |
| [使用 Pika 资源生成](https://hansimov.github.io/vflash/zh/guide/pika-flex) | PikaAVRuntime | 视频与指定配方生成的音轨 |
| [部署条件包推理服务](https://hansimov.github.io/vflash/zh/reference/http-api) | HTTP | 原生音视频潜变量 |

## 安装与选卡

```sh
python -m pip install https://github.com/Hansimov/vflash/releases/download/v0.31.0/vflash-0.31.0-py3-none-any.whl
vflash doctor
```

[安装与检查](https://hansimov.github.io/vflash/zh/guide/getting-started) · [显卡与显存要求](https://hansimov.github.io/vflash/zh/guide/profiles) · [从 0.28 升级](https://hansimov.github.io/vflash/zh/reference/releases)

代码采用 Apache-2.0。模型和适配器分别适用上游条款。[许可与致谢](https://hansimov.github.io/vflash/zh/reference/license)
