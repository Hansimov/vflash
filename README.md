# Vflash

Generate video with sound or edit existing footage using MiniMax H3, PyTorch and Triton. Use a complete pipeline for MP4s, or integrate a resident inference session into your own service.

[中文](README.zh-CN.md) · [Documentation](https://hansimov.github.io/vflash/) · [Download 0.31.1](https://github.com/Hansimov/vflash/releases/tag/v0.31.1)

## Start with a task

| Task | Interface | Output |
| --- | --- | --- |
| [Generate from text or images](https://hansimov.github.io/vflash/guide/complete-pipeline) | H3Pipeline / CLI | MP4 |
| [Edit a source video](https://hansimov.github.io/vflash/guide/remake) | Remake | Edited clip and recovery journal |
| [Select a portrait or setting](https://hansimov.github.io/vflash/guide/remake#reference-view) | CPU and Pillow | Image-editor view, video appearance reference and receipt |
| [Generate with prepared Pika assets](https://hansimov.github.io/vflash/guide/pika-flex) | PikaAVRuntime | Video and a chosen soundtrack |
| [Serve a conditioning bundle](https://hansimov.github.io/vflash/reference/http-api) | HTTP | Native audiovisual latents |

## Install and select hardware

```sh
python -m pip install https://github.com/Hansimov/vflash/releases/download/v0.31.1/vflash-0.31.1-py3-none-any.whl
vflash doctor
```

[Installation](https://hansimov.github.io/vflash/guide/getting-started) · [GPU and memory requirements](https://hansimov.github.io/vflash/guide/profiles) · [Upgrade from 0.28](https://hansimov.github.io/vflash/reference/releases)

The code uses Apache-2.0. Model and adapter weights have their own terms. [Licenses and credits](https://hansimov.github.io/vflash/reference/license)
