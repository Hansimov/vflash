# Vflash

Native MiniMax H3 inference for **RTX 3080 20 GB** and **RTX 4090 48 GB**, with support for pinned LightX2V Turbo LoRAs. Vflash implements its own denoising runtime with PyTorch and Triton; it does not depend on the LightX2V inference framework.

[Documentation](https://hansimov.github.io/vflash/) · [Getting started](https://hansimov.github.io/vflash/guide/getting-started) · [中文](README.zh-CN.md)

> **Alpha preview:** compiled conditioning bundles in, video and audio latents (tensors ready for decoding) out. Compatible runtime assets are required and are not yet publicly available. Prompt processing, reference uploads, and MP4 output are still in development.

## Get started

Install the lightweight CLI to inspect hardware and supported profiles. This does not download model weights or PyTorch.

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

Python 3.11 or newer is required. For GPU execution, see the [installation guide](https://hansimov.github.io/vflash/guide/getting-started) or [Docker and API setup](docker/README.md).

## Supported profiles

| GPU | Profiles | Memory strategy |
| --- | --- | --- |
| RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8 | Weights stay in GPU memory |
| RTX 3080 20 GB | Ref2VA Turbo4 | Blocks stream from host memory |

For the tested 3080 workload, the process uses about 60 GiB of RAM. We recommend at least **64 GiB of available system memory per worker**, plus headroom for other processes. Larger inputs need separate capacity checks. It has been checked for capacity and same-GPU serial/overlapped loading consistency; broader quality and independent reference checks remain pending. These profiles target the stated GPU memory capacities.

Turbo4 and Turbo8 use distilled adapters. Exact attention does not promise base-model quality or identical results across GPUs. See [profiles and hardware](https://hansimov.github.io/vflash/guide/profiles) for the complete scope.

## Use it in an application

- [Run a compiled bundle](https://hansimov.github.io/vflash/guide/getting-started#run-a-bundle) and save latent tensors.
- [Start the HTTP service](https://hansimov.github.io/vflash/guide/docker) to reuse a loaded model across requests.
- [Understand the runtime inputs](https://hansimov.github.io/vflash/reference/runtime-assets) and match their versions.
- [Measure performance](https://hansimov.github.io/vflash/reference/performance) with first-use and repeated-request costs separated.

## Development

```bash
python -m pip install -e '.[dev,server]'
pre-commit install
pre-commit run --all-files
pytest
```

Build the documentation with `npm ci` and `npm run docs:build`.

## License

Vflash source code is [Apache-2.0](LICENSE). Model weights and adapters retain their own licenses and usage terms; this repository contains no model weights.

Thanks to [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [LightX2V](https://github.com/ModelTC/LightX2V), and the PyTorch, Triton, and CUDA communities. See [license and acknowledgements](https://hansimov.github.io/vflash/reference/license) for sources and additional credits.
