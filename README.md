# Vflash

Native **MiniMax H3 inference** for RTX 3080 20 GB and RTX 4090 48 GB. Vflash uses PyTorch and Triton, with its own denoising runtime and support for pinned LightX2V Turbo LoRAs.

[Documentation](https://hansimov.github.io/vflash/) · [Get started](https://hansimov.github.io/vflash/guide/getting-started) · [Release notes](https://hansimov.github.io/vflash/reference/releases) · [中文](README.zh-CN.md)

**0.2.0.** Generate a five-second MP4 from text alone or from a prompt and one to three ordered reference images. The [complete pipeline](https://hansimov.github.io/vflash/guide/complete-pipeline) runs through Python or the container CLI on one RTX 4090 48 GB. [Compile its runtime assets](https://hansimov.github.io/vflash/guide/compile-weights) directly from fixed official weights. The native Python, CLI and HTTP interfaces also accept conditioning bundles and return video/audio latents on the GPUs below.

## Check your setup

Python 3.11 or newer is required. The base install does not download model weights or PyTorch.

```bash
git clone --branch v0.2.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

vflash doctor
vflash profiles
vflash plan ref2va-turbo4-exact-sm89 --gpu 0
```

| GPU configuration | Released profiles | Weight placement |
| --- | --- | --- |
| One RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8; T2VA Turbo4 | Resident by default; optional block streaming |
| One RTX 3080 20 GB | Ref2VA Turbo4 | Streamed from host RAM |
| Two RTX 3080 20 GB GPUs | Ref2VA Turbo4 | Shared host weights; cooperative execution |

For a cooperating 3080 pair, select the peer explicitly with `--peer-gpu 1`. The default paired strategy is `sequence-head`; `tensor` is also available. The engine never selects another device automatically.

For the native denoiser, allow **64 GiB or more of available system memory per worker** for the tested workload, plus headroom. The complete pipeline also keeps encoders and decoders in host memory and needs a larger budget. Larger inputs need separate capacity checks. See [hardware, adapters and quality limits](https://hansimov.github.io/vflash/guide/profiles).

## Build with Vflash

- [Generate an MP4](https://hansimov.github.io/vflash/guide/complete-pipeline) with the complete Ref4 or T2VA pipeline and its pinned official encoder/VAE adapters.
- [Prepare official weights](https://hansimov.github.io/vflash/guide/compile-weights), or [run a bundle](https://hansimov.github.io/vflash/guide/getting-started#run-a-bundle) with existing compatible assets.
- [Integrate through Python](https://hansimov.github.io/vflash/guide/python) and reuse a loaded model across requests.
- [Start Docker and HTTP](https://hansimov.github.io/vflash/guide/docker) for an isolated worker and bounded job queue.
- [Measure speed and quality](https://hansimov.github.io/vflash/reference/performance), with loading and end-to-end costs kept separate.

Turbo4 and Turbo8 are distilled adapters. Exact attention is not a base-model quality guarantee, and different GPU or parallel configurations need not produce bitwise-identical results. Complete generation supports Ref4 and T2VA Base4 v1.0 on SM89; each uses its own fixed model assets and session. SM86 and Turbo8 retain their native latent interfaces. W8, dynamic LoRA loading and first/last-frame generation are not released features. Read [the validation scope](https://hansimov.github.io/vflash/guide/profiles).

## Contribute

```bash
python -m pip install -e '.[dev,server]'
pre-commit install
pre-commit run --all-files
pytest
```

Build the bilingual docs with `npm ci` and `npm run docs:build`. See [the contributor map](AGENTS.md) for code ownership.

## License

Vflash source is [Apache-2.0](LICENSE). Models and adapters retain their own licenses and terms; this repository contains no model weights. The LightX2V inference framework is not a runtime dependency.

Thanks to [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [LightX2V](https://github.com/ModelTC/LightX2V), and the PyTorch, Triton and CUDA communities. [Sources and acknowledgements](https://hansimov.github.io/vflash/reference/license).
