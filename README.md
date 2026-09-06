# Vflash

Native **MiniMax H3 inference** for RTX 3080 20 GB and RTX 4090 48 GB. Vflash uses PyTorch and Triton, with its own denoising runtime and support for pinned LightX2V Turbo LoRAs.

[Documentation](https://hansimov.github.io/vflash/) · [Get started](https://hansimov.github.io/vflash/guide/getting-started) · [Release notes](https://hansimov.github.io/vflash/reference/releases) · [中文](README.zh-CN.md)

**0.1.0a5 · Developer preview.** The public interface accepts compiled conditioning bundles and returns video/audio latent tensors for decoding. Compatible runtime assets are required and are not yet published. Prompt processing, reference uploads and MP4 output are outside this release.

## Check your setup

Python 3.11 or newer is required. The base install does not download model weights or PyTorch.

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

| GPU configuration | Released profiles | Weight placement |
| --- | --- | --- |
| One RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8 | Resident by default; optional block streaming |
| One RTX 3080 20 GB | Ref2VA Turbo4 | Streamed from host RAM |
| Two RTX 3080 20 GB GPUs | Ref2VA Turbo4 | Shared host weights; cooperative execution |

For a cooperating 3080 pair, select the peer explicitly with `--peer-gpu 1`. The default paired strategy is `sequence-head`; `tensor` is also available. The engine never selects another device automatically.

Allow **64 GiB or more of available system memory per worker** for the tested workload, plus headroom. Larger inputs need separate capacity checks. These profiles cover the listed memory capacities. See [hardware, adapters and quality limits](https://hansimov.github.io/vflash/guide/profiles).

## Build with Vflash

- [Run a bundle](https://hansimov.github.io/vflash/guide/getting-started#run-a-bundle) with compatible compiled assets.
- [Integrate through Python](https://hansimov.github.io/vflash/guide/python) and reuse a loaded model across requests.
- [Start Docker and HTTP](https://hansimov.github.io/vflash/guide/docker) for an isolated worker and bounded job queue.
- [Measure speed and quality](https://hansimov.github.io/vflash/reference/performance), with loading and end-to-end costs kept separate.

Turbo4 and Turbo8 are distilled adapters. Exact attention is not a base-model quality guarantee, and different GPU or parallel configurations need not produce bitwise-identical results. W8 is not a released profile. The source tree also includes an SM89 T2VA Turbo4 preview; see [its scope and source revision](https://hansimov.github.io/vflash/guide/profiles#t2va). The `0.1.0a5` tag above remains Ref2VA-only.

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
