# Vflash

Native **MiniMax H3 inference** for RTX 3080 20 GB and RTX 4090 48 GB. Vflash uses PyTorch and Triton, with its own denoising runtime and support for pinned LightX2V Turbo LoRAs.

[Documentation](https://hansimov.github.io/vflash/) · [Get started](https://hansimov.github.io/vflash/guide/getting-started) · [Release notes](https://hansimov.github.io/vflash/reference/releases) · [中文](README.zh-CN.md)

**0.4.0.** Generate synchronized video and audio through a complete Python or container pipeline.
Turbo profiles create five-second MP4s from text, one to three images, or
[a short reference video](https://hansimov.github.io/vflash/guide/complete-pipeline#reference-video).
Official Base16 keyframe profiles accept a first frame, a last frame, or both, at integer durations
from five through ten seconds. One prepared keyframe pipeline can switch among I2VA, L2VA and FL2VA
without reloading weights.

The native core targets RTX 3080 20 GB (SM86) and RTX 4090 48 GB (SM89). Version 0.4.0 adds exact
destination-major relayouts for two-GPU `sequence-head`, preserves the dense BF16 attention and
collective wire contract, and documents a hardware-measured comparison with NVIDIA Sol-Engine.
Sparse attention, cross-step caches and quantized communication remain outside the exact defaults.
See [what was adopted, deferred or rejected](https://hansimov.github.io/vflash/reference/sol-engine-alignment).

The ten-second Base16 boundary has bounded complete-request evidence on the listed hardware, but it
is not a guarantee for every canvas or prompt. A matching SM89 pair is an explicit single-request
latency option; two independent workers remain the throughput default. Turbo and reference-video
profiles retain their separate five-second contracts. [Read the qualification boundary](https://hansimov.github.io/vflash/reference/releases#v0-4-0).

## Check your setup

Python 3.11 or newer is required. The base install does not download model weights or PyTorch.

```bash
git clone --branch v0.4.0 --depth 1 https://github.com/Hansimov/vflash.git
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
| One RTX 4090 48 GB | Ref2VA Turbo4 / Turbo8; T2VA Turbo4; Base16 I2VA/L2VA/FL2VA | Resident native core; complete pipeline uses block streaming |
| One RTX 3080 20 GB | Ref2VA Turbo4 | Streamed from host RAM |
| Two RTX 3080 20 GB GPUs | Ref2VA Turbo4; T2VA Turbo4; Base16 I2VA/L2VA/FL2VA | Shared host weights; cooperative `sequence-head` |
| Two RTX 4090 48 GB GPUs | Base16 I2VA/L2VA/FL2VA | Opt-in cooperative latency path; not the throughput default |

For a cooperating 3080 pair, select the peer explicitly with `--peer-gpu 1`. T2VA requires `sequence-head`; native Ref4 also offers `tensor`. The engine never selects another device automatically.

For the native denoiser, allow **64 GiB or more of available system memory per worker** for the tested workload, plus headroom. The complete pipeline also keeps encoders and decoders in host memory and needs a larger budget. Larger inputs need separate capacity checks. See [hardware, adapters and quality limits](https://hansimov.github.io/vflash/guide/profiles).

## Build with Vflash

- [Generate an MP4](https://hansimov.github.io/vflash/guide/complete-pipeline) with the complete Ref4 or T2VA pipeline and its pinned official encoder/VAE adapters.
- [Prepare official weights](https://hansimov.github.io/vflash/guide/compile-weights), or [run a bundle](https://hansimov.github.io/vflash/guide/getting-started#run-a-bundle) with existing compatible assets.
- [Integrate through Python](https://hansimov.github.io/vflash/guide/python) and reuse a loaded model across requests.
- [Start Docker and HTTP](https://hansimov.github.io/vflash/guide/docker) for an isolated worker and bounded job queue.
- [Measure speed and quality](https://hansimov.github.io/vflash/reference/performance), with loading and end-to-end costs kept separate.
- [Compare Sol-Engine mechanisms](https://hansimov.github.io/vflash/reference/sol-engine-alignment) against the actual SM86/SM89 contract.
- [Troubleshoot](https://hansimov.github.io/vflash/guide/troubleshooting) startup, memory, dual-GPU and output problems without weakening ownership checks.

Turbo4 and Turbo8 are distilled adapters. Exact attention is not a base-model quality guarantee, and different GPU or parallel configurations need not produce bitwise-identical results. Complete generation uses fixed Ref4 or T2VA Base4 v1.0 assets and sessions. Dual-SM86 Ref4 requires `sequence-head`; single-SM86 and Turbo8 use the native latent interface. W8 and arbitrary adapter or mode switching are outside the supported profiles. Read [the validation scope](https://hansimov.github.io/vflash/guide/profiles).

## Contribute

```bash
python -m pip install -e '.[dev,server]'
pre-commit install
pre-commit run --all-files
pytest
```

Build the bilingual docs with `npm ci` and `npm run docs:build`. See the
[contribution guide](https://hansimov.github.io/vflash/guide/contributing) and [contributor map](AGENTS.md)
for evidence, privacy and release requirements.

## License

Vflash source is [Apache-2.0](LICENSE). Models and adapters retain their own licenses and terms; this repository contains no model weights. The LightX2V inference framework is not a runtime dependency.

Thanks to [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [LightX2V](https://github.com/ModelTC/LightX2V), and the PyTorch, Triton and CUDA communities. [Sources and acknowledgements](https://hansimov.github.io/vflash/reference/license).
