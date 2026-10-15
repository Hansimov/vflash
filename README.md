# Vflash

Native **MiniMax H3 inference** with explicit GPU profiles. Vflash uses PyTorch and Triton, with its own denoising runtime and support for pinned LightX2V Turbo LoRAs.

[Documentation](https://hansimov.github.io/vflash/) · [Get started](https://hansimov.github.io/vflash/guide/getting-started) · [Release notes](https://hansimov.github.io/vflash/reference/releases) · [中文](README.zh-CN.md)

**0.9.0: direct RGB handoff and exact removal of repeated VAE work.** The Python
pipeline can deliver video-only uint8 arrays directly to a compositor, decode
only explicitly selected frames, and reuse identical padded encoder clips.
The ordinary MP4 interface and explicit 44/6 attention remain unchanged.
Two complete-video scenes preserved every final RGB frame and original audio
on one RTX PRO 6000. Savings depend on repeated inputs and unused frame spans;
this is a building block, not a general face-quality or latency guarantee.

Vflash provides two interfaces:

- **Complete video generation:** `H3Pipeline` and `vflash generate` own conditioning, native
  denoising, official video/audio decoding and MP4 encoding. Text, keyframes and references use
  distinct prepared profiles. Original LightX v0.1 and hybrid profiles support the qualified
  [duration, reference and pixel-frame limits](https://hansimov.github.io/vflash/guide/complete-pipeline#extended-hybrid).
- **Native inference service:** a resident GPU process accepts compiled conditioning bundles and
  returns video/audio latents. The bounded HTTP queue owns no user accounts or durable jobs;
  the integrating application owns authentication, scheduling and media delivery.

Explicit profiles cover single SM89, cooperating SM86/SM89 pairs, and qualified H100 and
Blackwell configurations. Exact GPU memory, mode and geometry matter; consult the
[current profile guide](https://hansimov.github.io/vflash/guide/profiles) rather than treating
hardware names or a historical release as a capacity guarantee. No automatic device substitution
or attention fallback occurs.

Reusable [video repaint and face-encoding APIs](https://hansimov.github.io/vflash/guide/complete-pipeline#explicit-native-face-repaint)
are opt-in building blocks. Face detection, track selection, restoration models and final compositing
belong to the integrating application. Installing the engine alone does not enable a product's full
face-restoration workflow. Existing model, precision and attention defaults remain unchanged.

## Check your setup

Python 3.11 or newer is required. The base install does not download model weights or PyTorch.

```bash
git clone --branch v0.9.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

vflash doctor
vflash profiles
vflash plan ref2va-turbo4-exact-sm89 --gpu 0
```

For cooperating GPUs, select the peer explicitly with `--peer-gpu`; one process owns the entire
group until it closes. Host memory must cover weights, encoders, decoders and the selected request.
See [hardware and memory limits](https://hansimov.github.io/vflash/guide/profiles).

## Build with Vflash

- [Generate an MP4](https://hansimov.github.io/vflash/guide/complete-pipeline) with the complete Ref4 or T2VA pipeline and its pinned official encoder/VAE adapters.
- [Prepare official weights](https://hansimov.github.io/vflash/guide/compile-weights), or [run a bundle](https://hansimov.github.io/vflash/guide/getting-started#run-a-bundle) with existing compatible assets.
- [Integrate through Python](https://hansimov.github.io/vflash/guide/python) and reuse a loaded model across requests.
- [Start Docker and HTTP](https://hansimov.github.io/vflash/guide/docker) for an isolated worker and bounded job queue.
- [Measure speed and quality](https://hansimov.github.io/vflash/reference/performance), with loading and end-to-end costs kept separate.
- [Compare Sol-Engine mechanisms](https://hansimov.github.io/vflash/reference/sol-engine-alignment) against the actual SM86/SM89 contract.
- [Troubleshoot](https://hansimov.github.io/vflash/guide/troubleshooting) startup, memory, dual-GPU and output problems without weakening ownership checks.

Turbo4 and Turbo8 are distilled adapters. Exact attention is not a base-model quality guarantee, and different GPU or parallel configurations need not produce bitwise-identical results. Complete generation uses explicit fixed-profile assets. Dual-SM86 Ref4 requires `sequence-head`; single-SM86 and other native-only Turbo profiles retain their latent interface. W8 and arbitrary adapter or mode switching are outside the supported profiles. Read [the validation scope](https://hansimov.github.io/vflash/guide/profiles).

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
