# Vflash

Native **MiniMax H3 inference** with explicit GPU profiles. Vflash uses PyTorch and Triton, with its own denoising runtime and support for pinned LightX2V Turbo LoRAs.

[Documentation](https://hansimov.github.io/vflash/) · [Get started](https://hansimov.github.io/vflash/guide/getting-started) · [Release notes](https://hansimov.github.io/vflash/reference/releases) · [中文](README.zh-CN.md)

**0.21.0: combine appearance images and motion video.** The hybrid pipeline accepts ordered image references alongside one video. Explicit video-only spatial pooling preserves the appearance latents and lowers motion-context cost. Initial complete clips ran on RTX 4090 48 GiB and RTX PRO 6000 Blackwell Server; see [hardware timings and measured scope](https://hansimov.github.io/vflash/reference/releases).

**0.17.2: choose the Veda layer partition independently of precision.** Set
`veda_dense_edge_layers=4` for 42 sparse / 8 dense layers, including the optional
audio context pass. SM120 keeps the selected Sage FP8 or FROST precision policy.
The default stays 40/10; see [matched execution results](https://hansimov.github.io/vflash/reference/releases).

**0.16.0: faster SM120 attention.** On RTX PRO 6000, `veda_dense_backend="sageattention2"` now runs
INT8 Q·K / FP8 P·V in both dense and Veda sparse layers: denoising 6–14% shorter, with flicker and
sharpness measured against the BF16 baseline. See the [release notes](https://hansimov.github.io/vflash/reference/releases).

**0.15.0: resident 4 MP on 96 GiB Blackwell.** Resident I2VA on one SM120 GPU now follows
the native-HD pixel-frame budget instead of a fixed 1536×864 canvas: an RTX PRO 6000 with the
FP8 trunk, face and six-step audio repair completed 1920×1088/5 s in 163 s, 2720×1536/5 s in
261 s and 1920×1088/10 s in 245 s at a 42.1 GiB peak. See
[residency](https://hansimov.github.io/vflash/guide/profiles#h100-resident).

**0.14.0: true 4 MP native HD.** Image-conditioned native HD keeps the 2048² pixel
cap but accepts a 2720-pixel long edge, so five-second requests reach 2720×1536 (16:9),
1536×2720 (9:16) or 2368×1760 (4:3). Five complete RTX 4090 requests across these shapes
stayed coherent; the joint pixel-frame budget is unchanged. See the
[canvas limits](https://hansimov.github.io/vflash/guide/complete-pipeline#extended-hybrid).

**0.13.0: opt-in audio hidden-context reuse.** The owned six-step audio interface
can freeze first-evaluation non-audio hidden states, with current audio and rebuilt
K/V on every later step. Two 512²/5s comparisons on a 48 GiB RTX 4090 retained every video frame
and received listening acceptance, with about 34% less audio time. This is an
explicit approximation, disabled by default; larger cached workloads remain
unqualified. See the [audio guide](docs/guide/complete-pipeline.md#audio-hidden-context).

The optional six-step audio pass supports the native BF16 weight ring and
qualified resident FP8 paths with Veda 44/6 and approximately 0.5 MP spatial
context. Complete video pixels remain unchanged. The [pipeline guide](https://hansimov.github.io/vflash/guide/complete-pipeline)
separates hardware qualification, listening evidence and execution cost.

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
git clone --branch v0.19.0 --depth 1 https://github.com/Hansimov/vflash.git
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
