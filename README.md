# Vflash

**MiniMax H3 video and audio inference**, built with PyTorch and Triton. Run complete
video pipelines or integrate a resident latent-generation service with explicit
model assets, sampling recipes and GPU allocations.

[Documentation](https://hansimov.github.io/vflash/) · [Get started](https://hansimov.github.io/vflash/guide/getting-started) · [Release notes](https://hansimov.github.io/vflash/reference/releases) · [中文](README.zh-CN.md)

## Choose an interface

| Interface | Result | Start here |
| --- | --- | --- |
| `H3Pipeline` / `vflash generate` | Complete MP4 from text, keyframes or content references within the selected profile | [Complete pipelines](https://hansimov.github.io/vflash/guide/complete-pipeline) |
| `PikaAVRuntime` | Prepared Pika H3 Flex video with explicit joint or Larry soundtrack recipes | [Pika integration](https://hansimov.github.io/vflash/guide/pika-flex) |
| Native session / HTTP worker | Video and audio latents from a compiled conditioning bundle | [Python](https://hansimov.github.io/vflash/guide/python) · [HTTP](https://hansimov.github.io/vflash/reference/http-api) |

The current stable release is **0.27.0**. Its prepared Pika text/first-frame path includes
qualified four-second and exact 768P delivery. Other modes, lengths and canvases have
their own profiles; the [profile guide](https://hansimov.github.io/vflash/guide/profiles)
records supported hardware and measured scope. A model name alone does not select a
complete working configuration.

Vflash owns conditioning, sampling, decoding and local execution resources. The
integrating application owns accounts, durable job scheduling, multi-step workflows
and media delivery. Its bounded native HTTP queue is not a persistent application queue.

The **0.28.0a1 preview** adds the [original Remake editing client](https://hansimov.github.io/vflash/guide/remake). Select it explicitly; the stable installation below remains 0.27.0.

## Install and check your setup

Python 3.11 or newer is required. The base install does not download model weights or PyTorch.

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

The plan above checks one explicit profile; it does not run a generation. Install the
dependencies and obtain the pinned assets required by your chosen pipeline using the
[installation guide](https://hansimov.github.io/vflash/guide/getting-started).
For cooperating GPUs, select the peer with `--peer-gpu`. One process owns the whole
allocation, and host memory must cover the selected model and media components.

## Use and integrate

- [Generate video](https://hansimov.github.io/vflash/guide/complete-pipeline): text, first/last frames and references are distinct inputs.
- [Prepare weights](https://hansimov.github.io/vflash/guide/compile-weights) and [runtime assets](https://hansimov.github.io/vflash/reference/runtime-assets): use the exact model, adapter and component inventory.
- [Reuse a Python session](https://hansimov.github.io/vflash/guide/python) or [run Docker](https://hansimov.github.io/vflash/guide/docker).
- [Measure performance](https://hansimov.github.io/vflash/reference/performance) or [troubleshoot](https://hansimov.github.io/vflash/guide/troubleshooting) startup, memory and output failures.

Recipe-specific quality, timing and hardware evidence lives in the guides and
[release notes](https://hansimov.github.io/vflash/reference/releases). Exact attention,
distillation and a successful CPU check are different claims; use each recipe's
documented qualification. Repaint and face-encoding primitives are optional building
blocks, not a complete tracking or character-replacement workflow.

Development setup and release instructions are in the
[contribution guide](https://hansimov.github.io/vflash/guide/contributing).

## License

Vflash source is [Apache-2.0](LICENSE). Models and adapters retain their own licenses and
terms; this repository contains no model weights. The LightX2V inference framework is
not a runtime dependency. See [sources and acknowledgements](https://hansimov.github.io/vflash/reference/license).
