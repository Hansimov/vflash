# Profiles and hardware

A profile chooses the model, LoRA revision, step count, and arithmetic. Its default memory strategy follows the selected hardware. Choose one that matches both your hardware and compiled assets.

## Available profiles {#available}

All current profiles run Ref2VA denoising from compiled conditioning bundles. Ref2VA is H3's reference-conditioned video and audio mode.

| GPU | Profile | Steps | Weight loading |
| --- | --- | ---: | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89` | 4 | Resident in GPU memory |
| RTX 4090 48 GB | `ref2va-turbo8-exact-sm89` | 8 | Resident in GPU memory |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` | 4 | Streamed from host memory |

The HTTP service defaults to Turbo4 on a 4090. To change profiles, restart the service with the selected profile and its matching assets.

```bash
vflash profiles
vflash plan ref2va-turbo8-exact-sm89 --gpu 0
```

## Memory and deployment {#memory}

| Hardware | Memory strategy | What to consider |
| --- | --- | --- |
| One 4090 48 GB | Resident weights by default | Reuse weights across requests; leave room for activations |
| One or two 3080 20 GB GPUs | Blocks streamed from host RAM | Full weights remain in system memory; a pair cooperates on one request |
| One 4090 needing more VRAM headroom | Explicit `block-ring` | Trade host RAM for device headroom and measure the latency cost |

Use `--weight-residency block-ring` in the CLI or the matching [Python session option](./python#memory). The HTTP service uses the selected profile's default strategy.

For the measured 928 × 512, 124-model-frame, four-step workload, allow **at least 64 GiB of available system memory per worker**, plus headroom for the OS and other processes. With block streaming, VRAM usage excludes the complete weights stored in host RAM. Larger inputs and concurrent workers require their own capacity checks.

A 3080 pair supports `sequence-head` or `tensor`; the caller selects the peer explicitly. The measured topology uses PCIe 3.0 x16 host-bridge links without peer access and does not require NVLink. See [dual-GPU setup](./getting-started#parallel) and the [measurement scope](../reference/benchmarks#sm86-parallel).

This support scope covers the 20 GB 3080 and 48 GB 4090 variants. Other capacities, models, larger GPU groups and arbitrary resolutions or frame counts have not received the same validation.

## Turbo LoRA support {#lora}

Vflash runs the pinned [LightX2V H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) adapters without the LightX2V inference framework. The profile's adapter, schedule, and step count must match the compiled resources.

| Adapter | Supported hardware | Upstream file |
| --- | --- | --- |
| Turbo4 v0.1 | One/two 3080s, one 4090 | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` |
| Turbo8 v1.0 768p | One 4090 | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |

Pinned revisions and sources are listed under [runtime assets](../reference/runtime-assets#versions). Only these Ref2VA files are supported. ComfyUI, FL2VA, custom LoRAs and future upstream revisions need separate integration.

Turbo4 and Turbo8 are distilled configurations. Fewer steps reduce compute, but they are not a promise of the same output quality as the 50-step base model. The `exact` name refers to the attention path and the selected adapter's execution; it does not promise identical tensors across different GPUs.

The two-device modes have completed full trajectories and one paired decoded-video/audio smoke. They retain exact attention but introduce different floating-point rounding; they are not a same-output or same-quality guarantee.

The single-device 3080 preview has been checked for capacity and repeatable results between serial and overlapped loading on the same GPU. Independent reference comparison and broader quality evaluation are still pending.

## What is outside this release {#scope}

The current public engine does not provide live text/reference encoding, VAE decoding, MP4 output, T2VA or first/last-frame generation, or dynamic LoRA loading. The HTTP API provides one serial execution lane; account management, billing, and distributed GPU scheduling belong to the application using Vflash.
