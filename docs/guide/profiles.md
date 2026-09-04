# Profiles and hardware

A profile chooses the model, LoRA revision, step count, arithmetic, and GPU memory strategy together. Choose one that matches both your hardware and compiled assets.

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

The **4090 profile** keeps weights in GPU memory. A resident service can reuse them across requests.

The **3080 profile** streams blocks from pinned system memory while the GPU computes. The tested 928 × 512, 124-frame, four-step workload uses about **60 GiB of process RAM**. We recommend **at least 64 GiB of available host RAM per worker** for that workload, with additional room for the operating system and other applications. Larger inputs and concurrent workers need separate capacity measurements; 64 GiB is not a universal capacity guarantee. Faster host-to-device transfers and sufficient RAM matter as well as GPU compute.

This release supports the 20 GB and 48 GB variants listed above. Other capacities, other GPU models, multi-GPU execution, and arbitrary resolution or frame-count combinations are not covered by the current support scope.

## Turbo LoRA support {#lora}

Vflash runs the pinned [LightX2V H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) adapters without the LightX2V inference framework. The profile's adapter, schedule, and step count must match the compiled resources.

Turbo4 and Turbo8 are distilled configurations. Fewer steps reduce compute, but they are not a promise of the same output quality as the 50-step base model. The `exact` name refers to the attention path and the selected adapter's execution; it does not promise identical tensors across different GPUs.

The 3080 preview has been checked for capacity and repeatable results between serial and overlapped loading on the same GPU. Independent reference comparison and broader quality evaluation are still pending.

Arbitrary LoRA files and automatic compatibility with future upstream revisions are not supported. See [runtime assets](../reference/runtime-assets) for version matching.

## What is outside this release {#scope}

The current public engine does not provide live text/reference encoding, VAE decoding, MP4 output, T2VA or first/last-frame generation, or dynamic LoRA loading. The HTTP API provides one serial execution lane; account management, billing, and distributed GPU scheduling belong to the application using Vflash.
