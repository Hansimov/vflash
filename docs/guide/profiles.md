# Profiles and hardware

A profile chooses the model, LoRA revision, step count, and arithmetic. Its default memory strategy follows the selected hardware. Choose one that matches both your hardware and compiled assets.

## Available profiles {#available}

Ref2VA generates video and audio from reference conditioning. T2VA uses text conditioning without reference images. Base and Ref weights are separate. The [complete pipeline](./complete-pipeline) accepts text alone or a prompt with one to three images on an RTX 4090 48 GB. Two RTX 3080 20 GB GPUs also support complete Ref4 generation. The native interfaces below use compiled conditioning bundles; their default residency differs from the complete pipeline.

| GPU | Profile | Steps | Weight loading |
| --- | --- | ---: | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89` | 4 | Resident in GPU memory |
| RTX 4090 48 GB | `ref2va-turbo8-exact-sm89` | 8 | Resident in GPU memory |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` | 4 | Streamed from host memory |
| RTX 4090 48 GB | `t2va-turbo4-exact-sm89` | 4 | Resident; validation scope below |
| Two RTX 3080 20 GB GPUs | `t2va-turbo4-exact-sm86` | 4 | Streamed; `sequence-head` only |

The HTTP service defaults to Turbo4 on a 4090. To change profiles, restart the service with the selected profile and its matching assets.

```bash
vflash profiles
vflash plan ref2va-turbo8-exact-sm89 --gpu 0
```

## Text-to-video {#t2va}

Since **0.2.0**, Vflash supports complete T2VA generation with `t2va-turbo4-exact-sm89`. [Prepare the Base4 assets](../reference/pipeline-profiles), then omit reference images from the request.

Use a Base4 v1.0 artifact, Base auxiliary tensors, a 6/3 video/audio schedule, and a T2VA bundle with no references. A Ref2VA artifact cannot process this task. Switching mode requires a separate session and matching assets.

The actual SM89 T2VA container repeated one fixed request, matching all 14 official conditioning tensors and both final FP32 AV latents. Both five-second videos and audio tracks fully decoded; repeated video frames matched and cancellation released the owner. Raw official audio VAE output can vary slightly. These are implementation and lifetime checks, not broad instruction or audio-quality qualification. T2VA Turbo8 and newer Base4 adapters remain outside this profile.

Version 0.3.2 adds `t2va-turbo4-exact-sm86` for exactly two 3080s and `sequence-head`. Compile Base-specific SM86 assets and provide a compatible T2 conditioning bundle to the native session. Application-owned stages completed five-second 928 × 512 and ten-second 640 × 352 requests, but the standalone five-second `H3Pipeline` wrapper has not been rerun for this profile. These two geometries do not qualify arbitrary larger inputs. See [the release evidence](../reference/releases#v0-3-2).

## Memory and deployment {#memory}

| Hardware | Memory strategy | What to consider |
| --- | --- | --- |
| One 4090 48 GB | Resident weights by default | Reuse weights across requests; leave room for activations |
| One or two 3080 20 GB GPUs | Blocks streamed from host RAM | Full weights remain in system memory; a pair cooperates on one request |
| One 4090 needing more VRAM headroom | Explicit `block-ring` | Trade host RAM for device headroom and measure the latency cost |

Use `--weight-residency block-ring` in the CLI or the matching [Python session option](./python#memory). The HTTP service uses the selected profile's default strategy.

For the measured 928 × 512, 124-model-frame, four-step workload, allow **at least 64 GiB of available system memory per worker**, plus headroom for the OS and other processes. With block streaming, VRAM usage excludes the complete weights stored in host RAM. Larger inputs and concurrent workers require their own capacity checks.

That budget covers the native denoiser. Complete Ref4 and T2VA pipelines also own encoders and official VAEs; their integration checks used a 240 GiB host-memory limit. The native core uses block streaming. SM89 stages take turns on one GPU; dual-SM86 encoding and decoding use the primary while both GPUs denoise. The representative three-reference SM86 check reached 114.67 GiB host RSS and 11.05/6.85 GiB device-wide use; leave headroom beyond this one-request measurement.

The native Ref4 latent interface supports `sequence-head` or `tensor` on a 3080 pair; T2VA and complete generation require `sequence-head`. The caller selects the peer explicitly. The measured topology uses PCIe 3.0 x16 host-bridge links without peer access and does not require NVLink. See [dual-GPU setup](./getting-started#parallel) and the [measurement scope](../reference/benchmarks#sm86-parallel).

This support scope covers the 20 GB 3080 and 48 GB 4090 variants. Other capacities, models, larger GPU groups and arbitrary resolutions or frame counts have not received the same validation.

## Turbo LoRA support {#lora}

Vflash runs the pinned [LightX2V H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) adapters without the LightX2V inference framework. The profile's adapter, schedule, and step count must match the compiled resources.

| Adapter | Supported hardware | Upstream file |
| --- | --- | --- |
| Ref Turbo4 v0.1 | One/two 3080s, one 4090 | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` |
| Ref Turbo8 v1.0 768p | One 4090 | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |

Pinned revisions and sources are listed under [runtime assets](../reference/runtime-assets#versions). T2VA also supports `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` for T2VA on SM89 or a cooperating SM86 pair, with alpha 128 / rank 128. Only the named files and revisions are supported. ComfyUI, FL2VA, custom LoRAs and future upstream revisions need separate integration.

Turbo4 and Turbo8 are distilled configurations. Fewer steps reduce compute, but they are not a promise of the same output quality as the 50-step base model. The `exact` name refers to the attention path and the selected adapter's execution; it does not promise identical tensors across different GPUs.

The two-device modes have completed full trajectories and one paired decoded-video/audio smoke. They retain exact attention but introduce different floating-point rounding; they are not a same-output or same-quality guarantee.

The single-device 3080 profile has been checked for capacity and repeatable results between serial and overlapped loading on the same GPU. Independent reference comparison and broader quality evaluation are still pending.

## What is outside this release {#scope}

The [complete profile table](../reference/pipeline-profiles) covers Ref4 on SM89 or two SM86 GPUs, and T2VA Base4 on SM89. Single-SM86 and Turbo8 remain native bundle-to-latents interfaces. Unlisted modes, adapters, quantization and temporal settings are outside these qualified profiles. The HTTP API provides one serial denoising lane; account management, billing and distributed GPU scheduling belong to the application.
