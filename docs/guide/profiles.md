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

The HTTP service defaults to Turbo4 on a 4090. To change profiles, restart the service with the selected profile and its matching assets.

### Experimental mixed FFN-in weights

The source tree includes `ref2va-turbo4-mixed-ffnin31-sm89`, an opt-in native
bundle-to-latents profile for one SM89 GPU. It quantizes the main FFN input
projection in 31 layers and their activations to INT8. The other 19 layers,
remaining matrices and LoRA operands stay BF16; attention and the four-step
schedule are unchanged. BF16 remains the default.

This profile needs the existing full BF16 artifact plus a separately prepared
weight sidecar, Linux CPython 3.11,
Torch 2.11.0 with CUDA 13.0, and `comfy-kitchen==0.2.31` from the `w8` extra.
It uses the native provider directly and rejects a different binary. There is
no automatic conversion, model download, HTTP exposure or complete-pipeline
selection for this profile.

```python
session = NativeEngineSession(
    plan,  # Resolve ref2va-turbo4-mixed-ffnin31-sm89 on one SM89 GPU.
    artifact=artifact,
    schedule_overlay=schedule,
    auxiliary_tensor=auxiliary,
    mixed_ffn_in=sidecar_directory,
)
```

The CLI equivalent adds `--mixed-ffn-in SIDECAR` to `vflash denoise` with this
profile. Two-slot streaming is required. Replaced BF16 matrices are skipped
during loading; one pinned INT8 representation is kept on the host. The two
device slots retain enough capacity for the unquantized layers.
The selected BF16 weights still exist on disk; this is not a standalone reduced
model package. Hardware selection remains limited to the existing 48 GB SM89 target.

The [sidecar format](../reference/runtime-assets#mixed-ffnin) binds each weight file to the base artifact.

The underlying loader preserved a tested mixed-precision trajectory exactly;
that comparison was against the previous INT8 implementation, **not BF16**.
Independent Ref4 image tests found motion differences. The source integration
has CPU ownership and dispatch checks, but has not yet been run on a GPU as
this assembled public profile. No general quality or end-to-end speed claim
is made. SM86, T2VA, Ref8, video references and broader input settings need
separate qualification.

```bash
vflash profiles
vflash plan ref2va-turbo8-exact-sm89 --gpu 0
```

## Text-to-video {#t2va}

Since **0.2.0**, Vflash supports complete T2VA generation with `t2va-turbo4-exact-sm89`. [Prepare the Base4 assets](../reference/pipeline-profiles), then omit reference images from the request.

Use a Base4 v1.0 artifact, Base auxiliary tensors, a 6/3 video/audio schedule, and a T2VA bundle with no references. A Ref2VA artifact cannot process this task. Switching mode requires a separate session and matching assets.

The actual T2VA container repeated one fixed request, matching all 14 official conditioning tensors and both final FP32 AV latents. Both five-second videos and audio tracks fully decoded; repeated video frames matched and cancellation released the owner. Raw official audio VAE output can vary slightly. These are implementation and lifetime checks, not broad instruction or audio-quality qualification. T2VA Turbo8, newer Base4 adapters and other GPU targets remain outside this profile.

## Memory and deployment {#memory}

| Hardware | Memory strategy | What to consider |
| --- | --- | --- |
| One 4090 48 GB | Resident weights by default | Reuse weights across requests; leave room for activations |
| One or two 3080 20 GB GPUs | Blocks streamed from host RAM | Full weights remain in system memory; a pair cooperates on one request |
| One 4090 needing more VRAM headroom | Explicit `block-ring` | Trade host RAM for device headroom and measure the latency cost |

Use `--weight-residency block-ring` in the CLI or the matching [Python session option](./python#memory). The HTTP service uses the selected profile's default strategy.

For the measured 928 × 512, 124-model-frame, four-step workload, allow **at least 64 GiB of available system memory per worker**, plus headroom for the OS and other processes. With block streaming, VRAM usage excludes the complete weights stored in host RAM. Larger inputs and concurrent workers require their own capacity checks.

That budget covers the native denoiser. Complete Ref4 and T2VA pipelines also own encoders and official VAEs; their integration checks used a 240 GiB host-memory limit. The native core uses block streaming. SM89 stages take turns on one GPU; dual-SM86 encoding and decoding use the primary while both GPUs denoise. The representative three-reference SM86 check reached 114.67 GiB host RSS and 11.05/6.85 GiB device-wide use; leave headroom beyond this one-request measurement.

The native latent interface supports `sequence-head` or `tensor` on a 3080 pair; complete generation requires `sequence-head`. The caller selects the peer explicitly. The measured topology uses PCIe 3.0 x16 host-bridge links without peer access and does not require NVLink. See [dual-GPU setup](./getting-started#parallel) and the [measurement scope](../reference/benchmarks#sm86-parallel).

This support scope covers the 20 GB 3080 and 48 GB 4090 variants. Other capacities, models, larger GPU groups and arbitrary resolutions or frame counts have not received the same validation.

## Turbo LoRA support {#lora}

Vflash runs the pinned [LightX2V H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) adapters without the LightX2V inference framework. The profile's adapter, schedule, and step count must match the compiled resources.

| Adapter | Supported hardware | Upstream file |
| --- | --- | --- |
| Ref Turbo4 v0.1 | One/two 3080s, one 4090 | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` |
| Ref Turbo8 v1.0 768p | One 4090 | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |

Pinned revisions and sources are listed under [runtime assets](../reference/runtime-assets#versions). T2VA also supports `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` for T2VA on SM89, with alpha 128 / rank 128. Only the named files and revisions are supported. ComfyUI, FL2VA, custom LoRAs and future upstream revisions need separate integration.

Turbo4 and Turbo8 are distilled configurations. Fewer steps reduce compute, but they are not a promise of the same output quality as the 50-step base model. The `exact` name refers to the attention path and the selected adapter's execution; it does not promise identical tensors across different GPUs.

The two-device modes have completed full trajectories and one paired decoded-video/audio smoke. They retain exact attention but introduce different floating-point rounding; they are not a same-output or same-quality guarantee.

The single-device 3080 profile has been checked for capacity and repeatable results between serial and overlapped loading on the same GPU. Independent reference comparison and broader quality evaluation are still pending.

## What is outside this release {#scope}

The [complete profile table](../reference/pipeline-profiles) covers Ref4 on SM89 or two SM86 GPUs, and T2VA Base4 on SM89. Single-SM86 and Turbo8 remain native bundle-to-latents interfaces. Unlisted modes, adapters, quantization and temporal settings are outside these qualified profiles. The HTTP API provides one serial denoising lane; account management, billing and distributed GPU scheduling belong to the application.
