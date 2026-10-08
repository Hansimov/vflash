# Profiles and hardware

## Explicit H100 trunk residency (0.6.14) {#h100-resident}

`H3Pipeline(..., weight_residency="resident")` can retain the trunk on a single
SM90 GPU with at least 75 GiB VRAM. It requires the original-v0.1 `HybridModel`,
`attention_backend="veda-triton"`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`
set **before starting the Python process**. The default remains `block-ring`.
The measured request envelope is I2VA with one first frame, area at most
1536 × 864 pixels and at most 15 seconds. Other modes and larger canvases must
use the block ring. The separate SM120 qualification is described below; H100
measurements do not qualify untested architectures.

A same-H100-SXM A/B/A2 experiment used separate processes, the same immutable
runtime/model/reference/seed, four updates and expandable segments in every arm.
Per-file eviction hints did not guarantee a cold remote storage server. Seven
complete videos passed decode and playback checks. Aggregate evidence from the
private integration at source `575f7986`:

| Observation | Block ring A | Resident B | Block ring A2 |
| --- | ---: | ---: | ---: |
| Initialization, seconds | 119.049 | 128.154 | 115.268 |
| First 512² five-second output | 156.137 | 124.491 | 126.772 |
| Warm 512² five-second output | 27.085 | 16.992 | 19.560 |
| Peak host RSS for short outputs, GiB | 102.956 | 62.941 | 102.948 |

Against the more favorable return control, warm latency improved 13.1%, but
initialization plus first output took 10.6 seconds longer. Five subsequent warm
requests are needed to recover that startup difference in this limited sample.
This is useful for sustained small-request batches or reducing host memory, not
an unconditional cold-start speedup. Host CPU, storage and compile state remain
part of the result; a single rental does not establish an SLA.

The resident arm additionally completed 1536 × 864 / 15 seconds in 227.337 seconds,
with 68.180 GiB peak host RSS and a **70.344 GiB denoising allocation peak**.
The post-decode counter was only 59.200 GiB and must not replace the earlier peak.
Without expandable segments, the prior resident trial failed this larger case.
No matched long-request speedup is claimed. Multiple-time visual review found the
same background/style drift already present in the block-ring baseline; no quality
improvement is claimed, and audio semantics were not independently reviewed.
Return to `weight_residency="block-ring"` for rollback.

## Full-memory Blackwell residency (0.6.17) {#sm120-resident}

The same explicit resident option now also admits a **single SM120 GPU with at
least 90 GiB VRAM**. Complete-video evidence is from RTX PRO 6000 Blackwell Server
96 GB; Workstation residency has not been separately measured. RTX 5090 and
48 GB MIG partitions remain outside this residency contract. The hybrid/Veda,
allocator and I2VA request limits above still apply; the default remains block ring.

A same-instance A/B/A2 comparison used separate processes, identical frozen inputs,
compact assets, four updates, seed and expandable allocator. The host provided
approximately 251 GB RAM and 16 vCPUs. Storage eviction hints do not establish
remote-server cold state. All nine complete outputs decoded and played successfully.
Aggregate evidence from private integration source `2f0444f7`:

| Seconds unless noted | Block ring A | Resident B | Block ring A2 |
| --- | ---: | ---: | ---: |
| Initialization | 105.228 | 123.279 | 108.208 |
| First 512² / 5 s | 77.668 | 86.367 | 83.024 |
| Warm 512² / 5 s | 21.015 | 15.693 | 19.168 |
| 1536 × 864 / 15 s | 256.104 | 245.326 | 253.650 |
| Whole three-output arm | 469.730 | 474.786 | 471.519 |
| Peak host RSS, GiB | 108.739 | 68.718 | 108.710 |

Against the faster return control, warm short requests improved **18.1%** and the
long request **3.3%**. Initialization plus first output cost 18.4 seconds more;
approximately six subsequent short requests amortize that difference. This tested
three-output batch itself took 3.3 seconds longer. Choose residency for sustained
warm batches or lower host memory, using total startup and batch time for deployment.
Denoising allocation peaked at **70.324 GiB**, versus 59.177 GiB after decoding.
Five-time-point visual review retained the baseline's unwanted zoom/tracking, with
no observed new material visual regression. No general quality or audio-semantic
improvement is claimed. Return to block ring for rollback. No weights or private
media are distributed with this release.

## H100 and Blackwell preview (0.6.13) {#heterogeneous}

Original-v0.1 I2VA with explicit hybrid modulation and `veda-triton` has complete
video evidence on **H100 SXM 80 GB (SM90)** and **RTX PRO 6000 Blackwell Server/Workstation
96 GB (SM120)**. Existing `veda-sm89` names, defaults and behavior are preserved.
SM103, RTX 5090, H100 NVL/PCIe, MIG partitions and new-device
FL2VA are explicit previews without complete-device qualification. A catalog
entry or CUDA probe is not proof that a workload fits or meets quality needs.

The frozen four-step diagnostic uses BF16 weights, two-slot block streaming,
Veda Triton INT8 sparse attention and 24 fps. Same-input warm total seconds:

| Request | PRO Server | H100 SXM |
| --- | ---: | ---: |
| 512² / 5 s | 19.969 | 25.403 |
| 928×512 / 5 s | 29.534 | 28.949 |
| 1280×704 / 5 s | 51.855 | 48.656 |
| 1536² / 5 s | 144.122 | 135.542 |
| 512² / 15 s | 43.624 | 52.007 |

Initialization took 169.575 / 147.109 seconds; first cold generation 112.538 /
110.788 seconds, respectively, excluding container provisioning. Peak host RSS
was 106.7 / 106.2 GiB. Provision at least 128 GB host RAM with headroom; device
memory alone does not determine compatibility. Six videos per card passed full
AV decoding, dimensions, frame rate and duration checks. Multi-time frame review
found background drift and imperfect motion/shape adherence; audio semantics
were not independently auditioned. These are not broad quality guarantees.

A separate product path with aspect-preserving cover preprocessing and decoded
keyframes on PRO Server measured 18.552, 24.996, 48.441, **199.668**, and 41.905
seconds for those warm cases. Different host CPU allocation and processing make
this an integration measurement, not a controlled speedup/regression comparison.
The 1536² product media stage alone took 92.540 seconds. End-to-end decisions must
include conditioning, media, initialization and the actual caller's preprocessing.
A follow-up product-path batch with the public core measured Server/Workstation
warm times of 17.885/21.192, 30.097/43.778, 63.470/62.591, 181.772/220.547 and
45.182/52.446 seconds for the same five geometries/durations. Hosts differed
(188 GB / 32 vCPU versus 282 GB / 16 vCPU), so these are whole-instance results,
not isolated GPU speed ratios. Provisioning through first output took
336.573/372.868 seconds. Both batches fully decoded and received six-time-point
visual inspection per clip; background/cloud drift and unassessed audio semantics
remain. Workstation is an additional measured I2VA option, not a faster default.

Aggregate evidence originated in product experiments through `4322801e`;
no product accounts, rental logic, prompts, media or machine configuration ship here.

Use `VFLASH_DEVICE_DISCOVERY=cuda-visible` in partitioned/container environments.
An isolated CUDA subprocess matches the visible UUID and **partition memory**
against NVIDIA enumeration; it never substitutes the parent card's capacity.
Discovery is opt-in, and failure is explicit. Tested MIG identity mapping is CPU
contract evidence only, not an actual MIG inference benchmark.

CPU preparation can happen before a GPU is rented:

```python
from pathlib import Path
from vflash.pipeline.assets import load_prepared_pipeline_assets
from vflash.pipeline.portable import prepare_portable_assets
from vflash.native.h3_runtime_artifact import load_h3_runtime_artifact
from vflash.native.h3_schedule_overlay import load_h3_schedule_overlay
from vflash.native.h3_prepared_hybrid import prepare
from vflash.adapters.checkpoints import IndexedCheckpoint

source = load_prepared_pipeline_assets(Path("/models/prepared-sm89.json"))
prepared = prepare_portable_assets(source.assets, Path("/models/sm120-view"), "sm120")
artifact = load_h3_runtime_artifact(prepared.assets.artifact, verify_content_hashes=False)
overlay = load_h3_schedule_overlay(prepared.assets.schedule_overlay, artifact=artifact)
prepare(Path("/models/hybrid-sm120"), artifact, overlay,
        IndexedCheckpoint(Path("/models/transformer_ref")))
```

Pass the new prepared receipt and `HybridModel(Path("/models/hybrid-sm120"))`
to the existing pipeline, explicitly selecting `attention_backend="veda-triton"`.
Keep the ingested source mounted read-only. Portable preparation uses existing
trusted inventories, records the original compile target and source manifest,
creates a directory view plus small schedule copies, and does not hash giant
weights. It must not hardlink serving files: changing link counts changes ctime
and invalidates old receipts. CPU-derived tables are checked separately by digest,
shape, dtype, finite values and backbone/schedule binding. No weights are bundled.
Source/wheel release only; select a compatible CUDA/Torch/Triton runtime explicitly.


A profile chooses the model, LoRA revision, step count, and arithmetic. Its default memory strategy follows the selected hardware. Choose one that matches both your hardware and compiled assets.

## Available profiles {#available}

Ref2VA generates video and audio from reference conditioning. T2VA uses text conditioning without reference images. I2VA anchors frame zero, L2VA anchors only the final frame, and FL2VA anchors both ends of a clip. Base and Ref weights are separate. The [complete pipeline](./complete-pipeline) accepts these inputs with a matching fixed profile. The native interfaces below use compiled conditioning bundles; their default residency differs from the complete pipeline.

| GPU | Profile | Steps | Weight loading |
| --- | --- | ---: | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89` | 4 | Resident in GPU memory |
| RTX 4090 48 GB | `ref2va-turbo8-exact-sm89` | 8 | Resident in GPU memory |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` | 4 | Streamed from host memory |
| RTX 4090 48 GB | `t2va-turbo4-exact-sm89` | 4 | Resident; validation scope below |
| Two RTX 3080 20 GB GPUs | `t2va-turbo4-exact-sm86` | 4 | Streamed; `sequence-head` only |
| RTX 4090 48 GB | `i2va-base16-bf16-sm89` | 16 | Resident |
| Two RTX 3080 20 GB GPUs | `i2va-base16-bf16-sm86` | 16 | Streamed; `sequence-head` |
| RTX 4090 48 GB | `fl2va-base16-bf16-sm89` | 16 | Resident |
| Two RTX 3080 20 GB GPUs | `fl2va-base16-bf16-sm86` | 16 | Streamed; `sequence-head` |

The HTTP service defaults to Turbo4 on a 4090. To change profiles, restart the service with the selected profile and its matching assets.

Base16 and the [544p keyframe Turbo pairs](../reference/pipeline-profiles#lightx-v01) share artifacts within each matching pair, so one loaded `H3Pipeline` accepts I2VA, L2VA and FL2VA serially without a profile restart. L2VA is a request contract, not a duplicate weight profile. The prepared profile ID remains its provenance identity. This does not permit cross-version artifacts or apply to Ref2VA and T2VA.

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
| Keyframe Turbo4 v0.1 544p (preview) | One 4090; bounded I2VA/L2VA/FL2VA media evidence | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` |
| Keyframe Turbo8 v1.0 544p (preview) | One 4090 | `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` |

Pinned revisions and sources are listed under [runtime assets](../reference/runtime-assets#versions). T2VA also supports `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` for T2VA on SM89 or a cooperating SM86 pair, with alpha 128 / rank 128. Base16 keyframe profiles have no adapter; the 544p Turbo pairs above use rank128 / alpha8 and shifts12 / 3. Only the named files and revisions are supported. ComfyUI, custom LoRAs and future upstream revisions need separate integration.

Turbo4 and Turbo8 are distilled configurations. Fewer steps reduce compute, but they are not a promise of the same output quality as the 50-step base model. The `exact` name refers to the attention path and the selected adapter's execution; it does not promise identical tensors across different GPUs.

The two-device modes have completed full trajectories and one paired decoded-video/audio smoke. They retain exact attention but introduce different floating-point rounding; they are not a same-output or same-quality guarantee.

The single-device 3080 profile has been checked for capacity and repeatable results between serial and overlapped loading on the same GPU. Independent reference comparison and broader quality evaluation are still pending.

## What is outside this release {#scope}

The [complete profile table](../reference/pipeline-profiles) owns each profile's current media-validation scope, including the 544p keyframe previews. Single-SM86 and Ref Turbo8 retain native bundle-to-latents interfaces; the 544p keyframe Turbo8 pair also has a complete pipeline. Unlisted modes, adapters, quantization and temporal settings are outside these profiles. The HTTP API provides one serial denoising lane; account management, billing and distributed GPU scheduling belong to the application.
