# Profiles and hardware

A profile fixes the model, LoRA revision, step count and arithmetic; its default memory strategy follows the
selected hardware. Choose one that matches both your GPUs and your compiled assets. Vflash never substitutes
devices or attention backends automatically, and a catalog entry or CUDA probe is not proof that a workload
fits or meets quality needs.

```bash
vflash profiles
vflash plan ref2va-turbo8-exact-sm89 --gpu 0
```

## Available profiles {#available}

Ref2VA generates from reference images, T2VA from text only, and the keyframe profiles serve I2VA (first
frame), L2VA (last frame) and FL2VA (both). Base and Ref weights are separate. The
[complete pipeline](./complete-pipeline) accepts these inputs with a matching fixed profile; the native
interfaces use compiled conditioning bundles and may default to a different residency.

| GPU | Profile | Steps | Weight loading |
| --- | --- | ---: | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89` | 4 | Resident |
| RTX 4090 48 GB | `ref2va-turbo8-exact-sm89` | 8 | Resident |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` | 4 | Streamed from host memory |
| RTX 4090 48 GB | `t2va-turbo4-exact-sm89` | 4 | Resident |
| Two RTX 3080 20 GB | `t2va-turbo4-exact-sm86` | 4 | Streamed; `sequence-head` |
| RTX 4090 48 GB | `i2va-base16-bf16-sm89`, `fl2va-base16-bf16-sm89` | 16 | Resident |
| Two RTX 3080 20 GB | `i2va-base16-bf16-sm86`, `fl2va-base16-bf16-sm86` | 16 | Streamed; `sequence-head` |
| RTX 4090 48 GB | LightX v0.1 / v1.0 544p keyframe pairs (`i2va-turbo4-v01-544-exact-sm89`, …) | 4 / 8 | See [pipeline profiles](../reference/pipeline-profiles#lightx-v01) |

Matching keyframe pairs share artifacts, so one loaded `H3Pipeline` serves I2VA, L2VA and FL2VA serially;
L2VA is a request contract, not a separate weight profile. The HTTP service loads one profile (Turbo4 on a
4090 by default); restart it with matching assets to change profiles.

## Supported GPUs {#heterogeneous}

| GPU | Paths | Notes |
| --- | --- | --- |
| RTX 4090 48 GB (SM89) | Every profile above; original-v0.1 hybrid with dense or Veda attention; [native HD up to a 2720 px long edge](./complete-pipeline#modes-and-limits); [extended 5–15 s / nine-image hybrid](./complete-pipeline#extended-hybrid) | The reference platform. The measured devices report 48 GiB; 24 GiB cards are not established for the same workloads. |
| Two RTX 3080 20 GB (SM86) | Ref4 native (`sequence-head` or `tensor`) and complete (`sequence-head`); T2VA and Base16 keyframes with `sequence-head` | PCIe 3.0 x16 without peer access; no NVLink required. |
| One RTX 3080 20 GB | Ref4 native latents | Capacity and serial/overlapped repeatability checked. |
| H100 SXM 80 GB (SM90) | Original-v0.1 hybrid I2VA with `veda-triton`; [resident trunk](#h100-resident) at ≥75 GiB | Complete-video evidence; H100 PCIe remains a preview. |
| RTX PRO 6000 Blackwell 96 GB (SM120) | Hybrid I2VA with `veda-triton`; [resident trunk](#sm120-resident) at ≥90 GiB (Server) | Workstation is a measured I2VA option without separate residency measurement. |
| H100 NVL, RTX 5090 32 GB | [Bounded block-ring I2VA](#small-blackwell) | Keep geometry within individually measured cases. |
| SM103, MIG partitions, new-device FL2VA | Preview only | No complete-device qualification. |

For the original-v0.1 hybrid on SM90/SM120, provision at least 128 GB of host RAM with headroom; device
memory alone does not determine compatibility.

## Memory and residency {#memory}

| Strategy | Where weights live | Use when |
| --- | --- | --- |
| Resident (default on one 4090 for native profiles) | GPU | Reusing weights across requests with room for activations |
| `block-ring` (default for the complete pipeline core and SM86) | Host RAM, streamed in blocks | Trading host RAM for device headroom; required on 20 GB cards |
| Explicit `weight_residency="resident"` on SM90/SM120 | GPU trunk | Sustained warm batches or lower host memory (see below) |

Select block ring with `--weight-residency block-ring` or the [Python session option](./python#memory). For the
measured 928×512, four-step native workload allow **at least 64 GiB of available system memory per worker**.
Complete pipelines also own encoders and official VAEs: their integration checks used a 240 GiB host limit, and
a representative three-reference SM86 request reached 114.67 GiB host RSS. These are tested budgets, not minima;
larger inputs and concurrent workers need their own checks. Paired SM86 encoding and decoding use the primary GPU
while both denoise ([dual-GPU setup](./getting-started#parallel), [measurement scope](../reference/benchmarks#sm86-parallel)).

<a id="sm120-resident"></a>

### Resident trunk on H100 and Blackwell {#h100-resident}

`H3Pipeline(..., weight_residency="resident")` keeps the trunk on one SM90 GPU with at least 75 GiB VRAM or
one SM120 GPU with at least 90 GiB. It requires the original-v0.1 `HybridModel`, `attention_backend="veda-triton"`
and `PYTORCH_ALLOC_CONF=expandable_segments:True` set **before the Python process starts**. Residency covers
I2VA with one first frame for up to 15 seconds. On SM90 the canvas is at most 1536×864 pixels; on 96 GiB SM120 it
follows the native-HD budget `width × height × model_frames ≤ 2048² × 124` (for example 2720×1536 for 5 s or
1920×1088 for 10 s, at a measured 42.1 GiB peak with the FP8 trunk). Other modes and larger requests use the block
ring, which remains the default and the rollback.

| Same-instance A/B/A2 | H100 SXM ring / resident / ring | RTX PRO 6000 Server ring / resident / ring |
| --- | --- | --- |
| Initialization (s) | 119.0 / 128.2 / 115.3 | 105.2 / 123.3 / 108.2 |
| Warm 512² / 5 s (s) | 27.1 / 17.0 / 19.6 | 21.0 / 15.7 / 19.2 |
| 1536×864 / 15 s (s) | — / 227.3 / — | 256.1 / 245.3 / 253.7 |
| Peak host RSS (GiB) | 103.0 / 62.9 / 102.9 | 108.7 / 68.7 / 108.7 |
| Denoising allocation peak (GiB) | 70.3 (resident, 15 s) | 70.3 (resident) |

Residency shortens warm requests but costs more at startup; about five to six warm short requests amortize the
difference in these samples. Choose it for sustained batches or lower host memory, judged on total startup plus
batch time.

### Bounded NVL and RTX 5090 cases {#small-blackwell}

| GPU | Completed I2VA cases (generation seconds) | Limits |
| --- | --- | --- |
| H100 NVL | 1920×1088/10 s 288.1, 1536×864/15 s 263.9, 2048²/5 s 321.6, 1440²/10 s 289.9 | Initialization 46.6 s; other modes unqualified |
| RTX 5090 32 GB / 600 W | 2048²/5 s 451.0 (27.0 GiB peak), 1440²/10 s 392.4, 1536×864/15 s 364.9 | Block ring with expandable segments; a 500 W card failed 2048²/5 s |

These are capacity results with complete decoding and multi-time visual review, not quality or routing defaults.

### Container device discovery

Set `VFLASH_DEVICE_DISCOVERY=cuda-visible` in partitioned or container environments. An isolated CUDA subprocess
matches the visible UUID and **partition memory** against NVIDIA enumeration and never substitutes the parent
card's capacity. Discovery is opt-in and fails explicitly.

### Portable preparation for another architecture

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
prepare(Path("/models/hybrid-sm120"), artifact, overlay, IndexedCheckpoint(Path("/models/transformer_ref")))
```

Pass the new receipt and `HybridModel(Path("/models/hybrid-sm120"))` to the pipeline with
`attention_backend="veda-triton"`. Keep the ingested source mounted read-only. Portable preparation records the
original compile target, creates a directory view plus small schedule copies and does not hash giant weights; it
must not hardlink serving files, because changing link counts invalidates existing receipts.

## Text-to-video {#t2va}

`t2va-turbo4-exact-sm89` generates from text alone with Base4 v1.0 artifacts, Base auxiliary tensors, a 6/3
video/audio schedule and a bundle without references ([prepare the assets](../reference/pipeline-profiles)).
A Ref2VA artifact cannot run this task. `t2va-turbo4-exact-sm86` runs on exactly two 3080s with `sequence-head`;
application-owned stages completed 928×512/5 s and 640×352/10 s, which do not qualify larger inputs. On SM89 a
repeated fixed request matched all 14 conditioning tensors and both final FP32 latents.

## Turbo LoRA support {#lora}

Vflash runs the pinned [LightX2V H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) adapters without the
LightX2V framework. The adapter, schedule and step count must match the compiled resources.

| Adapter | Hardware | Upstream file |
| --- | --- | --- |
| Ref Turbo4 v0.1 | One/two 3080s, one 4090 | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` |
| Ref Turbo8 v1.0 768p | One 4090 | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |
| Keyframe Turbo4 v0.1 544p | One 4090 | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` |
| Keyframe Turbo8 v1.0 544p | One 4090 | `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` |
| T2VA Turbo4 v1.0 768p | SM89 or an SM86 pair (alpha 128 / rank 128) | `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` |

Pinned revisions are under [runtime assets](../reference/runtime-assets#versions). Base16 keyframe profiles have no
adapter. Only the named files and revisions are supported. Turbo4/8 are distilled: fewer steps reduce compute but
do not promise 50-step base quality, and `exact` names the attention path, not identical tensors across GPUs. Paired
devices keep exact attention with different floating-point rounding.

## Outside the supported profiles {#scope}

The [complete profile table](../reference/pipeline-profiles) owns each profile's media-validation scope. Single-SM86
and Ref Turbo8 expose native bundle-to-latents interfaces. Unlisted modes, adapters, quantization, temporal settings,
GPU capacities and larger device groups are outside these profiles. The HTTP API provides one serial denoising lane;
accounts, billing and distributed scheduling belong to the application.
