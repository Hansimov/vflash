# Pika H3 Flex reference integration

`PikaFlexRuntime` adapts the original [Pika release](https://github.com/Pika-Labs/Pika-H3-Flex)
without changing the hybrid or segmented pipeline defaults. This initial adapter preserves the
upstream staged BF16 implementation. The staged path has completed 5-second 480p text and first-frame generation on a
96 GiB Blackwell PRO 6000. Other modes, larger canvases and serving integration remain
unqualified; CPU contract checks do not establish GPU output quality or service availability.

Use a separate Python 3.12 environment with the upstream pinned requirements and its two
Diffusers patches. The existing `vflash[pipeline]` stack pins different dependencies: do not
install both stacks into the same environment. Install vflash itself without its pipeline extra.
The source checkout is pinned to `4bf8ea508891a432106e70f675d287c27f02306d`; its 22 Python/YAML
files are checked against the packaged inventory before import. No code or models are downloaded
by this adapter. Explicit local-code trust is required.

```python
from pathlib import Path
from vflash.pipeline.pika_flex import PikaFlexRequest
from vflash.pipeline.pika_flex_runtime import PikaFlexRuntime

runtime = PikaFlexRuntime(
    source_root=Path("Pika-H3-Flex"),
    base_path=Path("models/MiniMax-H3"),
    adapters_root=Path("models/Pika-H3-Flex"),
    trust_local_code=True,
)
try:
    result = runtime.generate(
        PikaFlexRequest(prompt="integrated_multimodal_description: [Shot 1] ...\n"
                               "overall_soundscape: ...\nnon_diegetic_music: N/A"),
        Path("outputs/take.mp4"),
    )
finally:
    runtime.close()
```

Assets use MiniMax H3 revision `42ed227ee7df40d41602854ae760620d6eb651fe` and Pika adapter
revision `0ddf55ad7067ca1e76812253bd7b4600df87f759`. Each available adapter directory is named
`fl2va` or `ref2va` and contains its original `manifest.json` and `adapter.safetensors`.
FL handles text, first, last and both-frame input; Ref handles ordered content references.
The upstream executor verifies the publisher's adapter hash before loading. Its metadata records
local base overrides; the application must retain the base download/conversion provenance.

The release always generates video and audio jointly, with six model evaluations. It uses
knots `[1000,750,500,250,100,40,0]`, shifts 12/3 and fresh noise after the first five clean
predictions. A six-step Euler schedule is a different algorithm. No Larry audio pass is added.
FL uses BF16 conditioning and Ref uses FP32 conditioning. The adapters have rank and alpha 128.

The local release exposes 480p/768p buckets, eleven aspect buckets, integer 5–15-second
requests and 24 fps. Output retains the model's aligned `17n+5` frame clock, so a five-second
request emits 124 frames. Application trimming/cropping and display geometry must be recorded
separately. This adapter does not claim native 4K, arbitrary canvases, separate audio stems or
seamless extension. A failed request poisons this runtime; tear down its dedicated process
before retrying. Existing output files are never overwritten.

Pika code is Apache-2.0. The adapters are Model Derivatives under the MiniMax H3 Community
License Agreement. Keep the upstream LICENSE, NOTICE, modification notices and model licenses
with any distributed assets. The integration contains release identities, not model weights.

## Opt-in component reuse

`reuse_components=True` retains one original BF16 transformer partition and one original FP32
VAE pair. Before text encoding or decoding, the transformer moves to CPU; text weights load
onto their target device without materializing a second full CPU model. Changing FL/Ref releases
the previous trunk. Adapter injection/loading runs once for a retained trunk; changing the
adapter file or configuration rejects reuse. Sampling, conditioning precision and native audio
remain the original six-step computation.

This option is a lifecycle candidate: CPU tests cover reuse, partition changes, exact small-tensor
values, failure cleanup and restored hooks. Full-weight GPU memory, quality and latency validation
is still pending. It cannot be combined with `cpu_offload=True`. Use a dedicated process; all
`PikaFlexRuntime` calls in that interpreter share a nonblocking owner lock. `close()` releases
retained components. Results include `component_reuse` and load/reuse counters in `component_stats`.

The reusable activation adapter also accepts explicit `kind="pika"` with the original release
manifest. It validates all 312 rank-128 pairs, retains the original parameter names, uses
alpha/rank=1 and casts stored FP32 adapter tensors to BF16 as the released PEFT loader does.
It never applies LightX's fallback alpha. Partial installation failures remove all hooks.
This is a loader contract, not qualification of a quantized six-step generation pipeline.

`PikaSamplingClock` consumes separately prepared FL text/keyframe projection tables. It verifies
the original six timestamps, FP32 time embeddings, BF16 modulation and all 312 table entries
before use, and checks each actual model call. The table recipe is independent of a final audio
recipe; changing it never turns the Pika denoiser into an Euler sampler.

The native fixed-picture audio adapter can represent a text-only source with `condition_rows=0`.
It adds the clean-video clock before projection validation, preserves text/audio times and retains
all source video times when using spatial conditioning. It creates no first-frame image. CPU
checks cover row/clock mapping, source-picture preservation and cleanup; complete GPU audio
quality for a Pika/Larry combination still needs its own qualification. The existing first-frame
audio path and the optional original Pika runtime retain their separate identities.


## Prepared Pika picture with native Larry audio

The optional `vflash.pipeline.pika_av_runtime.PikaAVRuntime` reuses the explicit
original-FL row-FP8 components accepted by `ReferenceAVRuntime`. It replaces the
segmented session with one exclusively owned Pika session. Supply `pika_source`,
`pika_adapter`, `pika_manifest` and `pika_tables` in addition to those components;
`trust_local_code=True` validates and imports the pinned Pika sampler source.
No model location, GPU rental or product default is inferred.

```python
from vflash.pipeline import VideoRequest
from vflash.pipeline.pika_av import PikaAVRecipe
from vflash.pipeline.pika_av_runtime import PikaAVRuntime

# All paths, manifests and the text loader are explicit caller-owned components.
runtime = PikaAVRuntime(**prepared_components, **pika_components, trust_local_code=True)
try:
    result = runtime.generate(
        VideoRequest("A ceramic bird unfolds its wings.", width=960, height=544),
        output_directory,
        recipe=PikaAVRecipe(audio_steps=10, visual_attention="veda", overlap_decode=True),
    )
finally:
    runtime.close()
```

This prepared path currently accepts five-second T2VA or I2VA with generated audio
and decoded keyframes. It runs the original six-step re-noising schedule with the
unmerged Pika FL adapter; `audio_steps=0` keeps that joint soundtrack, while 10 or
12 explicitly selects a Larry native soundtrack over the frozen picture. Prompt
and first-frame embeddings are reused within the request. The visual fused-QKV
storage is released before audio and optional video-decode overlap. Execution
counts, attention, conditioning reuse, memory and complete-call timings are returned.

The original BF16 Pika pipeline above remains separate. Prepared row-FP8 weights,
Veda attention and spatial video-KV audio reuse are approximate choices; six visual
forwards plus ten audio forwards is not a six-forward complete request. Current
hardware evidence covers a full 96 GiB SM120 PRO 6000 and 960×544 / 832×480 five-second
clips. Other admitted canvases are not a broad quality or memory qualification.
Reported generation time excludes model initialization and application publication.
No sub-25-second 0.5 MP end-to-end guarantee is made. Ref2VA, longer clips, omitted
joint audio and exact endpoint repainting are not implemented by this prepared path.

### Bounded decoder tile batches

`PikaAVRecipe(decode_tile_batch_size=4)` optionally batches equal-size spatial decoder
tiles for five-second canvases up to 960×544 on the prepared SM120 runtime. The default
remains 1; 2 and 4 are explicit options. Original tile positions, temporal decoding and
overlap composition are preserved, and the instance hook is restored after the request.

Four short-scene trials on one PRO 6000 Server 96 GiB at a 600 W limit retained identical
generated video/audio latents and decoded audio. FP16 batch-dependent decoding and
encoding changed 8-bit pixels slightly (mean absolute differences about 1.3–1.8/255);
inspected temporal frames and detail crops showed no evident new structural degradation.
One repeated-input warm control bracket supported about 0.7–1.1 seconds saved in a
roughly 30-second complete resident call. Other scene pairs also include first-use/cache
differences. Initialization, orchestration and upload are separate; these trials do not
establish a sub-25-second delivered result or qualification above that canvas size.
The response records actual tile batch groups in `decode_tile_batches`.

### Strict rotary execution (source, pending release)

`PikaAVRecipe(strict_rotary=True)` opts into the existing strict BF16 split-half rotary
kernel in both visual and native audio stages. The default is false. This changes execution,
not the sampler, model weights or audio recipe. The scoped hook restores the source function
on success or failure; CPU, other dtypes and gradient-enabled calls use the original function.
Execution receipts count fused calls, reference fallbacks and required contiguous copies.

Four five-second T2VA/I2VA pairs at 960×544 on one 450 W PRO 6000 Workstation, using Pika6,
Larry10, Veda and native VAE tile batches of four, retained identical video latents, all
120 decoded RGB frames and decoded PCM audio. A similarly warmed phase-first control took
52.115 seconds versus 51.462 seconds with fusion. Larger cross-case differences also include
conditioning-cache and warm-up effects, so they are not an isolated rotary speedup. These
complete engine-call times exclude procurement, initialization and application delivery.
The existing generation-quality limitations remain; pixel equality establishes this execution
change's equivalence on the tested cases, not general content quality.
