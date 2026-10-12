# Measuring performance

Measure the path that matters to your application, and keep its scope visible. The complete pipeline produces an MP4; the native denoiser produces latent tensors. Their timing boundaries differ.

## Separate loading from repeated requests {#timing}

Each CLI invocation initializes its own models. A persistent `H3Pipeline` or native HTTP worker reuses its models across requests, so startup, first use and later requests have different costs.

For complete generation, report `H3Pipeline.initialization_seconds` separately from `VideoResult.elapsed_seconds`. The latter covers input preparation, encoding, denoising, media delivery and request cleanup. Its `stages` contain both outer durations and nested details; do not add nested costs twice. See the [pipeline timing contract](../guide/complete-pipeline#one-owned-pipeline).

Native denoiser results expose these values in `session`:

| Field | Meaning |
| --- | --- |
| `request_index` | The request number in this model session, starting at 1 |
| `initialization_seconds` | How long this session took to initialize |
| `initialization_charged_seconds` | Initialization time on the first request; zero on later requests |
| `request_wall_seconds` | Time spent executing the request in the initialized session, including writing its latent output |

For an engine-side first-request cost, add `initialization_charged_seconds` and `request_wall_seconds`. Queue wait, process startup, HTTP transfer, input encoding, and video decoding are outside that sum. Measure those separately when they are part of your application.

`/readyz` passing before the first job means the service can see the configured files and GPU. It does not mean the model is already loaded.

## Profile one denoising request {#denoise-profile}

Both `vflash generate` and `vflash denoise` accept `--profile-denoise`. The flag is deliberately opt-in: it records CUDA timing events for each evaluation and each streamed block, so use an unprofiled run for the final throughput comparison. It does not add an evaluation fence. A complete-pipeline request already fences every cooperating device before publishing each progress event; the profile reports those existing fences separately.

The returned `generation.denoise_profile` separates:

- primary-device row setup, input packing, invocation preparation, denoiser, final layer and latent update;
- per-rank H2D-active time, compute-stream waits for streamed weights, block execution and the copy/compute spans;
- per-evaluation and per-rank totals for AdaLN, attention normalization/modulation, QKV projection,
  Q/K normalization and rotary, attention, attention output, FFN normalization/modulation, FFN input
  and FFN output;
- nested block details for output projection versus gate/residual, and FFN input projection versus
  SiLU-times-gate; adapter-fused and tensor-parallel implementations retain implementation-specific labels;
- sequence/head collective calls, transmitted bytes, host issue time and host `Work.wait()` time;
- for sequence/head attention, QKV packing, inbound dependency waits, QKV unpacking, Flash-SDPA,
  attention-result packing, outbound dependency waits and final unpacking on the compute stream;
- per-evaluation engine submission, existing progress fences, callback time, final fence and profile materialization.

H2D, ready-wait, block-compute and rank-span values describe overlapping CUDA streams and are **not additive**. Block compute includes the collective critical path. Host collective wait is a launch/synchronization diagnostic, not GPU communication duration by itself. Use an external CUDA profiler when kernel-level NCCL attribution is required. The diagnostic output contains device indices and timing/byte counts, not model tensors, prompts, paths or GPU UUIDs.

The sequence/head `inbound_ready_wait` and `outbound_ready_wait` fields measure dependencies that
actually delay the compute stream. They do not measure the full lifetime of NCCL kernels, which can run
on another stream and overlap QKV packing or Flash-SDPA. The nested attention fields are already inside
the outer `attention` phase; do not add them to that phase or to block execution a second time.
Likewise, `block_detail_seconds` is nested inside `block_phase_seconds`; it exists to estimate a
fusion ceiling and must not be added to the outer phase totals.

## Make comparisons useful {#comparisons}

Use the same conditioning bundle, model and adapter revisions, profile, GPU, and output boundary. Report first-use and repeated-request timings separately, with enough runs to show variability.

When comparing memory use, distinguish GPU allocated memory, GPU reserved memory, device-wide usage, and host RAM. In particular, the 3080's streamed weights consume system memory that does not appear in a VRAM-only chart.

Changing from Turbo8 to Turbo4 changes the model's distilled schedule as well as the amount of computation. Label that difference alongside a speed comparison.

## Two-device execution {#parallel}

A cooperating pair can reduce latency for one request; independent workers serve a different throughput goal. Compare both using the same primary GPU, input and timing boundary. Keep one-request latency separate from the total throughput of two devices.

The published [a3 dual-3080 comparison](./benchmarks#sm86-parallel) measured a 1.725× speedup on one fixed workload. It retains its original software version, thermal observations and quality limits; it is not a speed guarantee for the current release or other workloads.

Version 0.4.0 has one controlled SM89 Base16 result: a fixed ten-second, 736 × 992 L2VA request took 774.153 and 773.515 seconds in the bracketing single-device controls, and 484.232 seconds on two devices. The 773.834-second control mean makes the paired reduction 37.4%. All three MP4 outputs were byte-identical. The paired run kept both 450 W cards at or below 78°C without thermal-slowdown samples; the second single-device control also recorded a 65.077°C maximum rolling ten-minute mean. Two serial paired requests would consume 968.464 device-seconds versus a 773.834-second mean makespan for two independent single-card requests, a 25.2% fleet penalty. This evidence qualifies an opt-in idle-peer latency path rather than a throughput default.

### Exact direct relayout boundary {#direct-relayout}

Version 0.4.0 removes materialized layout chains from the two-device `sequence-head` path. Final
unprofiled A/direct/A runs measured **174.776 → 172.149 seconds (1.503%)** for a five-second 512²
I2VA request on dual SM86, and **474.669 → 472.429 seconds (0.472%)** for a ten-second 736 × 992
L2VA request on dual SM89. The corresponding denoising reductions were 1.995% and 0.623%.

A separate profiled SM89 A/direct/A measured a 0.662% complete-request and 0.801% denoising
reduction. It showed QKV packing falling from about 4.51 to 2.13 seconds per rank and returned-head
merge from about 1.38 to 0.67 seconds per rank over 800 profiled blocks. Flash-SDPA remained about
223–235 seconds per rank and did not change, so the layout kernels are not the dominant bottleneck.

At the representative 55,413-token shape, direct QKV packing saved 1,136.4 MiB of transient
allocation and direct returned-head merge saved 378.8 MiB on both SM86 and SM89. Each was a 50%
operation-peak reduction; they are separate operation peaks, not additive pipeline VRAM. Every
operator comparison was element-equal and all complete A/B/A video elementary streams matched.
Repeated official audio decoding varied in both controls and candidates. No large model or media
file was hashed for these checks. See the [Sol-Engine alignment matrix](./sol-engine-alignment).

## Attention activation lifetimes

The normal native block gathers attention and FFN modulation separately and drops dead QKV,
normalization and attention tensors before FFN. GEMM shapes, sampling and model weights are unchanged;
the optional phase-profiler retains its original measurement path.

A single-SM89 48GiB exploratory 2048×1152, five-second, original-v0.1 BF16 hybrid/Veda control
reduced peak allocated memory from 34,914,620,416 to 24,200,463,872 bytes (30.69%). Denoising was
221.633 versus 223.787 seconds; this is a memory improvement, not a measured speedup. All 120 decoded
RGB frames matched exactly. Audio decoding was not bit-identical, consistent with previously observed
decoder variability; audio-content equivalence was not established. These are one paired case,
not a general quality guarantee or an expansion of the published pipeline limits. CPU float32/BF16
checks also cover block output and temporary ownership at the FFN boundary.

## Check output quality {#quality}

A faster result is useful only if it still meets your task. Judge decoded outputs against the original instructions and references: instruction following, identity and detail consistency, motion, visual artifacts, and the relationship between audio and video. A candidate matching a baseline may still fail the task if the baseline also fails.

Inspect multiple decoded frames over time, watch at normal speed, and listen to the audio. Still frames cannot establish natural motion; a waveform cannot establish correct meaning. Keep supported findings separate from unknown dimensions, sampling gaps and reviewer disagreements. Tensor error diagnoses implementation differences; it is not a generation-quality score.

Exact attention does not make a distilled adapter equivalent to the base model, or guarantee identical floating-point results across GPU architectures. The current [support notes](../guide/profiles#lora) describe the narrower checks completed for each profile.

The complete pipeline is available for the [qualified profiles](./pipeline-profiles). Its integration checks establish that measured requests complete correctly; they do not establish a general prompt-to-MP4 speed or quality guarantee.

## Extended hybrid delivery (0.6.12) {#extended-hybrid}

Five complete fresh-conditioning requests ran sequentially on one RTX 4090 48 GB / 450 W, original v0.1 BF16 hybrid, four evaluations, explicit Veda. The first request includes cold initialization; later requests reuse the pipeline. These are observed full-request latencies, not an isolated speed comparison. All delivered exact frames/dimensions and fully decoded video plus 32 kHz stereo audio. The host-memory cap was 260 GiB, not a measured minimum. Memory values are the maximum reported denoising/media-stage allocated peaks, not total process memory or a guaranteed minimum device size.

| Mode | Images | Canvas | Seconds | Full request (s) | Recorded stage peak (GiB) |
| --- | --- | --- | --- | --- | --- |
| I2VA | 1 | 1024×1024 | 15 | 484.13 | 27.74 |
| I2VA HD | 1 | 1440×1440 | 10 | 609.89 | 36.53 |
| FL2VA | 2 | 1344×768 | 15 | 405.50 | 27.81 |
| Ref2VA | 1 | 960×544 | 15 | 172.07 | 14.83 |
| Ref2VA | 9 | 960×544 | 5 | 93.15 | 9.44 |

Capacity evidence is separate from quality. Multiple-time visual inspection found camera/composition drift and fine-detail changes in some scenes, and complex reference/action adherence is not fully qualified. Audio was finite and decodable; silence or sound semantics are not certified by an AAC track. Use explicit silent delivery for digital silence. There is no H100, ordinary 24 GiB 4090, or arbitrary 4MP/15s qualification. The request guard enforces the joint resource boundary. Integration source: `90e19b3b`.


## Optional bounded RGB streaming {#media-streaming}

Source builds can select `H3Pipeline(..., media_video_input="pipe")` or
`vflash generate --media-video-input pipe`. The lower-level `encode_mp4` option is
`video_input="pipe"`. The default remains `file`. Both paths quantize the same eight-frame
blocks and retain H.264/AAC, audio delivery, exact temporal anchors and atomic no-clobber publication.
Pipe mode removes the raw RGB temporary file: 1,509,949,440 bytes for 120 frames at 2048².
It does not eliminate the decoded CPU tensor or qualify larger GPU requests.

A controlled, initialized four-step hybrid/Veda I2VA comparison at 2048² / five seconds measured:

| GPU / host scope | File A1 | Pipe B | File A2 | Whole-request conclusion |
| --- | ---: | ---: | ---: | --- |
| H100 SXM, one remote host | 282.452 s | 294.185 s | 283.594 s | Pipe 3.94% slower than control mean |
| RTX PRO 6000 Server, another remote host | 498.368 s | 428.771 s | 359.039 s | Within control drift; no established speedup |

Each three-video group was byte-identical. This establishes encoding equivalence for these cases,
not general visual quality or prompt adherence. The PRO controls had substantial memory-pressure
drift; do not interpret the first control alone as a 14% pipeline improvement. Use the option when
its measured CPU/I/O or temporary-storage benefit outweighs its cost for the actual workload.
H100 SXM retains the file path. No provider acquisition, model loading, transfer or queue time is
included in these initialized request timings. The pipe timeout owns feeding and FFmpeg execution;
failure kills/reaps the child and removes temporary outputs.


### Explicit CPU codec threads (source builds)

`vflash generate --media-video-threads 16`, `H3Pipeline(media_video_threads=16)` and
`encode_mp4(..., video_threads=16)` select H.264 codec threads. The default None retains
FFmpeg auto selection. Explicit values are integers 1–256; filter/audio threads are separate.
Both file and pipe inputs support this control. Lossy bitstreams can change with thread count.

A CPU-only comparison of the same 2048²/120-frame RGB input, libx264 medium/CRF18 and
8-CPU/8-GiB container measured auto at 7.127/7.432 s, 16 threads at 6.648 s and 8 at 9.847 s.
The 16-thread arm was 8.68% faster than the control mean with 50.54% less peak anonymous memory.
Default and 16-thread full-frame luma SSIM were both 0.994817 against the same decoded input;
decoded audio matched. Bitstreams differ. Sparse visual inspection and compression metrics do
not qualify model quality. This is an explicit CPU option, not a changed GPU default or proof
of a remote timeout fix, full-inference speedup, or an optimum tied directly to CPU quota.

[FFmpeg codec options](https://ffmpeg.org/ffmpeg-codecs.html#Codec-Options).


## Explicit buffered and lossless-compressed weight reads

`vflash[weight-io]` adds the optional Zstandard codec. With the existing Torch
runtime installed, `SequentialWeights` is an explicit CPU loader for immutable
safetensors. It reads into one owned shard buffer, with 1/2/4/8 bounded workers;
retained tensor views keep that whole buffer alive. Mapped loading remains the
default everywhere else. No GPU profile, model precision or public pipeline
admission changes automatically.

```python
from pathlib import Path
from vflash.native.h3_weight_codec import prepare
from vflash.native.h3_buffered_weights import SequentialWeights

source = Path("weights.safetensors")
sidecar = Path("prepared-sidecar")  # a new directory; originals are preserved
prepare(source, sidecar)             # CPU preparation, once per immutable source
with SequentialWeights(source, read_workers=4, compressed_directory=sidecar) as weights:
    tensor = weights.get_tensor("weight")
# tensor retains owned CPU storage after the scope closes
```

Use trusted artifacts and bind original files and sidecars to the same prepared
asset revision before loading. Layout validation and immutable-file checks are
not cryptographic authentication. Preparation compares each lossless frame to its
input without hashing model payloads. A ready sidecar is published manifest-last;
partial preparation cannot be consumed. Originals must remain available for the
header and rollback. Sidecar preparation costs CPU time and duplicate storage.

On one H20 / 150GiB host with slow retained storage, a 20.73→17.08GB native
sidecar reduced initialization 14.25% and initialization plus first output 5.54%
in an A/B/A comparison. Further 29.19→23.88GB text sidecars reduced the first
output path 286.93→270.07 seconds (5.88%) and actual registration 301.02→287.56
seconds across two paired workloads, with the other optimizations fixed. Warm
second output was 90.49/90.62 seconds. Both paired full decoded videos and PCM
streams were identical; these are copy-equivalence checks, not general semantic
quality acceptance. No physical storage-cache miss or cross-host average is
claimed. Anonymous-memory peak rose 7.89→37.38GB and total cgroup peak
65.10→90.86GB; this is a meaningful capacity cost. A fast local page cache was
slower with compression. Keep selection explicit and measure complete startup,
delivery, host memory and retained-storage costs on the intended machine.

The reusable core comes from validated application source `0ae285da`, with
header validation and failed-read buffer cleanup integrated here. This release
does not expose the application's FP8 runner or automatically upgrade a worker.

## Explicit approximate FP8 linear primitives

Version 0.6.29 exposes `vflash.native.h3_fp8_linear.RowFP8Execution` and
`RowFP8Linear`. They consume trusted E4M3 codes shaped `[out, in]` with FP32
row scales `[out, 1]`. This is an explicit numerical interface, **not** an FP8
asset loader, pipeline precision flag or change to an existing BF16 profile.
Install the GPU extra for Torch; Triton is imported only for its selected kernel.

```python
import torch
from vflash.native.h3_fp8_linear import RowFP8Execution

codes = torch.zeros((16, 32), dtype=torch.float8_e4m3fn)
scales = torch.ones((16, 1), dtype=torch.float32)
values = torch.ones((2, 32), dtype=torch.bfloat16)
reference = RowFP8Execution("weight-only").linear(values, codes, scales)
# For explicitly admitted SM90/SM120 CUDA tensors:
execution = RowFP8Execution("w8a8", activation_quantizer="triton", fast_accum=False)
# result = execution.linear(values.cuda(), codes.cuda(), scales.cuda())
```

Weight-only reconstructs approximate weights and uses ordinary GEMM. W8A8 also
quantizes activations per row and uses Torch's scaled GEMM with BF16 output.
Torch and fused Triton activation quantizers are explicit choices; unsupported
hardware fails without a precision fallback. Counters and optional asynchronous
measurement scopes belong to each execution object. No global class patch, model
read, device selection or GPU fence is introduced by these primitives.
`RowFP8Linear` initially owns meta weights/scales and preserves the original bias;
a trusted loader must assign all tensors before use. Caller-owned immutable
assets, finite prepared scales, model conversion and complete-pipeline admission
remain the caller's responsibility.

The fused kernel retains 64-bit row-address arithmetic before multiplication,
including a value-half FFN view whose physical stride crosses 2³¹ elements.
This is the already exercised implementation from application source `3bbaa52d`:
private H100, RTX PRO 6000 and H20 integrations exercised the arithmetic, and the
H20 batch includes nine complete outputs across four modes, 0.25MP/five seconds
through 4MP/fifteen seconds. This library extraction has CPU arithmetic/ownership
checks and offline SM90/SM120 compilation; it has **not** rerun those whole
pipelines on the new package. No universal speedup, baseline-equivalent temporal
quality, smaller-card capacity or public quantized-pipeline support is claimed.
FP8 is approximate; matching the extracted computation does not remove its
previously measured quality uncertainty. Production defaults are unchanged.

### Per-pipeline text loading

`H3Pipeline(..., text_encoder_loader=loader)` accepts an explicit trusted Python
callable. It receives the prepared model root as a `Path` and returns a CPU
`Qwen3VLForConditionalGeneration` matching the pinned architecture. The owning
conditioner still trims the consumed prefix, installs offload and releases the
model. Loading stays lazy until `prepare()` or the first request. Omitting this
option retains the pinned offline BF16 loader. Strings/module paths are not
resolved as code; it is not an HTTP or CLI plugin option.

A loader can bind separately prepared FP8 assets and an independent
`RowFP8Execution`, without replacing Transformers' global `from_pretrained`.
The caller still owns immutable asset identity and custom-precision qualification;
this hook alone does not turn a BF16 prepared receipt into an FP8 artifact.
Native FP8 construction and full quantized pipeline admission remain separate.
CPU owner/failure tests cover independent loaders. A tiny 51-layer pinned Qwen
fixture used 372 actual FP8 Linear modules and the real H3 text/image/multi-image/
video feature helper; all four outputs matched the restored-weight control.
This is interface and arithmetic evidence, not full-model, GPU or video quality
qualification. Keep nonpersistent rotary buffers in their original FP32 precision
when constructing such controls; converting the whole module to BF16 changes them.


For a trusted Python integration, `native_denoiser_loader` is forwarded from
`H3Pipeline` through `NativeEngineSession` to the owning native runtime. It receives
the validated artifact and the resident loader's keyword arguments (device, AdaLN
loader, attention and fusion settings), and must return a compatible denoiser
with the same lifecycle. It requires explicit `weight_residency="resident"` and
single-device execution; existing capacity/model gates still apply. It is not a
serialized plugin name or a CLI/HTTP option. `H3NativeDenoiserBF16Resident.load`
also accepts an owned `block_loader`; block subclasses may override
`_resident_weight` without replacing module globals. The defaults retain strict
BF16 placement. These ownership interfaces do not authenticate alternate assets
or qualify their numerical precision, performance or output quality.
