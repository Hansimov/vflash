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

## Check output quality {#quality}

A faster result is useful only if it still meets your task. Judge decoded outputs against the original instructions and references: instruction following, identity and detail consistency, motion, visual artifacts, and the relationship between audio and video. A candidate matching a baseline may still fail the task if the baseline also fails.

Inspect multiple decoded frames over time, watch at normal speed, and listen to the audio. Still frames cannot establish natural motion; a waveform cannot establish correct meaning. Keep supported findings separate from unknown dimensions, sampling gaps and reviewer disagreements. Tensor error diagnoses implementation differences; it is not a generation-quality score.

Exact attention does not make a distilled adapter equivalent to the base model, or guarantee identical floating-point results across GPU architectures. The current [support notes](../guide/profiles#lora) describe the narrower checks completed for each profile.

The complete pipeline is available for the [qualified profiles](./pipeline-profiles). Its integration checks establish that measured requests complete correctly; they do not establish a general prompt-to-MP4 speed or quality guarantee.
