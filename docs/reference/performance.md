# Measuring performance

Measure the path that matters to your application, and keep its scope visible. Vflash's current public output is a latent tensor file, so its inference timing is not a prompt-to-video latency measurement.

## Separate loading from repeated requests {#timing}

The CLI creates a new session for every invocation. The HTTP service reuses one loaded model, so the first job and later jobs have different costs.

Successful results expose these values in `session`:

| Field | Meaning |
| --- | --- |
| `request_index` | The request number in this model session, starting at 1 |
| `initialization_seconds` | How long this session took to initialize |
| `initialization_charged_seconds` | Initialization time on the first request; zero on later requests |
| `request_wall_seconds` | Time spent executing the request in the initialized session, including writing its latent output |

For an engine-side first-request cost, add `initialization_charged_seconds` and `request_wall_seconds`. Queue wait, process startup, HTTP transfer, input encoding, and video decoding are outside that sum. Measure those separately when they are part of your application.

`/readyz` passing before the first job means the service can see the configured files and GPU. It does not mean the model is already loaded.

## Make comparisons useful {#comparisons}

Use the same conditioning bundle, model and adapter revisions, profile, GPU, and output boundary. Report first-use and repeated-request timings separately, with enough runs to show variability.

When comparing memory use, distinguish GPU allocated memory, GPU reserved memory, device-wide usage, and host RAM. In particular, the 3080's streamed weights consume system memory that does not appear in a VRAM-only chart.

Changing from Turbo8 to Turbo4 changes the model's distilled schedule as well as the amount of computation. Label that difference alongside a speed comparison.

## Two-device execution {#parallel}

A target-hardware measurement on one frozen Ref2VA Turbo4 request (928 × 512, 124 model frames, four evaluations, 18,175 tokens, BF16 weights and exact attention) compared one RTX 3080 20 GB with the same primary device plus a second 3080. Both ran at their default 320 W limits, on PCIe 3.0 x16 host-bridge links without peer access. The runtime used PyTorch 2.11.0+cu130 and Triton 3.6 with eight CPU threads.

| Warm execution | Repeats | Median conditioning-to-latent time | Range |
| --- | ---: | ---: | ---: |
| One GPU | 4 anchors | 85.694 s | 85.539–85.805 s |
| Two GPUs, `sequence-head` | 3 | 49.671 s | 49.599–49.973 s |

This is a **1.725× speedup** on the measured workload, below a 1.8× optimization target. Three interleaved A/B/A2 comparisons had at most 0.227% anchor drift, no sampled thermal throttling or thermal-counter growth, and successful CUDA probes before and after every request. The same process retained both single/parallel device rings and shared their host weights to avoid repeated loading during the comparison. Memory figures below come from a separate standalone session.

The standalone sequence/head session initialized in 41.758 s with an already warm filesystem cache; its first request took 51.186 s and its next request 49.441 s. Denoising allocation peaked at 4.745 / 4.614 GiB across the pair. Active pinned host allocation was 58.009 GiB, with peak process RSS of 59.90 GiB. Initialization, storage cache, GPU allocation, reserved memory and process RAM are distinct costs.

Standard weight `tensor` completed three warm requests in a separate standalone session: median 58.615 s, range 58.526–58.709 s, approximately **1.462×** against the preceding same-primary single-GPU anchors. Initialization took 47.899 s and the first request 60.079 s. Denoising allocation peaked at 4.825 / 4.785 GiB, pinned host allocation at 58.792 GiB, and process RSS at 60.70 GiB. This is a separate TP screen without a new interleaved A/B/A2 comparison. One sample reported a transient software thermal flag without thermal-counter growth or a corresponding clock reduction. All three results are retained; this screen has a narrower evidence level than the isolated comparison above.

These measurements stop at the exported latent file. Prompt/reference encoding, VAE decoding, MP4 encoding, queueing and network transfer are outside the boundary. Two cooperating GPUs improve one-request latency; independent single-GPU workers provide a different throughput tradeoff.

Both parallel strategies completed full trajectories and a paired decoded-video/audio smoke. They are not bitwise equal to single-GPU execution. The pipelined head exchange itself matched the unpipelined sequence/head trajectory bitwise, but the partitioning changes GEMM shapes and standard tensor parallelism adds reduction boundaries. One decoded example is not cross-case quality qualification; no same-quality or general prompt-to-video speed claim is made.

## Check output quality {#quality}

A faster result is useful only if it still meets your task. Compare representative outputs for instruction following, reference consistency, motion, audio, and visible artifacts. Tensor similarity is helpful for debugging a fixed computation, but it does not replace reviewing decoded results.

Exact attention does not make a distilled adapter equivalent to the base model, or guarantee identical floating-point results across GPU architectures. The current [support notes](../guide/profiles#lora) describe the narrower checks completed for each profile.

This page does not publish a general prompt-to-MP4 speed claim. The public package does not yet include that complete path.
