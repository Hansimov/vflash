# Benchmark results

Each result belongs to its stated software version, hardware and workload. These published summaries preserve their original scope for reproduction; they are not a ranking of the current release. See [profiles and hardware](../guide/profiles) for current support and [performance and quality](./performance) for measurement guidance.

## Two 3080s · a3 {#sm86-parallel}

The comparison below was measured on the **a3** runtime, before a4 introduced segmented pinned storage. Its original timings and memory figures are retained as historical evidence, not a benchmark of every later release. The a5 loader and lifecycle changes do not update this ranking.

A target-hardware measurement on one frozen Ref2VA Turbo4 request (928 × 512, 124 model frames, four evaluations, 18,175 tokens, BF16 weights and exact attention) compared one RTX 3080 20 GB with the same primary device plus a second 3080. Both ran at their default 320 W limits, on PCIe 3.0 x16 host-bridge links without peer access. The runtime used PyTorch 2.11.0+cu130 and Triton 3.6 with eight CPU threads.

| Warm execution | Repeats | Median conditioning-to-latent time | Range |
| --- | ---: | ---: | ---: |
| One GPU | 4 anchors | 85.694 s | 85.539–85.805 s |
| Two GPUs, `sequence-head` | 3 | 49.671 s | 49.599–49.973 s |

This is a **1.725× speedup** on the measured workload. Three interleaved A/B/A2 comparisons had at most 0.227% anchor drift, no sampled thermal throttling or thermal-counter growth, and successful CUDA probes before and after every request. The same process retained both single/parallel device rings and shared their host weights to avoid repeated loading during the comparison. Memory figures below come from a separate standalone session.

That a3 standalone sequence/head session initialized in 41.758 s with an already warm filesystem cache; its first request took 51.186 s and its next request 49.441 s. Denoising allocation peaked at 4.745 / 4.614 GiB across the pair. Active pinned host allocation was 58.009 GiB, with peak process RSS of 59.90 GiB. Initialization, storage cache, GPU allocation, reserved memory and process RAM are distinct costs.

Standard weight `tensor` completed three warm requests in a separate standalone session: median 58.615 s, range 58.526–58.709 s, approximately **1.462×** against the preceding same-primary single-GPU anchors. Initialization took 47.899 s and the first request 60.079 s. Denoising allocation peaked at 4.825 / 4.785 GiB, pinned host allocation at 58.792 GiB, and process RSS at 60.70 GiB. This is a separate TP screen without a new interleaved A/B/A2 comparison. One sample reported a transient software thermal flag without thermal-counter growth or a corresponding clock reduction. All three results are retained; this screen has a narrower evidence level than the isolated comparison above.

These measurements stop at the exported latent file. Prompt/reference encoding, VAE decoding, MP4 encoding, queueing and network transfer are outside the boundary. Two cooperating GPUs improve one-request latency; independent single-GPU workers provide a different throughput tradeoff.

Both parallel strategies completed full trajectories and a paired decoded-video/audio smoke. They are not bitwise equal to single-GPU execution. The pipelined head exchange itself matched the unpipelined sequence/head trajectory bitwise, but the partitioning changes GEMM shapes and standard tensor parallelism adds reduction boundaries. One decoded example is not cross-case quality qualification; no same-quality or general prompt-to-video speed claim is made.

## 4090 host allocation · a4 {#sm89-host-memory}

One Ref2VA check on a single **RTX 4090 48 GB / SM89** used three reference images, 928 × 512, and 124 model frames at 24 fps. It compared the 0.1.0a3 allocator with the allocator shipped in 0.1.0a4, using PyTorch 2.11.0+cu130. It kept the fixed BF16 Ref Turbo4 v0.1 configuration: 4 NFE, video/audio shifts 12/3, LoRA strength 1 and alpha 8. Active pinned weight storage fell from about **58.009 GiB to 40 GiB**. With identical inputs, final video and audio latents matched the previous allocator bit for bit. Warm native denoising took **41.607 s before and 41.641 s after**: the measured benefit is lower host memory use with comparable inference time. The 40 GiB figure covers pinned weight storage; the process still needs additional RAM.

A separate same-GPU, four-step capacity check compared resident weights with block streaming. Final latents matched bit for bit, while block streaming took longer. That check supports the capacity option, not a claim that streaming is faster.
