# Sol-Engine alignment on SM86 and SM89

Vflash reviews NVIDIA Sol-Engine as an upstream source of optimization mechanisms, not as a
configuration to copy wholesale. The current deployment targets are RTX 3080 20 GB (SM86) and RTX
4090 48 GB (SM89), usually as one GPU or a two-GPU PCIe pair. Sol-H3's newest headline results use
different accelerators, interconnects, step counts, adapters, precision and attention semantics.

This page fixes the comparison to Sol-Engine commit
[`ca26dbd`](https://github.com/NVlabs/Sana/tree/ca26dbd2b7034cc90a64c093d715d99b0bfa5b7f)
from 2026-09-19. A newer upstream revision requires a new review.

## What is enabled

The two-device `sequence-head` path now packs QKV and merges returned attention heads with direct
Triton relayouts. The collective wire format and BF16 values are unchanged. Each kernel reads the
source strides and writes destination-major storage once, replacing materialized
`stack/permute/contiguous` or `cat/permute/contiguous` chains. The idea follows Sol-Engine's direct
Ulysses relayout, while Vflash retains its own four-chunk overlap layout and ownership model.

The implementation is on by default for CUDA tensors on both supported architectures. Set
`VFLASH_H3_DIRECT_RELAYOUT=0` only for a controlled diagnostic comparison. Any other value fails
closed. CPU and unsupported tensor layouts use the reference materialization.

At the representative 55,413-token Base16 shape, isolated A/direct/A measurements produced the
following hot-copy medians. Every candidate output matched its reference element for element.

| Operation | SM86 reference → direct | SM86 speedup | SM89 reference → direct | SM89 speedup |
| --- | ---: | ---: | ---: | ---: |
| QKV destination pack | 8.627 → 3.551 ms | 2.429× | 5.152 → 2.639 ms | 1.952× |
| Attention result split | 0.453 → 0.345 ms | 1.312× | 0.334 → 0.266 ms | 1.254× |
| Returned-head merge | 2.501 → 1.234 ms | 2.027× | 1.762 → 0.912 ms | 1.932× |

The allocation screen on both architectures also reduced the QKV pack's incremental peak from
2,383,245,312 to 1,191,622,656 bytes, saving 1,136.4 MiB (50%). Returned-head merge fell from
794,415,104 to 397,207,552 bytes, saving 378.8 MiB (50%). Attention-result split retained its
94.7 MiB output with no peak change. These buffers occur at different points and must not be added
into a claimed pipeline peak.

Final unprofiled complete-request A/direct/A measurements used the same persistent pipeline and the
last control as the hot denominator:

| Hardware and fixed workload | Hot control → direct | Request reduction | Denoising reduction |
| --- | ---: | ---: | ---: |
| Dual SM86 · 5 s · 512² I2VA · 16 evaluations | 174.776 → 172.149 s | 1.503% | 1.995% |
| Dual SM89 · 10 s · 736 × 992 L2VA · 16 evaluations | 474.669 → 472.429 s | 0.472% | 0.623% |

The compressed video elementary streams matched across all three arms. The official audio VAE
retained its known repeat-decode variability, independently of this denoising-only change. Neither
run recorded a thermal-slowdown sample. The direct path therefore stays enabled for its exact
layout and transient-memory improvement, not as a large complete-video speedup. Full attribution
and measurement boundaries are in the [performance guide](./performance).

## Upstream gap and disposition

| Sol-H3 mechanism | Vflash state | Current SM86/SM89 decision |
| --- | --- | --- |
| Destination-major QKV packing and head merge | Same mechanism, Vflash-specific four-chunk layout | **Adopted.** Exact element movement; measured on both architectures. |
| AdaLN precomputation | Schedule overlays already store fixed-step AdaLN tables | **Already present.** No duplicate port. |
| Fused modulation, gates, rotary and SwiGLU | Strict Triton kernels preserve the existing BF16 rounding boundaries | **Already present.** Keep exact defaults. |
| Merged QKV projection | The compiler stores Q/K/V as one `attn.qkv` weight and the runtime issues one wide GEMM before splitting views | **Already present.** Do not add a second projection wrapper. |
| Regional `torch.compile` | The upstream 4090 path compiles selected regions, but reports it inside a bundled “lossless opt” arm rather than as an isolated delta | **Deferred.** Vflash already owns the fixed block loop and hot kernels explicitly. A compiler layer needs a compile-only, persistent-worker A/B/A on each target architecture before it can replace that ownership. |
| Layerwise component offload and temporary VAE residency | Vflash uses explicit CPU masters, a two-slot block ring and serial encoder/core/VAE ownership | **Already present with a different lifecycle.** Do not add a second offload manager; improve the owned lifecycle only when stage and peak-memory measurements justify it. |
| LoRA consumer fusion | Base16 has no adapter; qualified Turbo paths already fuse selected QKV merge and FFN adapter/activation consumers | **Partly present and profile-scoped.** The upstream FastH3 adapter is not interchangeable, and unsupported adapters keep the explicit residual path. |
| Combined RMSNorm/AdaLN and QKNorm/RoPE/pack | Upstream fusion changes reduction or rotary arithmetic boundaries | **Not copied.** Requires independent same-architecture numerical and full-request proof. |
| SOL/BSA sparse attention | Optional approximate attention upstream | **Not a production backend.** A conservative SM89 candidate changed the generated trajectory and missed the preregistered complete-request gate. |
| TeaCache / FirstBlockCache | Upstream 4090 result uses 49 DiT forwards and reuses 35 | **Deferred.** Vflash Base16 uses 16 evaluations; the reuse opportunity and quality risk are different. |
| INT8 QKV and FP8 output transport | Default in the eight-B300 fast profile | **Rejected for the exact default.** Lossy transport and a different topology need a separate profile and quality gate. |
| Fused MXFP8 linears | SM100-family path; upstream reports material output divergence | **Out of scope.** It does not target SM86/SM89 and cannot inherit an exact label. |
| Parallel video/audio VAE and tile sharding | Built around an eight-rank Sol-H3 decoder lifecycle | **Deferred.** Vflash's two-card pipeline currently gives media ownership to the primary after denoising. A port must first improve that lifecycle and show complete-media equivalence. |
| Four-step FastH3 adapter | Changes 49 forwards to four in the headline comparison | **Not a runtime optimization claim.** Adapter quality and model semantics require an independent profile. |
| H3 draft → LTX refinement with TAEH decoders | A separate coarse-to-fine product that changes model, adapter, decoder, resolution path and sampling schedule | **Out of this exact profile.** It can only enter as a separately named quality-qualified pipeline, never as an SM86/SM89 runtime speedup. |

## Exact and approximate are separate products

The released Vflash profiles use dense PyTorch Flash SDPA and exact BF16 transport. “Exact” means
the selected implementation preserves its declared arithmetic and attention contract; it does not
mean that two GPU architectures, two parallel decompositions, or a distilled adapter must emit the
same tensor.

Approximate methods can be valuable, but they must be named, default-off, measurable and removable.
They need a frozen prompt/reference/seed suite, full decoded video and audio review, actual playback
and listening, repeat timing, thermal/stability evidence and a clear rollback. Tensor similarity or
an upstream sample alone cannot qualify them.

Sol-H3's 1/4/8-B300 table combines a four-forward adapter, sparse attention, quantized
communication and parallel decoding. Its optional MXFP8 row is explicitly lossy. Neither result is
a runtime-only speedup for Vflash's dense Base16 16-evaluation SM86/SM89 path.

## Reproducing a comparison

Use one fixed artifact, request and device group. Warm both arms, run A/B/A in one persistent
process, and report the successful request boundary separately from initialization. For a two-GPU
path, report single-request latency and two-worker throughput independently. Use
`--profile-denoise` only for attribution; final throughput should use an unprofiled run.

Do not infer adoption from a faster kernel alone. A candidate enters a supported profile only after
its complete request, decoded media, cleanup and repeated target-hardware measurements pass.
