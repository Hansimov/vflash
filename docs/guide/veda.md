# Veda attention

`attention_backend="veda-sm89"` selects learned sparse video attention for the
original LightX v0.1 four-step keyframe profiles on one SM89 GPU. It also accepts
the explicit FL/Ref `HybridModel`. `auto` remains dense for these profiles.
Other model profiles and cooperating GPUs reject this option.

After installing Vflash's pipeline dependencies, run `python -m vflash.install_veda`.
The installer fetches Veda revision `fd59c7277ccc37ebf1a8f6823474b8c2ef2e33a0`,
packages its independent core and retains its MIT license and SageAttention 1.0.6
BSD notice. It does not install ComfyUI or download model weights. Supply the
[Veda H3 predictor](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)
separately under the MiniMax H3 Community License; the evaluated revision is
`76f202874608115408d73280be9531c4ab888242`.

```python
pipeline = H3Pipeline(
    prepared, device=device, trust_local_code=True,
    attention_backend="veda-sm89",
    veda_predictor=Path("models/veda-predictor.safetensors"),
    hybrid_model=HybridModel(Path("models/MiniMax-H3/transformer_ref")),
)
```

`generate` and `denoise` accept `--attention-backend veda-sm89 --veda-predictor FILE`.
For the native HTTP service set `VFLASH_ATTENTION_BACKEND=veda-sm89` and
`VFLASH_VEDA_PREDICTOR` to the mounted predictor. Missing or incompatible assets,
a dependency revision mismatch, or a failed GPU self-test stops execution; there
is no dense fallback. The predictor is session-owned; geometry and statistics
reset before each serial request and resources are released on close.

Layers 0–4 and 45–49 retain Torch Flash attention. The other forty layers use
Veda's actual `triton-int8` kernel with generated keep budget 0.1. Text, audio and
condition connections remain dense within that approximate kernel; this is not
an exact-arithmetic guarantee. Receipts identify the actual kernel, 160 sparse
and 40 dense calls per four-step request, target grid, predictor-plan match and
observed sparse-layer work. The predictor's training schedule and geometry do not
make this four-step combination numerically equivalent to dense H3.

An exploratory matched-input/actual-noise control on one RTX 4090 48 GB measured
Hybrid denoising at 102.928 s dense and 72.867 s Veda; denoising plus media was
124.408 s versus 94.063 s. Workload: 1536×640, five seconds, four evaluations,
BF16 trunk/runtime LoRA, two-slot block ring, Torch 2.11/CUDA 13.0, frozen clean
conditioning and a common official decoder. Another GPU was active: these are
bounded control timings, not isolated complete-request or same-quality claims.
Eight sampled times and native face/fabric crops showed motion changes without
new large structural failures; fine-detail softness remained. Audio semantics
were not assessed. Independent complete-pipeline qualification is separate.

Six complete Python-pipeline requests additionally ran on the same single GPU,
with the real official encoder and decoder: two source families, I2VA and L2VA
paired controls, a larger I2VA canvas, and a one-image hybrid Ref request.
At 896×512/5s, I2VA denoising was 37.507 s dense and 34.196 s Veda; L2VA was
36.582 s and 34.372 s. The matched warm L2VA request was 64.733 s versus 61.608 s
(4.83% faster). Cold I2VA totals were 165.293 s and 171.789 s, including different
94.647/101.967 s initialization and first-use costs: no cold-start win is claimed.
Veda I2VA at 1344×768 took 118.634 s warm, including 74.010 s denoising;
hybrid Ref at 896×512 took 59.002 s warm. This shows why one speed ratio must not
be applied to every canvas. Native-size crops showed more facial detail on the
larger canvas, alongside the extra cost; existing object-contact errors persisted.
Changed expressions/reflections are visible in the attention comparison. Audio
semantic equivalence and universal quality gains remain unestablished.


## Explicit SM120 dense-block acceleration

**Quality review reopened (2026-10-10).** Temporal instability has been reported in high-resolution Sage output. Consecutive-frame review of a matched 2048-square pair shows detail fluctuations in both Flash/Veda and Sage/Veda, with larger variation in one Sage region; the cause is not isolated. Earlier five-frame sampling and successful playback do not qualify temporal quality. Treat this option as experimental, not a quality-qualified production recommendation. Flash dense blocks remain the default; removing Sage is not a proven flicker fix.

On a tested single RTX PRO 6000 or RTX 5090, keep `attention_backend="veda-triton"`
and select `veda_dense_backend="sageattention2"` in `H3Pipeline`. The complete
`generate` CLI exposes `--veda-dense-backend sageattention2`. Default dense blocks
remain Torch Flash; missing extensions and failed kernels stop execution.

Install [SageAttention 2.2.0](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5)
separately, compiled for SM120 and the actual Torch/CUDA environment. The tested
build used Torch 2.11/CUDA 13.0 and the pinned upstream revision above. SageAttention
has its own Apache-2.0 license; no optional extension or model is bundled here.
Do not use an SM90-only wheel on SM120, or treat successful CPU import as GPU qualification.

Only layers 0–4 and 45–49 switch to INT8 QK / FP8 PV, with per-warp QK quantization,
FP32+FP16 accumulation, key smoothing and no value smoothing. The other forty Veda
layers, keep budgets, four-step schedule, conditioning and decoder stay unchanged.
Receipts identify both effective backends, dense precision and actual call counts.
This option is intentionally approximate; it is not an equivalence guarantee.

Same-PRO-6000-Server I2VA comparisons with BF16 hybrid weights and block-ring loading
measured 1536×864/15 s at 257.718 s Flash versus 223.718 s Sage (13.19% shorter), and
2048²/5 s at 314.753 versus 302.123 s (4.01% shorter). These are complete warm
conditioning/denoising/media requests, excluding initialization and transfer. At
1344×768/5 s, Flash/Sage/Flash was 49.683/49.775/52.914 s: no stable short-request
win. The batch's 1205.634 s initialization shows why warm gains do not guarantee
lower first-output latency or rental cost.

All eight outputs decoded with audio/video and played in the application. Five
sampled times per clip plus native detail crops retained subject/architectural
structure without a new obvious structural failure; motion trajectories changed.
Existing camera adherence defects persist and audio semantics are unassessed.
These bounded observations do not establish universal quality equivalence. H100
and other architectures retain Flash dense blocks until separately qualified.

Same-RTX-5090 1344×768/5 s Flash/Sage/Flash measured 75.632/70.395/75.428 s;
1536×864/15 s measured 379.763/310.595 s (18.21% shorter). Both fifteen-second
arms peaked at 24.143 GiB of denoising allocation with the corrected profiler.
The longer result has full audio/video decode and application playback; sampled
faces, fabric and motion retain the same limitations described above.

The same 5090 batch also completed 2048²/5 s with Sage in 368.807 s, at 26.982 GiB peak denoising allocation. It has complete AV decode and application playback, but no matched same-instance Flash control; this establishes capacity and delivery, not a four-MP speedup.


## Explicit SM90 CuTe Flash dense blocks

`H3Pipeline(..., attention_backend="veda-triton", veda_dense_backend="cute-flash")`
and `generate --veda-dense-backend cute-flash` select the pinned
[FlashAttention CuTe implementation](https://github.com/Dao-AILab/flash-attention).
Install `flash-attn-4==4.0.0b34` separately; the measured environment used
`nvidia-cutlass-dsl==4.8.0`, Torch 2.11.0/CUDA 13.0 and Triton 3.6.0.
No optional extension, private model, or runtime credential is bundled.

The option currently admits SM90 only. It retains BF16 QKV, complete dense context,
noncausal attention, one split and the original forty Veda layers. Only the ten
dense layers change implementation. Kernel errors, unsupported devices and wrong
package versions fail instead of silently choosing another backend. Torch Flash
remains the default. This does not restore Sage, enable FP8 attention, change the
model precision, or expand a profile's geometry/memory admission limits.

Six actual H100 SXM kernel probes at H3 lengths 9,984–438,784 measured Torch Flash
at 7.888–16,796.029 ms and CuTe at 4.292–8,383.208 ms. Both kernels ran on the device;
these are dense-layer timings, not whole-video speedups. Full I2VA comparisons
hold experimental FP8 Linear/text, hybrid four-step denoising and Veda fixed.
Their limited quality review and complete timings are recorded below; this option
is not a claim of universal same-quality performance or new public FP8 support.


| Canvas / duration | Torch full request seconds | CuTe full request seconds |
| --- | ---: | ---: |
| 512×512 / 5s | 81.744 | 77.015 |
| 768×768 / 10s | 41.032 | 40.115 |
| 1024×1024 / 15s | 130.214 | 114.534 |
| 1536×1344 / 10s | 207.546 | 168.019 |
| 2048×2048 / 5s | 235.523 | 195.820 |
| 2048×2048 / 15s | 1237.038 | 891.409 |

One physical H100 SXM ran all six controls before all six candidates, with one seed
per scene. The table includes conditioning, denoising and media, excluding initialization
and network delivery. Six-case process makespan including initialization was
2068.473 versus 1629.436 seconds. Initialization itself increased from 131.435 to
138.192 seconds. The first case includes cold conditioning and first-layout compilation;
its total difference is not proof of a short-clip dense-kernel improvement. For the
largest case, measured dense attention fell from 692.458 to 348.078 seconds and the
whole request from 1237.038 to 891.409 seconds (27.94% shorter). Nested timings are not additive.

All twelve videos fully decoded and played in the application. Sampled people,
objects and architecture retained their structure; consecutive native-size rain-street
and bookshelf crops showed no newly observed major structural failure. Both the control
and candidate retain an early library camera reframe and insufficient intended action.
This limited inspection is not full quality equivalence. Audio semantics, other modes
and multiple seeds remain unassessed. Sequential order/cache effects remain, and warm
stage gains do not imply a faster complete cold start.
