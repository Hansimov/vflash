# Veda attention (explicit SM89 option)

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
