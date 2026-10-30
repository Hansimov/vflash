# Segmented native audio/video

`ReferenceAVRuntime` owns an explicit original-FL reference pipeline on one complete 96 GiB SM120 GPU.
Its default recipe uses Larry v4 EMA for the first four of ten visual evaluations, the original model
for the remaining six, and twelve Larry evaluations for native sound conditioned on that complete video.
The second pass preserves every generated video latent. It generates sound with H3's audio model.

Visual attention defaults to Veda: 44 sparse layers and six dense SageAttention2 layers, with INT8 QK / FP8 PV.
Full Sage is a separate explicit backend and is never an automatic fallback. The returned visual attention
receipt verifies all 440 sparse and 60 dense block calls. Audio uses live text/audio rows with cached visual KV.

The initial scope is five-second I2VA, one image already normalized without stretching to the exact 32-aligned
model canvas, at most 2 × 1024² pixels and a 2048-pixel longest edge. Model execution uses 124 frames;
delivery uses 120 frames at 24 fps. A 1920×1088 model canvas can be cropped to 1920×1080 by the application.
Other modes, durations, smaller GPUs and face restoration retain their existing separately configured pipelines.

All temporal video positions and the full first-frame condition remain in the audio pass. Spatial conditioning
uses stride two up to 1376×768 pixels and stride four above that. This is approximate conditioning; the selected
video itself stays intact. At the smaller sizes, visual decoding overlaps audio sampling. Larger canvases decode
sequentially to retain memory headroom. QKV fusion, strict BF16 linear execution, prompt-encoding reuse and a
single final encode reduce overhead. Each request restores hooks, adapters and schedulers on success or failure.

The caller supplies versioned local assets and a text-weight loader. Loading requires `trust_local_code=True`:

- Original FL base matrices in the declared row-FP8 layout, with no merged acceleration adapter.
- Original-FL time embeddings and modulation tables for ten/twelve evaluations, video/audio shift 12/3.
  The table manifest records filenames, sizes and actual time grids. Every model call checks its time grid.
- [Larry v4 EMA weights](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/tree/43a74557ac3f6539db8e0f2a959d03feb7a81480),
  `minimax_h3_turbo_v4_step600_ema.safetensors`. Twelve audio evaluations are a downstream operating point,
  rather than the author's four-to-eight-step recommendation.
- [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/13bb3c625e8b1e457d7124d92207b49e00e4a6fe),
  its `videox_fun` package, official media decoders and the Veda predictor.

The loader applies an explicit correction that preserves the original FP32 time embedding through SiLU before
the BF16 modulation projection. Upstream source remains unchanged. Asset distribution, licenses, model receipts,
request admission and result publication remain with the caller. No application state or private assets ship in Vflash.

`runtime.generate(VideoRequest(...), output_directory)` returns the MP4 path, native latent files, per-pass
steps and CUDA timing, attention coverage, conditioning reuse, media stages, decoder-window peak allocation and complete-call time.
Loading, acquisition, upload and browser playback are separate costs. The preceding `H3Pipeline` API keeps its contract.

Motion blur, small faces and temporal identity still require output review against the brief; this recipe is a
balanced motion/texture/native-audio choice, not a universal quality guarantee.

## Python integration

Install the `pipeline` extra from the [0.22.0 release](https://github.com/Hansimov/vflash/releases/tag/v0.22.0),
plus the pinned Veda/Sage dependencies used by this runtime. Supply the verified paths below; the example
is an integration function, not an automatic asset downloader or a CLI profile. `first_frame` must already
be exactly 1024×1024 for this example. Each output directory must be new.

```python
from pathlib import Path
from vflash.pipeline import VideoRequest
from vflash.pipeline.reference_av_runtime import ReferenceAVRuntime


def generate_clip(*, cache_root, source_root, tables, adapter, decoder_directory,
                  predictor, text_loader, first_frame, prompt, output_directory):
    runtime = ReferenceAVRuntime(
        cache_root=Path(cache_root), source_root=Path(source_root),
        tables=Path(tables), adapter=Path(adapter),
        decoder_directory=Path(decoder_directory), predictor=Path(predictor),
        text_loader=text_loader, trust_local_code=True,
    )
    try:
        request = VideoRequest(
            prompt=prompt, first_frame=Path(first_frame), width=1024, height=1024,
            duration_seconds=5, seed=42, audio_delivery_profile="web-v1",
        )
        return runtime.generate(request, Path(output_directory))
    finally:
        runtime.close()
```

`text_loader` is a zero-argument callable returning the loaded Qwen3-VL text encoder. It owns the
prepared quantization layout and source identity. `vflash.native.h3_fp8_text.load_row_fp8_text_encoder`
with explicit `RowFP8TextShard` entries is the reusable loader; supply matching `RowFP8Execution` and
original unchanged shards. Never substitute LightX-merged trunk weights for the original-FL cache.

The runtime has no context-manager protocol: call `close()` in `finally` and serialize requests.
`result["path"]` is the MP4; `visual.safetensors` and `native.safetensors` live in the output directory.
`seconds` excludes construction and application delivery. `peak_cuda_bytes` is sampled after the
media decoder resets CUDA peak statistics; it is **not a whole-request peak**. Use independent
whole-window memory observation when sizing a deployment. The default call performs 22 evaluations.
