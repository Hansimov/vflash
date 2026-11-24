# Soundtrack refinement and face repaint

Choose an audio pass when the picture already meets the brief. Face repaint has its own masks, weights and execution limits. Record the selected options with the delivered film.

## Optional audio refinement {#optional-audio-only-six-step-refinement-source-main}

`AudioRefinementRequest` adds an explicit original-model audio pass after a four-step hybrid I2VA generation.
It freezes the completed video, keeps every temporal position and uniformly samples spatial context up to
500,000 pixels with the original rotary coordinates. Video pixels, duration and delivery clock are unchanged.

```python
from vflash.native.h3_audio_refinement import AudioRefinementRequest
from vflash.native.h3_refinement_schedule import refinement_schedule

refinement = AudioRefinementRequest(refinement_schedule(steps=6, denoise=0.5), tuple(audio_tables))
result = pipeline.generate(request, output_path, audio_refinement=refinement)
```

It requires hybrid weights, one device, Veda 44/6, one same-canvas first image and generated audio, with
either resident weights and `veda-triton` or the native two-slot BF16 `block-ring` and `veda-sm89`. Silent
delivery, video-only output, other modes and simultaneous repaint are rejected. The default
`context_pixels=500_000` conditions the pass on a uniformly subsampled 0.5MP video context;
`context_pixels=250_000` selects the 0.25MP budget (since 0.15.1) and `context_pixels=None`
selects full-context six-step control. Progress reports 4 + 6 evaluations and `audio_refinement` records the
added work; failure restores bindings, attention, video and original audio. There is no CLI or HTTP switch
and no default activation.

### Encoded-source soundtrack restoration

Advanced native sessions can take `encoded_audio_source=EncodedAudioSource(video_rows, audio_rows)`
with an explicit `AudioRefinementRequest`. Both tensors are finite, normalized model-coordinate rows:
video excludes the conditioning keyframe prefix, while stereo audio uses channel-major packing.
The session owns a copy and skips the ordinary video trajectory. Keep the original encoded picture
stream for final muxing; VAE decoding is needed only for the new audio.

The default six-step recipe also accepts encoded source media. Source ownership, waveform encoding,
versioning and final original-picture muxing belong to the calling application.

### Longer audio passes (0.19)

`AudioRefinementRequest` also accepts `evaluations=12, denoise=0.9` and
`evaluations=20, denoise=0.95`. Each uses tables compiled with the same steps and denoise value:
`compile_refinement_tables(..., steps=20, denoise=0.95, mode="audio")`, for example.
The pass runs after ordinary four-step I2VA generation or an explicit encoded source. The configured
spatial context budget, hidden/K/V cache and audio/video decode overlap remain independent choices.
Higher re-noising can change the sound itself: compare naturalness, content fidelity and transients
against the sound brief and measure the complete generated MP4 call.

`dense_attention=True` is an additional twenty-step diagnostic: every audio-pass context block uses
its dense attention backend, with the original policy restored on success or failure. It preserves
the four-step video policy. Cached context still freezes non-audio states after the first evaluation.
These explicit candidates keep the ordinary request default unchanged.

### Compact prepared tables (0.19)

`compact_audio_tables(source, output)` in `vflash.compiler.refinement_compact` writes a new
`audio-table-dedup-v1` directory. It deduplicates identical complete rows, retaining **every BF16
bit of the original table**, including unused rows. `load_tables` reconstructs the full contiguous
layout and supports legacy tables. Source files stay intact; inference memory and computation stay
unchanged. A twenty-step set decreases from 611.00 MB to 400.92 MB. Full-table bit comparisons
cover all 51 actual files; six/twelve/twenty-step fixtures also cover signed zeros and exponent bits.

### Hidden-context reuse {#audio-hidden-context}

`reuse_hidden_context=True` reuses first-evaluation BF16 non-audio hidden states; all selected audio evaluations
remain, later steps rebuild K/V and process only audio rows. This is an explicit approximation, never changes
video latents and defaults to false. The cache is scoped to one refinement (at most 40 GiB, with at least
12 GiB of device memory kept free) and released on success or failure.

### K/V reuse and audio/video overlap {#audio-kv-overlap}

On resident SM120, `reuse_kv_context=True` reuses first-evaluation post-RoPE BF16 non-audio K/V while
recomputing audio rows for the remaining evaluations. Select either this or `reuse_hidden_context`.
This explicit approximation retains the selected solver and video latents, with the same 40 GiB cache
limit and 12 GiB headroom. The measured SM89 path uses hidden-context reuse.

`H3Pipeline(..., audio_decode_overlap=True)` overlaps final video VAE decoding with the selected audio pass on
a separate CUDA stream for five-second requests up to 1 MP. It submits only with sufficient memory
headroom and otherwise runs serially. The pipeline owns and joins the thread/future on success and
failure; `stages.audio_video_overlap` reports actual execution. Audio refinement depends on the
completed video, so main video denoising runs first; the overlap is with final picture decoding.
Keep joint audio during main denoising to retain the original motion and mouth conditioning.
Complete-call timings, precision and decoded-media scope are in the [0.18 release record](../reference/releases#_0-18-0).

### BF16 ring weight reuse (0.20) {#audio-ring-weight-cache}

`AudioRefinementRequest(..., weight_cache_bytes=20 * 1024**3)` retains a prefix of the
BF16 ring's weights on device for the audio pass. The default is zero. It preserves
weight values, original compute slots and attention/solver choices; later steps
read cached weights through device-to-device copies instead of PCIe. The scope
restores original bindings on success and failure and reserves 20 GiB for context
and overlapping VAE work. Resident trunks already retain their weights and skip it.
The diagnostic `weight_cache` records the actual retained blocks, bytes and setup time.

Initial RTX 4090 48 GiB qualification covers twelve-step, five-second I2VA with
hidden-context reuse and decode overlap, at 960×544 and 1024². Four complete matched
pairs saved 8.93–11.06 seconds (5.6%–13.6%); all delivered decoded video frames and
stereo PCM samples matched their controls. This measures weight-placement equivalence
within the existing audio recipe. Other durations and memory layouts need their own capacity evidence.

## Face repaint building blocks {#explicit-native-face-repaint}

These are opt-in engine APIs. Detection, tracking, crop/keyframe alignment, restoration models, source-audio
preservation and compositing belong to the application.

- `VideoRepaint` / `VideoRepaintRequest`: low-noise repaint with compiled FL/Ref modulation at the actual
  sigmas, preserved conditioned rows and frozen audio. Compile tables with
  `compile_refinement_tables(..., steps=2 or 4, denoise=0.035, mode="face")`. The native request runs on a
  preloaded `NativeEngineSession` with `generate_audio=False`; single device only.
- `vflash.media.face_encoder.face_encoder(vae, profile)`: scoped crop encoding for square 320/384/512 crops
  of 120–362 model frames. `batch2-fp16-norm` and `batch2-fp16-norm-reuse` are approximate FP16 batches
  (the latter reuses exactly equal clips); `batch2-fp16-norm-reuse-untiled` removes spatial tiling for
  512-square, 124-frame crops at higher memory. `original` keeps unchanged encoding.
- `H3Pipeline.source_scene_conditioning(source)` (development candidate): reuses the source scene's Qwen
  features for related video-only repaints. This deliberately changes conditioning.

The workflow was informed by the MIT-licensed [ComfyUI H3 FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine).
No face-restoration or identity weights are bundled.
