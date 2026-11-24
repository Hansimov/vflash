# Pika reference recipes and QAD visual weights

For runtime setup, start with [Pika H3 Flex](./pika-flex). Choose one of the prepared paths below according to the references, duration, available weights and device.

## Experimental Pika image-reference runtime

`PikaReferenceAVRuntime` is a separate prepared-weight candidate; the fixed-profile
CLI/server does not select it. It uses the original Ref adapter for six picture
forwards and an independent FL model for optional Larry native audio. It currently
accepts ordered image references within 1 MP, with original FP32 conditioning and
sequential GPU residency for the two transformers. The default recipe is five seconds.
For eight seconds, select `PikaReferenceRecipe(duration_seconds=8, audio_steps=10)` and a matching request. The 192 model frames use 57 latent frames and deliver 192 frames at 24 fps.
For twelve seconds, select `PikaReferenceRecipe(duration_seconds=12, audio_steps=10)`
and a matching `VideoRequest`. The reference encoder, Veda attention, FL audio and
decoder follow the same 294-frame model clock and deliver 288 frames at 24 fps.

Eight- and twelve-second audio can retain up to 12 GiB of frozen attention keys and values in
host RAM when the device has insufficient cache space. The cache keeps live text
and audio rows on their original step, and retains the CUDA decode reserve.
The audio-stage receipt reports the resident layer count and host-cache bytes.
This scope was exercised on one 96 GiB SM120 GPU with 720p delivery. Other durations,
equipment and strict dialogue fidelity require separate evaluation.

`reference_short_edge=None` retains the original tiered reference geometry. An
explicit `reference_short_edge=1024` or `2048` changes only the reference-image
encoding size, with upscaling enabled; the requested output canvas stays unchanged.
The resolved policy, source/encoded image sizes and reference-row count are included
in the conditioning receipt. An override is an approximate quality/performance
candidate, not an equivalence claim or a qualified default. Keep same-input controls
and check identity, detail, action/contact and sound on the delivered videos.


The experimental session accepts a frozen `PikaReferenceRecipe`. Its identity includes
`ref6`, the native soundtrack and the reference encoding budget, for example
`pika-ref6-larry10-r1024-v1`. It cannot be confused with the FL-only
`pika6-larry10-v1` recipe, and its budget must match the initialized runtime.

## Experimental QAD visual weights (0.26.0a2)

The Python `PikaAVRuntime` accepts an explicit `qad_packed_weights`, `qad_dense_weights`,
and `qad_manifest` set. Select `PikaAVRecipe(visual_weights="nvfp4-qad", audio_steps=10)`
to use the pinned FastVideo Pika QAD FL checkpoint for all six visual forwards and
restore the independently retained original FL model before Larry audio sampling.
The ordinary recipe remains unchanged. This is an experimental SM120 capability;
CPU tests establish ownership and arithmetic contracts. Six complete five-second T2VA/I2VA
outputs have also executed on RTX PRO 6000 with the pinned SM120 backend, native VAE
and independent Larry ten-step audio. Actual forward coverage, original-model restoration
and release of visual CUDA copies before audio KV allocation were checked. This is
execution evidence, not general quality acceptance or a matched-device speed benchmark.
Human comparison found weaker squash motion and slightly lower sharpness than the
original Pika visual path. Keep this option experimental and compare complete outputs
against the intended brief before choosing its speed/quality trade-off.

The model source is `FastVideo/Pika-H3-Flex-NVFP4-QAD` at
`0434eb89466c15ab451cb92fdd20995b4a2d1f1a`. Its Pika adapter is already merged in
300 main-block linears and the dense text refiner. Do not apply the adapter again.
Preparation must retain all 238 dense tensor descriptors and establish byte equality
of all 106 projection/time/output tensors with the original base before reusing the
original six-step tables. The manifest validation contract is in
`vflash.adapters.h3_qad.validate_manifest`; model licenses remain those of the sources.

Install the optional `qad` dependencies in addition to the complete pipeline. The
backend uses FlashInfer 0.6.18 CUTLASS W4A4, 16-value blocks and 128×4 scale layout;
precompile its SM120 quantization/GEMM modules in the matching CUDA/Torch environment
or provide a CUDA compiler for first use. No unsupported-device or arithmetic fallback
is provided. The original audio model stays resident. QAD keeps its immutable source tensors
mapped on the host and materializes CUDA copies only for the visual scope, releasing
them before the audio KV cache is built. Copy time remains part of complete latency.
Attention, native VAE and the audio recipe retain their explicit independent choices.
The execution receipt distinguishes visual NVFP4 from original row-FP8 sound and
records complete matrix/refiner coverage and restoration, including failure cleanup.

For a large Remake edit of lower-resolution source footage, the 0.28.0a5 preview accepts `RemakeRequest(…, half_size_source_reference=True)`. The source video is area-averaged by two on each axis before VAE encoding. Each reference image and pinned frame keeps its normal encoding. Generated dimensions, frame timing and the sampling schedule stay at their requested values. Both dimensions must be multiples of 64 and the short edge at least 960. The source mask requires full-size conditioning. The default uses full-size source conditioning. Evaluate regional detail, movement and correspondence with the original source for each new workload.
