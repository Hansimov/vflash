# Runtime assets

The latent CLI starts from compiled inputs. Ref4 and T2VA Base4 on SM89 also have a [complete Python pipeline](../guide/complete-pipeline) and an [official-weight compiler](../guide/compile-weights). Installing the package does not automatically download model weights or example bundles.

The CLI contract below applies to developers supplying compatible compiled assets.

## The four inputs {#inputs}

| Input | Contents | CLI option |
| --- | --- | --- |
| Weight artifact | `artifact.json` and the compiled transformer blocks | `--artifact` |
| Schedule overlay | `overlay.json`, `schedule.safetensors`, and matching schedule data | `--schedule-overlay` |
| Auxiliary tensors | A `.safetensors` file for projections and other tensors outside the transformer blocks | `--auxiliary-tensor` |
| Conditioning bundle | `bundle.json`, `conditioning.safetensors`, and the bundle's declared files | `--bundle` |

The first three inputs belong to a fixed model/profile. Each conditioning bundle contains the input tensors for one request, including the encoded conditioning and initial latent state.

These are separate inputs because a service can load a model once and process several bundles without reloading its weights.

## Match the versions {#versions}

Use resources compiled for the selected GPU target, model revision, adapter revision, and schedule. Turbo4 and Turbo8 need different schedules; SM86 and SM89 need their respective target artifacts.

The released profiles pin these sources:

| Source | Revision |
| --- | --- |
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| [LightX2V Turbo4](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/83b617309219e859c1c264520eba07492d22e958) | `83b617309219e859c1c264520eba07492d22e958` |
| [LightX2V Turbo8](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/0eebcc7e79f9cb200927c80b8e7595265b770e34) | `0eebcc7e79f9cb200927c80b8e7595265b770e34` |
| [LightX2V Base4 v1.0](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/ec01fa4c86263832faa0bd1d6d8f36a281eaabb2) | `ec01fa4c86263832faa0bd1d6d8f36a281eaabb2` |

Conditioning capture hardware is recorded as provenance: an SM86 and an SM89 encoder capture can use the same model artifacts when their model identity, encoder revision, arithmetic profile, and tensor layout match. This does not promise identical conditioning tensors across GPUs. The denoiser artifact must still match its execution target.

Vflash validates the declared resource metadata. A file with the expected name is not enough: renaming a folder or changing a manifest cannot make mismatched weights compatible.

## Store inputs and outputs {#storage}

Keep model assets and bundles outside the source checkout. In Docker, mount them read-only and give the service a separate writable output directory. The [Docker guide](../guide/docker) shows the relevant settings.

The latent CLI returns a safetensors file with video and audio tensors. The complete Python pipelines handle encoding and official VAE decoding internally, then publish an MP4.

Model and adapter files retain their own [licenses and terms](./license), independently of the Vflash source code.

## Adapter file verification {#adapter-files}

Turbo4 uses `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`; Turbo8 uses `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`. Their digests match the upstream repository checked on 2026-09-05 at revision `2f015e66b37c585cea9dc4ae6f1850ea8788e742`. This records a fixed-source check, not automatic compatibility with later revisions.

The T2VA profile uses the separate Base4 v1.0 file at the revision above. Its SHA-256 is `1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c`, with alpha 128 / rank 128. It is not interchangeable with Ref4 or newer Base4 releases.

## Typed video conditioning {#video-conditioning}

The video adapter uses **bundle schema 2**, with a true `VideoReference` and `<Video 1>` in the encoded prompt. Existing image and T2VA bundles keep schema 1. Video frames are not separate picture references.

The native reader accepts one visual-only reference, 2–5 seconds at a normalized 24 fps, for Ref4 on one SM89 GPU. Output is 5 seconds at 24 fps, within 928×512 total pixels. Source audio, mixed image/video inputs, Ref8 and SM86 video conditioning are outside this boundary. This is reference-guided regeneration; it does not promise exact editing or preservation of every source frame.

The adapter must record the source file and decoded RGB hashes, source dimensions, normalized frame count, the actual official canvas, VAE input/latent frames and condition-video rows. The policy is `official-video-cfr24-v1`: decode at the source display size, then let the pinned official setup resize once. The final canvas has a maximum dimension of 1376, at most 1376×768 pixels and at most 33,024 condition-video rows. The VAE consumes complete temporal chunks; the text encoder samples the full normalized clip. These are different boundaries.

`H3ConditioningCaptureSession.finish(..., schema_version=2)` seals this metadata and the same 14 captured tensors. Loading checks source identity, the exact Ref4 schedule, tensor widths, modality indices and temporal prefixes. A schema change does not convert an existing image capture into video conditioning. The 0.3.1 `H3Pipeline` accepts a local `VideoRequest(reference_video=Path(...))` and creates this capture through the official video setup; see [input limits and examples](../guide/complete-pipeline#reference-video). The installed public pipeline completed an image/video/image sequence, checking 14 conditioning tensors, final FP32 audio/video latents, every delivered RGB frame and cancellation cleanup. The [release notes](./releases#v0-3-0) distinguish that implementation evidence from content quality and audio reproducibility.

## Experimental mixed FFN-in sidecar {#mixed-ffnin}

The sidecar's schema-1 `manifest.json` contains `profile_id`, `base_signature`,
`alpha: 0.5`, and ordered `layers`. Each layer has `block` and `files`; `q`,
`scale` and `d` each declare `bytes` and `sha256`. Files are
`block-NNN/{q,scale,d}.bin`: row-major INT8 weights, FP32 output-row scales and
FP32 input-channel balances. The fixed layer list and `artifact_signature`
live in `vflash.native.h3_mixed_ffnin`. Verify the sidecar with
`FFNInPayloads(sidecar, artifact, verify_content_hashes=True)` at ingestion;
normal starts check metadata, file sizes and scales without rehashing the model.
