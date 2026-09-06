# Release notes

The current release is **0.2.0**. Generate a video from text alone or from one to three ordered reference images on an RTX 4090 48 GB. See [the support table](../guide/profiles) for the separate complete-video and native-latent interfaces.

## 0.2.0 · complete text-to-video {#v0-2-0}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.2.0)

The Python and container pipelines now generate a five-second, 24 fps MP4 without reference images. Select `t2va-turbo4-exact-sm89`, prepare the fixed Base4 v1.0 assets and omit `--reference`. Ref4 keeps its one-to-three-image interface. Each mode owns separate prepared assets and a persistent pipeline; mismatched requests are rejected before execution. The [model-profile guide](./pipeline-profiles) shows the complete T2VA commands.

The official-weight compiler now creates Base4 assets as well as Ref4 assets. On SM89, all 50 compiled Base4 blocks, four schedule tensors and nine auxiliary tensors matched independently prepared BF16 assets exactly. The actual installed container then completed two requests for one fixed 928 × 512 case: all 14 conditioning tensors and final FP32 video/audio latents matched independent controls. Every repeated raw video frame matched; both MP4s decoded fully with 120 frames and five-second stereo audio. Cancelling another request after its first denoising evaluation removed temporary output and retired the pipeline.

These checks establish implementation and resource lifetime for this workload, not broad instruction or audio quality. The Ref4 path retains its 0.1.0 multi-reference qualification. Small repeat differences in the official audio VAE remain possible. Both complete profiles use BF16, pinned LoRAs and strict original audio delivery; this release does not include W8, dynamic LoRA or first/last-frame generation. Complete SM86 generation is not yet qualified and is rejected; its native interfaces remain available.

The images also provide a writable Inductor cache for arbitrary user IDs. Final image checks cover installed package identity, supported profile contracts and CPU startup; the cache-directory and support-list changes preserve the GPU-qualified numerical implementation.

Published Linux AMD64 images: [`hansimov/vflash:0.2.0-pipeline`](https://hub.docker.com/r/hansimov/vflash/tags) for complete generation and `hansimov/vflash:0.2.0` for the native HTTP service. Both support anonymous pulls. The [immutable digest inventory](https://github.com/Hansimov/vflash/blob/v0.2.0/docker/images.json) records their source and qualification. Model weights are separate downloads under their own licenses.

## 0.1.0 · multi-reference video generation {#v0-1-0}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0)

Use `vflash generate` or `H3Pipeline` to produce a five-second, 24 fps MP4 from a prompt and one to three images on an RTX 4090 48 GB. Images keep their supplied order and `<Picture N>` labels. The original single-image Python argument remains compatible. `vflash prepare-pipeline` verifies model assets once before use. The [container guide](../guide/docker#pipeline) includes complete preparation and generation commands.

The actual installed pipeline image passed three sequential complete GPU requests with one, two and three references at 928 × 512. All 14 conditioning tensors and final FP32 video/audio latents matched their fixed controls. The first case also matched every raw FP32 video frame; the second matched all 120 delivered RGB frames. All three MP4s decoded fully with 120 frames and a five-second stereo audio track. Cancelling a three-reference request after its first denoising evaluation retired the pipeline, removed temporary output and released its resources.

These are implementation and lifecycle checks, not a broad instruction-quality or speed guarantee. The denoiser, LoRA arithmetic and official media decoding are unchanged from a7. Raw audio may vary slightly across repeated official VAE calls; audio bitwise reproducibility is not promised. References do not impose strict keyframe timing.

The complete image targets Ref4 on SM89. Ref8, T2VA Turbo4 and single/dual SM86 profiles retain their native bundle-to-latent interfaces. W8, dynamic LoRA and FL2VA are outside this release. Model weights remain separate downloads under their own licenses.

Published Linux AMD64 images: [`hansimov/vflash:0.1.0-pipeline`](https://hub.docker.com/r/hansimov/vflash/tags) for complete generation and `hansimov/vflash:0.1.0` for the native HTTP service. Both support anonymous pulls. The [immutable digest inventory](https://github.com/Hansimov/vflash/blob/v0.1.0/docker/images.json) identifies the exact images.

## 0.1.0a7 · prompt and image to MP4 {#a7}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a7)

A prompt and one image can now produce a five-second, 24 fps MP4 on one RTX 4090 48 GB. The Python pipeline owns the native denoiser and stage lifetimes, with explicit pinned official text, reference and VAE adapters. It supports sequential reuse, progress and cancellation. [Generate a video](../guide/complete-pipeline).

The [Ref4 compiler](../guide/compile-weights) builds native assets from fixed official H3 weights and Turbo4 v0.1, without a captured request or another project. Base weights and LoRA identities remain separate. All 1,250 layer tensors, four schedule tensors and nine auxiliary tensors matched the qualified BF16 assets exactly.

Complete checks reproduced conditioning and native latents exactly, with matching decoded video. Small variations in the official audio VAE remain under investigation; this is not a bitwise audio or broad quality guarantee. Total request timing now includes input preparation and cleanup, while weight transfers and stage calls are reported separately.

The complete pipeline and compiler initially cover Ref4 on SM89. T2VA Turbo4, Ref8 and SM86 keep their existing bundle-to-latent interfaces. Dynamic LoRA, FL2VA and W8 are not released. The Docker HTTP service also remains a latent interface. Model weights are downloaded separately under their own licenses.

## 0.1.0a6 · text-to-video {#t2va}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a6)

SM89 now supports T2VA Turbo4 with Base4 v1.0 assets. The engine checks the task, adapter identity and conditioning before execution, so Ref assets cannot accidentally process a text-only request. Applications can keep separate Base and Ref sessions to avoid loading different weights for each request.

One decoded-output case and repeated session calls passed implementation checks. Read the [validation scope](../guide/profiles#t2va) for the workload and remaining quality limits. The Ref profiles retain their existing weights and arithmetic. Newer Base4 adapters, T2VA Turbo8 and T2VA on SM86 are not part of this release.

Install this source tag using [the quick start](../guide/getting-started). The [Docker guide](../guide/docker) builds `vflash:0.1.0a6` locally; that version had no public prebuilt image.

## 0.1.0a5 · loading and resource ownership {#a5}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a5)

- Read groups of tensors from each immutable file into their final owned CPU storage. This avoids repeated header parsing and intermediate copies during loading.
- Release session-owned weights and pinned storage when the session closes, including when the caller retains the closed Python object.
- Report failed device or communication cleanup as a failure. Close a failed session and stop its worker if cleanup cannot be confirmed.

The model weights, Turbo adapters, denoising schedule and BF16 arithmetic are unchanged. These changes improve loading and resource lifetime; they are not a new cross-workload speed or quality claim.

Install from this tag for a fixed source release:

```bash
git clone --branch v0.1.0a5 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install -e .
```

The [Docker guide](../guide/docker) builds the tagged source locally. That version was distributed from source without a public prebuilt image. Keep using assets that match your profile; this update does not convert weights or conditioning bundles.

## 0.1.0a4 · lower host allocation {#a4}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a4)

Block streaming packs tensors into shared segmented pinned storage to reduce unused allocation space. Failed loads release their partially built resources. See the [4090 measurement](./benchmarks#sm89-host-memory) for its scope and numerical comparison.

## 0.1.0a3 · explicit memory placement {#a3}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a3)

Added explicit block streaming on a 4090 and completed-step callbacks for integrations. The [a3 dual-3080 measurements](./benchmarks#sm86-parallel) retain their original version and workload instead of being relabeled as current-release results.

## Availability {#availability}

The package provides the listed **BF16 Ref2VA Turbo4/Turbo8 and SM89 T2VA Turbo4 denoisers**, plus the **single-SM89 Ref4 and T2VA Python/container pipelines and official-weight compiler**. The compiler creates Ref4 and Base4 assets for SM89; assets for other native profiles must already be prepared. Published container images are available separately; model weights are not bundled.

New modes, adapters and hardware require their own installation, numerical and decoded-output checks before entering the support table.
