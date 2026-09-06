# Runtime assets

Vflash currently starts from already compiled inputs. Installing the package does not prepare model weights, encode a prompt, or download an example bundle. The public runtime asset pack and an end-user preparation command are not yet available.

This preview is intended for developers who already have compatible compiled assets.

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

The output is a safetensors file with video and audio latent tensors. It is suitable for a compatible decoder, which is outside the current public runtime.

Model and adapter files retain their own [licenses and terms](./license), independently of the Vflash source code.

## Adapter file verification {#adapter-files}

Turbo4 uses `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`; Turbo8 uses `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`. Their digests match the upstream repository checked on 2026-09-05 at revision `2f015e66b37c585cea9dc4ae6f1850ea8788e742`. This records a fixed-source check, not automatic compatibility with later revisions.

The T2VA profile uses the separate Base4 v1.0 file at the revision above. Its SHA-256 is `1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c`, with alpha 128 / rank 128. It is not interchangeable with Ref4 or newer Base4 releases.
