# Release notes

The current release is **0.1.0a5**, a developer preview of Ref2VA denoising. Its public interface accepts compiled conditioning and exports video/audio latents. See [the support table](../guide/profiles) before choosing a profile.

## Source preview · text-to-video {#t2va}

The source tree adds SM89 Base4 T2VA execution with explicit task, adapter and conditioning checks. This is not included in the `0.1.0a5` tag. Read the [preview scope](../guide/profiles#t2va), which separates one decoded-output smoke and implementation checks from broader quality qualification.

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

The [Docker guide](../guide/docker) builds the tagged source locally. We do not currently publish a prebuilt image in a public container registry. Keep using assets that match your profile; this update does not convert weights or conditioning bundles.

## 0.1.0a4 · lower host allocation {#a4}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a4)

Block streaming packs tensors into shared segmented pinned storage to reduce unused allocation space. Failed loads release their partially built resources. See the [4090 measurement](./benchmarks#sm89-host-memory) for its scope and numerical comparison.

## 0.1.0a3 · explicit memory placement {#a3}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a3)

Added explicit block streaming on a 4090 and completed-step callbacks for integrations. The [a3 dual-3080 measurements](./benchmarks#sm86-parallel) retain their original version and workload instead of being relabeled as current-release results.

## Availability {#availability}

The public package currently supports the listed **BF16 Ref2VA Turbo4/Turbo8 profiles**. The source tree also contains the T2VA preview described above. W8 quantization, live prompt/reference encoding and MP4 output are not released capabilities. Runtime assets and an end-user preparation command are not yet published.

New capabilities need their own installation path, hardware validation and decoded-output checks before they appear in the public support table. Applications can supply stages outside the current engine boundary; that does not make those stages part of this package.
