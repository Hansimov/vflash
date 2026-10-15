# Docker and API

Generate a complete video with the [pipeline image](#pipeline), or run the native denoiser as a local HTTP service. The service accepts a compiled conditioning bundle and returns video and audio latents (tensors ready for decoding). It loads one profile and reuses it across serial requests.

::: tip Published images
Version **0.9.0** is available as `hansimov/vflash:0.9.0` (native HTTP) and
`hansimov/vflash:0.9.0-pipeline` (complete MP4). Pin the digest from the
[image inventory](https://github.com/Hansimov/vflash/blob/v0.9.0/docker/images.json) for deployment.
Images contain dependencies, not model weights or application face-restoration configuration.
:::

The standard `runtime` and `pipeline` builds now include pinned Sol dependencies. Single-SM89
official Base16 defaults to approximate Sol; other profiles/pairs stay dense. Set
`VFLASH_ATTENTION_BACKEND=torch-flash` for dense HTTP execution, or add `--attention-backend torch-flash`
to `generate`/`denoise`. `/readyz` reports selection (not executed calls); job runtime metadata reports
the actual backend. Missing dependencies make the service unready rather than silently falling back.

## Skip unused Sol dependencies in a dedicated build

The published 0.9.0 images retain Sol for the supported single-SM89 Base16 `auto` path.
An installed backend is not necessarily executed: Veda requests select Veda explicitly.
Release source builds also accept `--build-arg VFLASH_INSTALL_SOL=0` for either
`runtime` or `pipeline`. This omits both Sol dependency installation and its source fetch;
the published general-purpose images retain their pinned Sol dependencies.

```bash
docker build --target runtime --build-arg VFLASH_INSTALL_SOL=0 -t vflash:veda-base .
```

Provision pinned Veda separately with `python -m vflash.install_veda` in a derived image,
and explicitly select `veda-triton` plus its predictor. Dense-only deployments select
`torch-flash`. A Sol-free image cannot serve a profile that still selects Sol; missing
optional dependencies remain a readiness error, with no silent fallback.

## Veda service configuration

The 0.9.0 service accepts `VFLASH_VEDA_DENSE_BACKEND` (default `torch-flash`) and
`VFLASH_VEDA_DENSE_EDGE_LAYERS` (default `5`; explicit `3` selects 44 sparse / 6 dense layers).
Select `VFLASH_ATTENTION_BACKEND=veda-triton` and provide `VFLASH_VEDA_PREDICTOR` as a
container path inside a read-only mounted runtime directory. Install the pinned Veda dependency
at image build time (`python -m vflash.install_veda`); the ordinary image does not silently install it
at startup. FROST requires its separately pinned SM120 dependency; it is not available on SM89.

`/readyz` includes the installed package version and selected partition. It checks architecture,
predictor and optional dense dependencies before declaring readiness; this does not mean inference
has run. The same options are available to `vflash denoise` as `--veda-dense-backend` and
`--veda-dense-edge-layers`. Preserve a pinned prior image for rollback and verify an actual complete
request in the consuming application before replacing its serving backend.

## Requirements {#requirements}

Use Linux AMD64 with Docker Compose v2, NVIDIA Container Toolkit, and an NVIDIA driver compatible with the image's CUDA 13.0 runtime. You also need one supported GPU or a cooperating 3080 pair from the [hardware list](./profiles) and all four [runtime inputs](../reference/runtime-assets).

For the tested 3080 workload, we recommend **at least 64 GiB of available host RAM per worker**, with additional headroom for other processes. Larger inputs need separate [capacity checks](./profiles#memory). Model assets are mounted separately; they are not included in the image.

## Configure and start {#start}

From the Vflash checkout:

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env`. Replace every example path with an absolute path on the Docker host:

```dotenv
VFLASH_IMAGE=hansimov/vflash:0.9.0
VFLASH_PROFILE_ID=ref2va-turbo4-exact-sm89
VFLASH_GPU_DEVICE=0

VFLASH_HOST_ARTIFACT=/path/to/artifact
VFLASH_HOST_SCHEDULE=/path/to/schedule
VFLASH_HOST_AUXILIARY=/path/to/auxiliary.safetensors
VFLASH_HOST_BUNDLES=/path/to/bundles
VFLASH_HOST_OUTPUTS=/path/to/outputs
```

`VFLASH_GPU_DEVICE` selects one host GPU by index or UUID. The container sees the selected GPU as device `0`. For a 3080, use `ref2va-turbo4-exact-sm86` and matching SM86 resources. For 4090 Turbo8, use `ref2va-turbo8-exact-sm89` and its corresponding resources.

The container runs as UID/GID `10001`. Create the output directory with write access for that user, then pull and start the published image:

```bash
sudo install -d -o 10001 -g 10001 /path/to/outputs
docker compose --env-file docker/.env -f docker/compose.yaml pull
docker compose --env-file docker/.env -f docker/compose.yaml up -d --no-build
```

Versioned images are published on [Docker Hub](https://hub.docker.com/r/hansimov/vflash/tags). Their immutable digests are recorded in [the image inventory](https://github.com/Hansimov/vflash/blob/v0.9.0/docker/images.json). Models stay mounted read-only; outputs and kernel caches use separate writable storage. To build version 0.9.0 locally, use `docker build --target runtime -t vflash:0.9.0 .` and select that image in your environment file.

The Compose configuration binds the API to **127.0.0.1:8000**. The engine has no built-in authentication. Keep this binding for local use, or put the API behind your application's authentication before allowing remote access.

## Generate an MP4 in a container {#pipeline}

The `pipeline` image includes the official encoder/VAE adapters, CUDA-matched Torchvision, FFmpeg and FFprobe. Its entry point is `vflash`; `generate` writes a complete video. The separate native image runs the HTTP latent service.

Pull the complete-pipeline image:

```bash
docker pull hansimov/vflash:0.9.0-pipeline
mkdir -p inputs outputs cache
```

Put your six-path `pipeline-assets.json`, prompt and reference images in `inputs`. In that JSON, use final container paths under `/models` for all model assets. Mount the containing model directory read-only. Prepare the receipt inside this final mount layout; this step hashes the assets once and needs no GPU:

```bash
docker run --rm --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,mode=1777 \
  -v /absolute/model-directory:/models:ro \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs:rw" \
  -v "$PWD/cache:/cache:rw" \
  hansimov/vflash:0.9.0-pipeline prepare-pipeline \
  --assets /inputs/pipeline-assets.json --receipt /outputs/prepared-assets.json
```

Generate on one RTX 4090 48 GB. Reference order determines the `<Picture N>` labels in your prompt:

```bash
docker run --rm --gpus device=0 --shm-size 4g \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,mode=1777 \
  -v /absolute/model-directory:/models:ro \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs:rw" \
  -v "$PWD/cache:/cache:rw" \
  hansimov/vflash:0.9.0-pipeline generate \
  --prepared-assets /outputs/prepared-assets.json \
  --prompt-file /inputs/prompt.txt \
  --reference /inputs/subject.png --reference /inputs/setting.png \
  --output /outputs/video.mp4 --gpu 0 --seed 1234 --trust-local-code
```

Use one to three `--reference` arguments for Ref4. For T2VA, prepare Base4 assets with `--profile t2va-turbo4-exact-sm89` and omit references during generation. Both produce five seconds at 24 fps. Progress JSON goes to stderr, the final result to stdout. No model weights are included in the image. Review the [complete pipeline contract and memory budget](./complete-pipeline) and [model licenses](../reference/license).

For two RTX 3080 20 GB GPUs, use [SM86-compiled assets](../reference/pipeline-profiles#sm86) and add `--profile ref2va-turbo4-exact-sm86` to `prepare-pipeline`. In the generation command, replace `--gpus device=0` with `--gpus '"device=0,1"'` and add `--peer-gpu 1 --strategy sequence-head` after `--gpu 0`. Keep both cards assigned to this process until it exits. The primary owns encoding and decoding; the pair cooperates in denoising. These are complete-pipeline options, separate from the native HTTP Compose setup below.

To build the complete image from the v0.9.0 tagged source, run `docker build --target pipeline -t vflash:0.9.0-pipeline .` and substitute that local image name in the commands.

The images reuse the immutable 0.8.0 base for their matching interface. Build the 0.9.0 wheel from the inventory’s `implementation_revision`, then use `docker/release-from-wheel.Dockerfile` with `VFLASH_BASE_IMAGE`, `VFLASH_VERSION=0.9.0` and `VFLASH_REVISION` set from that inventory. The recipe installs offline and updates the visible source as well as the package. Native CMD and pipeline ENTRYPOINT are inherited from the selected base. The ordinary root Dockerfile remains available for fresh dependency builds; do not redownload CUDA for a code-only update. The inventory separates image checks from GPU evidence.

The image defaults to UID/GID `10001`; these examples instead use your current user so outputs remain writable. Jiterator uses the writable `/cache` root directly, including a newly mounted empty directory. The explicit Inductor cache also works when that user has no account entry inside the image, including with the 0.1.0 image. `/cache` must permit loading compiled shared libraries: do not put it on a `noexec` mount. Asset paths and filesystem identities must remain the same as during preparation. Repeated work should use a persistent Python `H3Pipeline` in a dedicated container process, avoiding a new model load per command.

## Cooperating GPU pair {#parallel}

For two RTX 3080 20 GB devices, use the SM86 artifact and schedule paths, then set:

```dotenv
VFLASH_GPU_DEVICE=0
VFLASH_PEER_GPU_DEVICE=1
VFLASH_PARALLEL_STRATEGY=sequence-head
```

Use `tensor` for standard weight tensor parallelism. With Docker Compose **2.24.4 or later**, add the parallel override:

```bash
docker compose --env-file docker/.env \
  -f docker/compose.yaml -f docker/compose.parallel.yaml up -d --no-build
```

The override selects the SM86 Turbo4 profile, exposes exactly the two selected host devices, and reserves 1 GiB of container shared memory for NCCL. One worker owns the pair and processes requests serially. Use distinct GPU groups for additional workers. This mode has been measured on PCIe 3.0 x16 host-bridge links without peer access; it does not require NVLink.

The override replaces the device reservation using Compose's [`!override` merge rule](https://docs.docker.com/reference/compose-file/merge/#replace-value). Readiness checks both GPUs. A rank failure closes the pair; restart the service before submitting more work.

## Check readiness {#readiness}

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/profiles
```

`/healthz` checks that the HTTP process is alive. `/readyz` checks the configured files, output directory, GPU, and worker availability. The model loads on the first job; readiness does not mean it is already warm.

## Submit and download {#jobs}

If your bundle is stored at `VFLASH_HOST_BUNDLES/example-bundle`, submit its relative directory name:

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/denoise/jobs \
  -H 'content-type: application/json' \
  -d '{"bundle":"example-bundle"}'
```

Use the returned `id` to poll the job. Its status is `queued`, `running`, `succeeded`, or `failed`.

```bash
curl -fsS http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID
```

When the status is `succeeded`, download the latent output:

```bash
curl -fLo result.safetensors \
  http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID/output
```

This file contains tensors, not a playable video. Requesting output before a job succeeds returns `409`.

## Queue and recovery {#queue}

One CUDA worker runs one job at a time. The following optional settings in `docker/.env` control the service:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `VFLASH_API_PORT` | `8000` | Host port, bound to localhost by Compose |
| `VFLASH_JOB_TIMEOUT_SECONDS` | `1800` | Maximum time allowed for a worker request |
| `VFLASH_MAX_PENDING_JOBS` | `8` | Maximum queued and running jobs combined |
| `VFLASH_JOB_HISTORY_LIMIT` | `128` | Maximum completed job records retained in memory |

When capacity is full, submission returns `429` with `Retry-After`. Honor that delay and retry from your application.

Job records live in memory. Restarting loses them, and older completed records are eventually removed. Removing a record does not delete its output file, but its API lookup and download are no longer available. Download results promptly and let your application manage durable task records and file retention.

A failed or timed-out CUDA worker makes readiness fail until the service restarts. Jobs are not automatically replayed. Completed files remain in the output mount.

For timing fields and first-request overhead, see [performance measurement](../reference/performance).

## API reference {#api}

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | HTTP process health |
| `GET` | `/readyz` | File, GPU, and worker readiness |
| `GET` | `/v1/profiles` | The service's active profile |
| `POST` | `/v1/denoise/jobs` | Submit a bundle |
| `GET` | `/v1/denoise/jobs/{id}` | Read job status and result metadata |
| `GET` | `/v1/denoise/jobs/{id}/output` | Download a completed latent file |

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs` on your running service.

## Run without Docker {#python-server}

Install `.[gpu,server]`, set the paths and profile in your process environment, and start one server process:

```bash
python -m vflash.server
```

The Python server uses `VFLASH_ARTIFACT_PATH`, `VFLASH_SCHEDULE_OVERLAY_PATH`, `VFLASH_AUXILIARY_TENSOR_PATH`, `VFLASH_BUNDLE_ROOT`, and `VFLASH_OUTPUT_ROOT` for local paths, and `VFLASH_GPU_INDEX` for the physical GPU index. Set `VFLASH_API_HOST=127.0.0.1` for local access; the direct Python entry point otherwise binds to all interfaces. The profile, timeout, queue, and history settings use the names in this guide.

For a two-device Python service, set `VFLASH_PEER_GPU_INDEX` and optionally `VFLASH_PARALLEL_STRATEGY=tensor`. With a peer selected, the default strategy is `sequence-head`. Both selected physical devices are owned by one worker.

Dependency downloads and pinned Git source setup use separate cached layers. The Sol
Python dependency step also retains a BuildKit pip cache, so a source-network failure
does not discard successful wheel downloads. Select proxy bypasses per source using
Docker's predefined `HTTPS_PROXY`/`NO_PROXY` build arguments; keep values in the caller
environment. A loopback-only host proxy needs a reachable build route. Do not globally
change the build network just to retry one source: that invalidates unrelated dependency
layers. Mirrors must provide the pinned versions; compare a bounded real download first.
