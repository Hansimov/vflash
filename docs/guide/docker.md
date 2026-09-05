# Docker and API

Run Vflash as a local HTTP service. The service accepts a compiled conditioning bundle and returns video and audio latents (tensors ready for decoding). It loads one profile and reuses it across serial requests.

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
VFLASH_IMAGE=vflash:0.1.0a3
VFLASH_PROFILE_ID=ref2va-turbo4-exact-sm89
VFLASH_GPU_DEVICE=0

VFLASH_HOST_ARTIFACT=/path/to/artifact
VFLASH_HOST_SCHEDULE=/path/to/schedule
VFLASH_HOST_AUXILIARY=/path/to/auxiliary.safetensors
VFLASH_HOST_BUNDLES=/path/to/bundles
VFLASH_HOST_OUTPUTS=/path/to/outputs
```

`VFLASH_GPU_DEVICE` selects one host GPU by index or UUID. The container sees the selected GPU as device `0`. For a 3080, use `ref2va-turbo4-exact-sm86` and matching SM86 resources. For 4090 Turbo8, use `ref2va-turbo8-exact-sm89` and its corresponding resources.

The container runs as UID/GID `10001`. Create the output directory with write access for that user, then build the current source and start the service:

```bash
sudo install -d -o 10001 -g 10001 /path/to/outputs
docker compose --env-file docker/.env -f docker/compose.yaml up -d --build --pull never
```

These instructions build `vflash:0.1.0a3` locally from your checkout. The first build downloads the pinned runtime dependencies. Model resources stay mounted read-only; outputs and kernel caches use separate writable storage.

The Compose configuration binds the API to **127.0.0.1:8000**. The engine has no built-in authentication. Keep this binding for local use, or put the API behind your application's authentication before allowing remote access.

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
  -f docker/compose.yaml -f docker/compose.parallel.yaml up -d --build --pull never
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
