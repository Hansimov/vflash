# Run Vflash with Docker

[中文](README.zh-CN.md) · [Full Docker and API guide](https://hansimov.github.io/vflash/guide/docker)

Use the [pipeline image](https://hansimov.github.io/vflash/guide/docker#pipeline) to generate a complete MP4 from text alone or a prompt and one to three references on SM89, or from references on two SM86 GPUs. The separate native HTTP service is described below.

The service accepts compiled conditioning bundles and returns video and audio latents (tensors ready for decoding). It uses one GPU or a cooperating pair and reuses a loaded model across serial requests.

You need Linux AMD64, Docker Compose v2, NVIDIA Container Toolkit, a CUDA 13.0-compatible driver, a supported GPU, and matching compiled runtime assets. The image does not contain model weights and serves the bundle-to-latent API. The separate [weights compiler](../docs/guide/compile-weights.md) prepares Ref4 assets for SM86/SM89 and Base4 assets for SM89; [complete MP4 generation](../docs/guide/complete-pipeline.md) uses the pipeline image or Python extra.

## Start the released image

From the repository root:

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env` with your absolute asset paths and selected GPU. Use `VFLASH_IMAGE=hansimov/vflash:0.2.2` for the published image. Choose one of:

| GPU | `VFLASH_PROFILE_ID` |
| --- | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89`, `ref2va-turbo8-exact-sm89` or `t2va-turbo4-exact-sm89` |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` |

Use resources compiled for SM86 on the 3080. For the tested workload, we recommend **at least 64 GiB of available host RAM per worker**, plus headroom for other processes. Check capacity separately for larger inputs.

Create your output directory with write access for UID/GID `10001`, replacing the example path with `VFLASH_HOST_OUTPUTS`, then pull and start:

```bash
sudo install -d -o 10001 -g 10001 /path/to/outputs
docker compose --env-file docker/.env -f docker/compose.yaml pull
docker compose --env-file docker/.env -f docker/compose.yaml up -d --no-build
curl -fsS http://127.0.0.1:8000/readyz
```

Compose binds the API to localhost. The engine has no authentication; remote access needs your application's authentication layer.

## Submit a bundle

For a bundle at `VFLASH_HOST_BUNDLES/example-bundle`:

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/denoise/jobs \
  -H 'content-type: application/json' \
  -d '{"bundle":"example-bundle"}'

curl -fsS http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID
```

After `succeeded`, download its output:

```bash
curl -fLo result.safetensors \
  http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID/output
```

The queue is bounded; overload returns `429` with `Retry-After`. Job records are temporary and disappear on restart or history eviction. Output files remain on disk. Download results promptly and let your application manage durable jobs and output retention.

See the [full guide](https://hansimov.github.io/vflash/guide/docker) for readiness, settings, timeout recovery, and all API endpoints. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs` on the running service.

For two RTX 3080 20 GB GPUs, add `docker/compose.parallel.yaml` and select `VFLASH_PEER_GPU_DEVICE`. See the [dual-GPU setup](https://hansimov.github.io/vflash/guide/docker#parallel) for `tensor` and `sequence-head` execution.
