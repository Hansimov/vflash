# Deploy an inference service

The native service keeps one configured profile and serializes denoising on its device. An HTTP client uploads no models: the server reads an existing bundle from a mounted directory. The published Docker images carry their own version (0.21.0). Build the current source tag when the deployment needs the new Python package.

## Configure storage and devices {#start}

From the tagged checkout, copy the environment template and set the image, profile, GPU and absolute host paths. Mount model inputs read-only. The output directory must be writable by the container user (UID/GID 10001). Start with a single GPU.

```bash
cp docker/.env.example docker/.env
```

```bash
sudo install -d -o 10001 -g 10001 /path/to/outputs
docker compose --env-file docker/.env -f docker/compose.yaml pull
docker compose --env-file docker/.env -f docker/compose.yaml up -d --no-build
```

## Build the 0.31.1 runtime

Run these commands in a v0.31.1 checkout when you need the current package. Use a distinct local image name so the published image remains independently identifiable. Set VFLASH_IMAGE in this shell so Compose starts the build you just verified.

```bash
docker build --target runtime -t vflash-local:0.31.1 .
docker run --rm --entrypoint python vflash-local:0.31.1 -c 'import vflash; print(vflash.__version__)'
export VFLASH_IMAGE=vflash-local:0.31.1
docker compose --env-file docker/.env -f docker/compose.yaml up -d --no-build
```

## Check the running service {#readiness}

Check health, readiness and the selected profile separately. A live process can still be waiting for a compatible GPU or valid model inventory. Submit a small known bundle and download the tensors to verify the complete path.

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/profiles
```

## Operate the queue

Use the [HTTP reference](../reference/http-api) for submission and retrieval. The service uses bounded in-memory history. Keep persistent job records and completed files in the calling application. Measure VRAM and available host RAM before placing another worker on the same host.

## Generate an MP4 in a container {#pipeline}

Put the model inventory in inputs/pipeline-assets.json and the prompt and reference images in inputs. Replace the host model directory with its absolute path. Use identical container paths in the two calls so the preparation receipt remains valid for generation.

```bash
docker pull hansimov/vflash:0.21.0-pipeline
mkdir -p inputs outputs cache
```

```bash
docker run --rm --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,mode=1777 \
  -v /absolute/model-directory:/models:ro \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs:rw" \
  -v "$PWD/cache:/cache:rw" \
  hansimov/vflash:0.21.0-pipeline prepare-pipeline \
  --assets /inputs/pipeline-assets.json --receipt /outputs/prepared-assets.json
```

```bash
docker run --rm --gpus device=0 --shm-size 4g \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,mode=1777 \
  -v /absolute/model-directory:/models:ro \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs:rw" \
  -v "$PWD/cache:/cache:rw" \
  hansimov/vflash:0.21.0-pipeline generate \
  --prepared-assets /outputs/prepared-assets.json \
  --prompt-file /inputs/prompt.txt \
  --reference /inputs/subject.png --reference /inputs/setting.png \
  --output /outputs/video.mp4 --gpu 0 --seed 1234 --trust-local-code
```

## Use a cooperating GPU pair {#parallel}

Select a supported pair of SM86 devices in docker/.env and its corresponding profile in the [hardware guide](./profiles). Start Compose with the two-device overlay.

```dotenv
VFLASH_GPU_DEVICE=0
VFLASH_PEER_GPU_DEVICE=1
VFLASH_PARALLEL_STRATEGY=sequence-head
```

```bash
docker compose --env-file docker/.env \
  -f docker/compose.yaml -f docker/compose.parallel.yaml up -d --no-build
```
