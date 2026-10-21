# Get started

Vflash turns a prompt and optional images into a complete MP4 with audio. Most users want the
**complete pipeline** (`vflash generate`, `H3Pipeline` or the Docker pipeline image). Applications
that already build their own conditioning can use the lower-level **native denoiser**, which returns
latents instead of video ([below](#native-latents)).

Nothing is downloaded or compiled implicitly: you choose a profile that matches your GPU, prepare its
assets once, and Vflash refuses requests outside that profile's contract instead of silently changing
the model, device or precision.

## 1. Install {#install}

Linux, Python 3.11+, an NVIDIA driver for CUDA 13.0, and `ffmpeg`/`ffprobe` on `PATH`. The base
install is a light CLI without PyTorch; add the extra for the interface you use.

```bash
git clone --branch v0.17.2 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[pipeline]'     # complete MP4 pipeline (includes GPU dependencies)
```

Use `.[gpu]` for the native denoiser only and `.[server]` for the native HTTP service. The
[Docker images](./docker) pin the same environment (PyTorch 2.11 / CUDA 13.0) if you prefer containers.

The [GitHub v0.17.2 release](https://github.com/Hansimov/vflash/releases/tag/v0.17.2) also provides
a wheel and source archive from the same tag. Install the complete pipeline from the wheel:

```bash
python -m pip install "vflash[pipeline] @ https://github.com/Hansimov/vflash/releases/download/v0.17.2/vflash-0.17.2-py3-none-any.whl"
```

## 2. Check your GPU {#check-your-gpu}

```bash
vflash doctor      # GPUs visible to this process
vflash profiles    # profiles in this release
```

Pick the profile for your hardware and let `plan` check the match without loading weights:

| GPU | Typical profile | Command |
| --- | --- | --- |
| RTX 4090 48 GB | `i2va-turbo4-v01-544-exact-sm89` (keyframes, hybrid, native HD) | `vflash plan i2va-turbo4-v01-544-exact-sm89 --gpu 0` |
| RTX PRO 6000 96 GB | `i2va-turbo4-v01-544-exact-sm120` (hybrid image modes; see hardware limits) | `vflash plan i2va-turbo4-v01-544-exact-sm120 --gpu 0` |
| H100 SXM 80 GB | `i2va-turbo4-v01-544-exact-sm90` (hybrid I2VA) | `vflash plan i2va-turbo4-v01-544-exact-sm90 --gpu 0` |
| Two RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` | `vflash plan ref2va-turbo4-exact-sm86 --gpu 0 --peer-gpu 1 --strategy sequence-head` |

Text-only (`t2va-turbo4-exact-sm89`), reference-image (`ref2va-turbo4-exact-sm89`) and Base16
keyframe profiles are also available on the 4090. The memory sizes are requirements: 24 GB 4090s and
10/12 GB 3080s are not supported configurations. See [choose hardware](./profiles) for host RAM,
residency and the measured envelope of each GPU.

## 3. Prepare the model assets {#prepare}

A profile needs the official MiniMax H3 components at a pinned revision, its pinned Turbo LoRA (if any),
and three compiled native files (artifact, schedule and auxiliary tensors). Download and compile them
with the [official-weight compiler](./compile-weights), describe the six paths in a JSON file
([fields](./complete-pipeline#installation-and-assets)), then bind them once:

```bash
vflash prepare-pipeline --assets pipeline-assets.json --receipt prepared-assets.json
```

The receipt records the hash of every consumed file. Later starts only check identity and timestamps,
so keep the assets in their final read-only location; moving them requires a new receipt.

## 4. Generate a first video {#first-video}

Write the prompt to a UTF-8 file and run one request. With a first-frame image (I2VA):

```bash
vflash generate --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --first-frame first.png --width 928 --height 512 --duration 5 --seed 1234 \
  --gpu 0 --output video.mp4 --trust-local-code
```

- Text only: prepare `t2va-turbo4-exact-sm89` and omit the image arguments.
- Reference images: repeat `--reference` (refer to them as `<Picture 1>`, `<Picture 2>` … in the prompt).
- Last frame or both ends: `--last-frame`, or `--first-frame` plus `--last-frame`.

Progress is printed as JSON lines on stderr and the result on stdout. `--trust-local-code` allows the
official decoder code from your verified local snapshot; review the [licenses](../reference/license)
first. Each CLI call loads and releases the models, so the first run includes model loading.

Fifteen-second requests, up to nine references and native HD up to 2720×1536 use the hybrid model from
Python; the [generation guide](./complete-pipeline#extended-hybrid) shows how.

## 5. Keep the models loaded {#reuse}

For more than one request, keep one process alive:

- **Python:** one [`H3Pipeline`](./python) serves requests serially and reuses loaded weights.
- **Docker / HTTP:** the [native service](./docker) queues jobs and keeps its model resident.

Measure the first (cold) request separately from repeated (warm) requests; see
[speed, memory and quality](../reference/performance).

## Native latents (advanced) {#native-latents}

`vflash denoise` and the native HTTP service take a compiled conditioning bundle and write video and audio
**latents** (`.safetensors`), not an MP4. Use them only if your application owns conditioning and decoding:

```bash
vflash denoise ref2va-turbo4-exact-sm89 --gpu 0 \
  --artifact /path/to/artifact --schedule-overlay /path/to/schedule \
  --auxiliary-tensor /path/to/auxiliary.safetensors \
  --bundle /path/to/bundle --output-latents result.safetensors
```

Two RTX 3080s cooperate on one request with `--peer-gpu 1 --strategy sequence-head` (the default
strategy when a peer is selected; `tensor` is the alternative). Results need not be bitwise identical to
single-GPU execution. The [runtime assets](../reference/runtime-assets) page defines the four inputs.

## If a check fails {#troubleshooting}

| Symptom | What to check |
| --- | --- |
| No GPU listed | `nvidia-smi`; in Docker, that the device is exposed to the container. |
| Profile does not match | Choose the profile for the compute capability and memory `doctor` reports; renaming a profile never converts assets. |
| Assets rejected | Re-run `prepare-pipeline` on the final, unchanged files. |
| Request rejected before loading | Canvas, duration or input count is outside the [modes and limits](./complete-pipeline#modes-and-limits). |
| Out of host memory | Block-ring and SM86 workers keep weights in RAM: allow at least 64 GiB available per worker. |

More cases are in [troubleshooting](./troubleshooting).
