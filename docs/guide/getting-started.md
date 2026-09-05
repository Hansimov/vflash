# Get started

Install Vflash to inspect your GPU and choose a supported profile. To run inference, you will also need compatible compiled model assets and a conditioning bundle.

::: info Before you begin
This alpha release runs **compiled conditioning → video and audio latents (tensors ready for decoding)**. The runtime asset pack is not yet publicly available. Prompt processing, reference uploads, and MP4 output are not included in the current API.
:::

## Install the CLI {#install}

Use Python 3.11 or newer. The base installation is lightweight and does not download model weights or PyTorch.

```bash
git clone https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Check your GPU {#check-your-gpu}

```bash
vflash doctor
vflash profiles
```

`doctor` lists the NVIDIA GPUs visible through `nvidia-smi`. `profiles` lists the configurations supported by this release.

Choose the GPU index shown by `doctor` and check a profile:

```bash
# RTX 4090 with 48 GB
vflash plan ref2va-turbo4-exact-sm89 --gpu 0

# RTX 3080 with 20 GB
vflash plan ref2va-turbo4-exact-sm86 --gpu 0
```

`plan` checks the hardware match without loading model weights. For the 3080 profile, we recommend **at least 64 GiB of available system memory per worker for the tested workload**, plus headroom for other processes. Larger inputs need separate capacity checks. These profiles target the stated memory capacities; the usual 10/12 GB 3080 and 24 GB 4090 are outside this release's supported configurations.

See [profiles and hardware](./profiles) for the full support table.

## Run a bundle {#run-a-bundle}

Use [Docker](./docker) for a pinned GPU environment. For a Python source installation, install the GPU dependencies in your environment:

```bash
python -m pip install -e '.[gpu]'
```

The source runtime uses PyTorch 2.11 and Triton 3.6 on Linux with a compatible NVIDIA driver. The Docker build pins the CUDA 13.0 PyTorch build and its dependencies.

Prepare a matching [artifact, schedule, auxiliary tensor file, and conditioning bundle](../reference/runtime-assets). Replace the example paths with your files:

```bash
vflash denoise ref2va-turbo4-exact-sm89 \
  --gpu 0 \
  --artifact /path/to/artifact \
  --schedule-overlay /path/to/schedule \
  --auxiliary-tensor /path/to/auxiliary.safetensors \
  --bundle /path/to/example-bundle \
  --output-latents ./result.safetensors
```

For a 3080, select `ref2va-turbo4-exact-sm86` and use assets compiled for that profile. Renaming a profile or changing its GPU suffix does not convert the assets.

The command writes the video and audio latent tensors to `result.safetensors` and prints a JSON summary. The tensors are inputs for a compatible decoder; the file is not a playable video.

## Use two 3080s for one request {#parallel}

Select two RTX 3080 20 GB devices with the same SM86 Turbo4 assets:

```bash
vflash plan ref2va-turbo4-exact-sm86 --gpu 0 --peer-gpu 1 --strategy tensor
```

Add the same `--gpu`, `--peer-gpu`, and `--strategy` options to `vflash denoise`. Both GPUs belong to one request; this does not start two independent workers.

| Strategy | What is divided | Default when a peer is selected |
| --- | --- | --- |
| `tensor` | QKV, attention output, FFN, and LoRA projection weights | No |
| `sequence-head` | Token rows for GEMMs, then attention heads for complete-sequence attention | Yes |

`sequence-head` streams full weights to each GPU from one shared host copy. Four groups of heads overlap NCCL communication with attention. `tensor` streams half-sized weight shards and reduces projection results, including the native LoRA branches. Both execute the complete four-step schedule with BF16 weights and exact attention. No distributed launcher or LightX2V runtime is required.

Parallel execution changes GEMM shapes or reduction order. Results are **not bitwise identical to single-GPU execution**. Full latent and decoded-media smoke checks passed on one workload; cross-case instruction-quality qualification remains pending. See [performance measurement](../reference/performance#parallel) for scope and memory accounting.

## Reuse a loaded model {#reuse}

Each `denoise` command starts a fresh session. For repeated requests, use the [HTTP service](./docker): it loads a fixed profile on the first job and reuses that model for later jobs.

Python integrations can retain a `vflash.native.runner.NativeEngineSession` and call `session.generate(bundle, output_latents, progress_callback=on_step)`. The optional callback receives `(completed_steps, total_steps)` after each evaluation has completed on all selected GPUs. Omit it when step notifications are unnecessary; enabling it adds a device synchronization at each step. Close the session when its owner stops. A 4090 session can select `weight_residency="block-ring"` during construction to reserve more device memory for activations.

## If a check fails {#troubleshooting}

| Symptom | What to check |
| --- | --- |
| No GPU is listed | Run `nvidia-smi` and check driver access. In Docker, confirm that the selected GPU is visible in the container. |
| The profile does not match the GPU | Select the profile for the device and memory capacity shown by `doctor`. |
| Runtime assets are missing or incompatible | Check all four inputs and their model, adapter, schedule, and hardware versions. See [runtime assets](../reference/runtime-assets). |
| The 3080 process runs out of host memory | Free system memory or reduce the number of workers. Start with 64 GiB or more of available RAM per worker for the tested workload; check capacity for larger inputs. |
