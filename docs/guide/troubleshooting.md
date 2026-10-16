# Troubleshooting

Start with the smallest check that reproduces the problem:

```bash
vflash doctor                       # GPUs this process can see
vflash plan <profile> --gpu 0       # profile/device match, no weights loaded
```

For a complete-pipeline request, `validate_request` runs the same admission rules as `generate` without
CUDA or model loading ([example](./complete-pipeline#check-a-request-before-allocating-hardware)).

## Request rejected before loading {#rejected}

Vflash never shrinks, re-times or reroutes a request to make it fit. The message names the rule:

| Message mentions | Cause | Fix |
| --- | --- | --- |
| width/height, multiple of 32, aspect ratio | Canvas outside 32-pixel steps or 1:4–4:1 | Choose a valid canvas. |
| more than 1,048,576 pixels / native HD | Above 1 MP without the hybrid image-conditioned path | Use the [hybrid model](./complete-pipeline#extended-hybrid) with a first-frame or reference image. |
| long edge 2720 / pixel-frame budget | `width × height × model_frames > 2048² × 124`, or a long edge above 2720 | Shorten the request (4 MP is a five-second request) or lower the canvas. |
| 6–15 s requires image conditioning | Long duration on T2VA or a non-hybrid profile | Use hybrid I2VA/FL2VA/Ref2VA. |
| resident I2VA supports… | `weight_residency="resident"` outside its envelope (SM90: 1536×864; SM120: the pixel-frame budget; I2VA with one first frame only) | Use `block-ring` for that request. |
| mode differs from the prepared profile | e.g. references sent to a keyframe profile | Prepare the profile for that mode. |

## Startup and assets {#startup}

| Symptom | Check | Action |
| --- | --- | --- |
| No GPU listed | `nvidia-smi`; container device selection | Expose only the intended GPU(s); in partitioned containers set `VFLASH_DEVICE_DISCOVERY=cuda-visible`. |
| Profile/device mismatch | Compute capability and memory reported by `doctor` | Choose the matching profile; renaming does not convert compiled assets. |
| Prepared assets rejected | Files moved or changed after `prepare-pipeline` | Re-run `prepare-pipeline` against the final read-only snapshot. |
| `trust_local_code` error | The official decoder code needs explicit trust | Pass `--trust-local-code` / `trust_local_code=True` after reviewing the snapshot. |
| Sol or Veda missing | `python -m vflash.install_sol` / `install_veda` not run, or wrong revision | Install the pinned revision. There is no silent fallback to another kernel. |
| Resident startup refused | `PYTORCH_ALLOC_CONF` not set before Python started | Export `PYTORCH_ALLOC_CONF=expandable_segments:True` in the launching environment. |
| Triton cannot load a module | Cache directory is read-only or `noexec` | Mount a writable, executable cache directory. |
| Imports resolve to another installation | A wrapper interpreter replaces `PYTHONPATH` | Install Vflash into that interpreter's environment, or prepend your paths to `sys.path` in the entry script. |
| First request much slower | Model loading, kernel compilation, cold page cache | Call `pipeline.prepare()` first and report cold and warm requests separately. |

The base package contains no PyTorch or model files; install the `pipeline`, `gpu` or `server` extra for the
interface you use, and keep `ffmpeg`/`ffprobe` on `PATH` for MP4 delivery.

## Memory failures {#memory}

Distinguish four numbers: PyTorch allocated VRAM, PyTorch reserved VRAM, device-wide VRAM and host RSS. The
complete-pipeline default `block-ring` keeps denoiser weights in host RAM, so a small VRAM figure does not mean
enough capacity.

- Stop unrelated GPU processes before treating an out-of-memory error as a profile limit.
- Budget host RAM generously: at least 64 GiB available per native worker, about 128 GB for the hybrid
  pipeline on SM90/SM120, and more for concurrent workers.
- Expand one dimension at a time from a measured canvas and duration ([choose hardware](./profiles)).
- After any CUDA failure, close the pipeline or session. If cleanup cannot be confirmed, exit the worker
  process instead of reusing its CUDA context.

## Two-GPU execution {#parallel}

Both devices need the same supported architecture and one prepared artifact; the complete pipeline uses
`sequence-head`. A peer failure aborts the request, and a hang or partial cleanup ends the worker; do not
retry collectives inside the same process. For a controlled regression check only,
`VFLASH_H3_DIRECT_RELAYOUT=0` restores the earlier materialized copies ([details](../reference/sol-engine-alignment)).

## Output and quality {#quality}

A playable MP4 proves neither prompt adherence nor numerical equivalence. Check frame count, duration,
canvas and audio first, then review several frames, normal-speed playback and the actual audio against the
prompt and references. For unexpected endpoint detail, compare the conditioning image, the VAE reconstruction
and any `keyframe_delivery_profile="exact-v1"` restoration separately. When comparing a speed option, keep
model, adapter, schedule, prompt, inputs, seed, canvas, duration and device fixed.

## What to include in a report {#report}

Use `--profile-denoise` for one attribution run. Include the result JSON, Vflash and dependency versions,
profile ID, GPU model and count, and a short thermal trace. Do not publish prompts, input media, local paths,
device UUIDs or model files.
