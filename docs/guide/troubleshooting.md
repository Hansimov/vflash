# Troubleshooting

Start with the smallest boundary that reproduces the problem. `vflash doctor` checks visible GPUs;
`vflash profiles` shows the supported profile contracts; `vflash plan` validates a proposed device
selection without loading model weights.

## Startup and assets

| Symptom | Check | Action |
| --- | --- | --- |
| No GPU appears | `nvidia-smi`; container device selection | Expose only the intended GPU or pair, then rerun `vflash doctor`. |
| Profile/device mismatch | Compute capability, memory and GPU count | Select the matching SM86 or SM89 profile. A profile name cannot convert an artifact. |
| Prepared assets rejected | Model, adapter, schedule and target identities | Re-run `prepare-pipeline` against the final read-only asset snapshot. Do not edit files behind an existing receipt. |
| Triton cannot load a compiled module | Cache path is writable and executable | Mount a persistent executable cache. A `noexec` temporary filesystem cannot load Triton shared libraries. |
| First request is much slower | Initialization, shape compilation and cold filesystem pages | Call `pipeline.prepare()`, warm every production shape, and report cold and warm latency separately. |

The base package deliberately contains no PyTorch or model payload. Install the `gpu`, `pipeline` or
`server` extras needed by the selected interface. `ffmpeg` and `ffprobe` must be present for MP4
delivery.

## Memory failures

Distinguish four different numbers: PyTorch allocated VRAM, PyTorch reserved VRAM, device-wide VRAM
and host RSS. A streamed 3080 worker holds most model weights in host memory, so a low VRAM figure
does not imply enough capacity.

- Stop unrelated workers before treating an out-of-memory result as a profile limit.
- Keep at least 64 GiB available host memory for the measured native worker and substantially more
  for the complete pipeline. These are starting budgets, not arbitrary-shape guarantees.
- Use the profile's tested canvas and duration before expanding one dimension at a time.
- Do not enable a second cooperative GPU implicitly. Select it explicitly and give the pair one
  lifetime owner.

After a CUDA failure, close the session. If completion or NCCL cleanup cannot be confirmed, exit the
worker process; do not reuse the context or reset a device owned by another process.

## Two-GPU execution

Both devices must have the same supported architecture and a compatible prepared artifact. The
current complete-pipeline cooperative strategy is `sequence-head`; other combinations fail before
model loading. Inspect both GPUs for existing processes and memory use before starting.

`sequence-head` uses a local NCCL group. A peer failure aborts the lane. A hang, timeout or partial
cleanup is a worker-terminal error, not a reason to retry collectives inside the same process.

The exact direct relayout is enabled by default. For a controlled regression check only, launch a
fresh worker with:

```bash
VFLASH_H3_DIRECT_RELAYOUT=0 vflash generate ...
```

This restores the previous materialized copies without changing the collective layout. It is a
diagnostic switch, not a throughput recommendation. See [Sol-Engine alignment](../reference/sol-engine-alignment).

## Output and quality

A successfully encoded MP4 proves neither prompt quality nor numerical equivalence. Check the
reported frame count, duration, dimensions, audio layout and completion first. Then review multiple
frames, normal-speed playback and actual audio against the prompt and references.

For unexpected first/last-frame detail, compare the conditioning image, the official VAE
reconstruction and any explicit `keyframe_delivery_profile` separately. Exact endpoint delivery is
a post-decode policy; it does not prove that intervening model frames retain the same detail.

When comparing a speed candidate, hold model, adapter, schedule, prompt, references, seed, canvas,
duration and device group fixed. A changed MP4 can be an implementation difference without being a
quality failure; it still requires full review before promotion.

## Useful diagnostic output

Use `--profile-denoise` for one attribution run, not every throughput run. Preserve the result JSON,
software versions, profile ID, GPU model/count, topology class and a bounded thermal trace. Do not
publish prompts, references, local paths, device UUIDs or model files. Do not hash multi-gigabyte
model payloads during each request; preparation receipts already bind immutable assets.

If the issue remains reproducible, include the smallest public-safe contract and whether cleanup was
confirmed when opening an issue.
