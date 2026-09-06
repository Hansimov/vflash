# How Vflash works

Vflash turns an encoded H3 request into video and audio latents. Its PyTorch and Triton runtime owns the transformer, LoRA residuals, schedule updates and output heads. It does not import the LightX2V inference framework.

## The data path {#data-path}

```text
Compiled weights + adapter + schedule
                    ↓
Conditioning bundle → Native session → Video and audio latents
```

The conditioning bundle contains the encoded text and references, token layout and initial noise. Prompt/reference encoding runs before this boundary; VAE decoding and MP4 creation run after it. Those surrounding stages are not included in the public package yet.

Model, adapter, schedule and hardware identities must agree. Changing a profile name does not convert its assets or change the mathematics of its compiled request.

## One owner at each layer {#sessions}

| Layer | Owns | Lifetime |
| --- | --- | --- |
| CLI or HTTP service | Validation, job status and output paths | Command or server process |
| Native session | One fixed profile, GPU group and loaded runtime | Reused across serial requests |
| Request execution | Conditioning tensors, working latents and progress | One call to `generate` |

`native/runner.py` is the common session entrypoint. `native/h3_native_conditioning_runtime.py` connects validated inputs to the mathematical modules. The HTTP service delegates CUDA work to a spawned worker process; it does not build a second model execution path.

Use `NativeEngineSession` as a context manager, or close it explicitly. Closing first waits for the session's device group, then closes communication and releases owned weight references. Retaining the closed Python object does not retain those weights. The process still owns its CUDA context and allocator caches; exiting the worker releases them.

The HTTP worker exits after a failed trajectory. Python callers should close a failed session and never resume it. If device completion or cleanup cannot be confirmed, terminate its worker rather than reuse that CUDA context. The service's in-memory job records are separate from this execution lifetime; an application supplies durable storage, authentication and fleet scheduling.

## Loading weights once {#loading}

A tensor store indexes an immutable safetensors file without retaining an open file descriptor. Grouped reads validate the requested shapes, then fill the final CPU tensor storage directly. They avoid repeatedly parsing the same header and allocating an intermediate byte buffer followed by a clone. Each returned tensor owns its storage and remains valid after the file is closed.

Block streaming uses a different, scoped path: a temporary memory map supplies the compiled block, then tensors are copied into a shared segmented pinned-memory arena before the map closes. The arena spans blocks, reducing allocation padding. Large payload hashes are checked when assets are published; workers retain format, identity and size checks without rehashing the entire model on every request.

## Choosing memory placement {#memory-strategies}

On a **4090 with 48 GB**, the default is resident weights. Python integrations can select `weight_residency="block-ring"` to leave more VRAM for activations. Extra resident memory is useful only when it improves your workload; measure the first request and repeated requests separately.

On a **3080 with 20 GB**, weights remain in pinned system memory and stream through two device buffers. Copy events mark a buffer ready; compute events prevent its reuse until the previous operation finishes. The session reuses these resources across requests.

The two strategies share the same numerical execution interface. They do not promise equal performance or bitwise equality across GPU architectures.

## Two 3080s for one request {#parallel}

For a 3080 service focused on request latency, start with two GPUs using `sequence-head`. The caller selects the peer explicitly. A single-GPU session remains available; Vflash never acquires another GPU automatically.

A paired session owns two device rings, two submitting CPU threads and a local NCCL group. `sequence-head` shares one host weight store, divides token rows for projections, then exchanges attention data so each device processes complete sequences for its assigned heads. `tensor` instead streams weight shards and reduces partial projections, including the LoRA branches. No global `torch.distributed` group or distributed launcher is required.

Both strategies preserve the complete profile and use event-protected buffers. A rank failure aborts its peer. Partitioning changes floating-point reduction order, so parallel output need not be bitwise identical to single-GPU output. See the [measured scope](./performance#parallel).

## LoRA is part of the profile {#lora-execution}

The released Turbo profiles preserve the original low-rank residual computations alongside the base weights. Their compiled adapter and schedule are fixed for the session. Switching adapters requires compatible resources and a new session; it is not an unchecked per-request option.

Exact attention describes an implementation choice. It does not mean that a distilled Turbo profile has base-model quality. Judge a generated result against the task and references, not only against tensor similarity.
