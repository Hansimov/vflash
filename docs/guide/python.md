# Python integration

Use [`H3Pipeline`](./complete-pipeline) to generate complete videos from text or reference images on SM89. Use a `NativeEngineSession` when your application already prepares compatible conditioning bundles and needs to reuse loaded weights. A session owns one fixed profile and GPU group, and processes requests **serially**.

For process isolation and an HTTP queue, use the [Docker service](./docker) instead.

## Create a session {#session}

Install `.[gpu]` in the [supported environment](./getting-started#run-a-bundle), prepare the [four runtime inputs](../reference/runtime-assets), and replace the paths below. Select devices before your process initializes CUDA.

```python
from pathlib import Path

from vflash.catalog import ProfileCatalog
from vflash.hardware import discover_nvidia_devices
from vflash.native.runner import NativeEngineSession
from vflash.planner import resolve_plan

# Physical GPU indices are the ones reported by `vflash doctor`.
devices = {device.index: device for device in discover_nvidia_devices()}
plan = resolve_plan(
    ProfileCatalog.bundled(),
    profile_id="ref2va-turbo4-exact-sm89",
    device=devices[0],
)
outputs = Path("./outputs")
outputs.mkdir(parents=True, exist_ok=True)

with NativeEngineSession(
    plan,
    artifact=Path("/path/to/artifact"),
    schedule_overlay=Path("/path/to/schedule"),
    auxiliary_tensor=Path("/path/to/auxiliary.safetensors"),
) as session:
    for name in ("example-a", "example-b"):
        result = session.generate(
            Path("/path/to/bundles") / name,
            outputs / f"{name}.safetensors",
        )
        print(result["session"])
```

Each output contains video and audio latent tensors for a compatible decoder. It is not an MP4. The returned `session` fields separate [initialization and request timing](../reference/performance#timing).

## Select a 3080 pair {#parallel}

Replace the plan above and use the matching SM86 assets:

```python
plan = resolve_plan(
    ProfileCatalog.bundled(),
    profile_id="ref2va-turbo4-exact-sm86",
    device=devices[0],
    peer_device=devices[1],
    strategy="sequence-head",
)
```

Both devices belong to the same request. `tensor` is another supported strategy; omitting the peer selects single-device execution. See [supported hardware and memory](./profiles#memory) before adding concurrent workers.

## Receive completed-step progress {#progress}

Pass a callback when your application needs denoising progress:

```python
def on_step(completed: int, total: int) -> None:
    print(f"Denoising: {completed}/{total}")

# Within the active session:
result = session.generate(
    Path("/path/to/bundles/example-a"),
    Path("./outputs/example-a.safetensors"),
    progress_callback=on_step,
)
```

The callback runs after that evaluation finishes on every selected GPU. It is **denoising progress**, not a percentage of total video-generation time. Keep callbacks short. Enabling them adds synchronization at each step; omit the callback if you do not need notifications.

## Leave more VRAM for activations {#memory}

A 4090 session can pass `weight_residency="block-ring"` at construction to stream weights from host RAM. Its default is resident weights. The 3080 profile always streams blocks; it does not support full BF16 residency.

Measure both memory and latency with your own inputs. Streaming reduces device weight storage but needs substantial host RAM. The selected strategy remains fixed for the session.

## Close the session {#lifetime}

Use `with`, as above, or call `session.close()` when its owner stops. Closing waits for the selected devices, closes communication resources and releases owned weights. Keeping a reference to the closed Python object does not retain those weights.

The process still owns its CUDA context and allocator caches; process exit releases them. A closed session rejects new requests. After failed inference, close the session rather than trying to resume it. If device completion or cleanup fails, stop its worker process instead of reusing that CUDA context.

Use a separate spawned process for each independently owned GPU group. Do not call one session concurrently, initialize CUDA before device selection, or attempt to change its profile between requests. Changing models or adapters requires a new session with matching assets.
