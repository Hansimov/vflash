# Edit a window of source footage

Use Remake when an existing shot supplies the action and timing. Prepare an `edit` window to change a scene or product surface, or a `swap` window for a replacement subject. The application selects the subject and prepares source-aligned references.

## Prepare the environment

Install the wheel’s `remake-client` extra for the caller. Provision the local Comfy runtime separately using [Viggle’s pinned setup](https://huggingface.co/Viggle/Viggle-Remake/blob/7e5be143702dd53b9154482de673b8519ed2d4d6/COMFYUI.md), ComfyUI revision `355cd6b14c92c19ac3188fe528e47a3f21ef5178`, the premerged INT8 model for the selected mode and its original VAEs. Register the Vflash conditioning nodes in that environment.

```bash
git clone --branch v0.28.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install '.[remake-client]'
```

```python
from vflash.pipeline.remake_conditioning import NODE_CLASS_MAPPINGS
```

## Prepare the source and target

Use a local H.264 MP4 on a 24 fps clock with square pixels, zero start time and matching canvas. Normalize rotation first. Use a target pin at an exact model frame only for a complete edited picture at that moment. A catalog photo can guide appearance as an unpinned reference. For a preserved soundtrack, check the final audio stream; for silence, request `silent`.

## Run once and recover

The following integration fragment assumes a runtime owner that can stop and confirm its process. Give each execution a fresh journal directory. After a caller restart, `recover_completed_remake` can read a completed execution with the original request, directories and unchanged files. For a pending or uncertain execution, reconcile its recorded ID before further work.

```python
from pathlib import Path
from vflash.pipeline.remake import RemakeReference, RemakeRequest
from vflash.pipeline.remake_execution import LocalRemakeRuntime, execute_remake

inputs = Path('/srv/remake/input')
request = RemakeRequest(
    source=inputs / 'source.mp4',
    prompt=Path('prompt.txt').read_text(),  # Original H3 six-section editing prompt.
    mode='swap', frames=124, width=832, height=480,
    references=(RemakeReference(inputs / 'edited-scene.png'),),
    soundtrack='preserve', seed=0,
)

# runtime_owner is the application's existing allocation/process controller.
# Its stop_and_confirm() must stop this runtime and return True after confirmation.
result = execute_remake(
    request,
    input_directory=inputs,
    output_directory=Path('/srv/remake/output'),
    execution_directory=Path('/srv/remake/executions/window-001'),
    runtime=LocalRemakeRuntime('http://127.0.0.1:8188'),
    stop_runtime=runtime_owner.stop_and_confirm,
    timeout_seconds=600,
    cancel_check=runtime_owner.cancelled,
)
print(result['artifact'], result['media'])
```
