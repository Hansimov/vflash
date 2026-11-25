# Edit a window of source footage

Use Remake when an existing shot supplies the action and timing. Prepare an `edit` window to change a scene or product surface, or a `swap` window for a replacement subject. The application selects the subject and prepares source-aligned references.

For a phone clip, screen recording or non-24-fps input, follow [prepare and finish footage](./source-window). Keep the preparation receipt to remove model padding after the edit.

## Prepare the environment

Install the wheel’s `remake-client` extra for the caller. Provision the local Comfy runtime separately using [Viggle’s pinned setup](https://huggingface.co/Viggle/Viggle-Remake/blob/7e5be143702dd53b9154482de673b8519ed2d4d6/COMFYUI.md), ComfyUI revision `355cd6b14c92c19ac3188fe528e47a3f21ef5178`, the premerged INT8 model for the selected mode and its original VAEs. Register the Vflash conditioning nodes in that environment.

```bash
git clone --branch v0.29.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install '.[remake-client]'
```

```python
from vflash.pipeline.remake_conditioning import NODE_CLASS_MAPPINGS
```

## Prepare the source and target

Use a local H.264 MP4 on a 24 fps clock with square pixels, zero start time and matching canvas. Normalize rotation first. Use a target pin at an exact model frame only for a complete edited picture at that moment. A catalog photo can guide appearance as an unpinned reference. For a preserved soundtrack, check the final audio stream; for silence, request `silent`.


<span id="extended-canvas-preview" aria-hidden="true"></span>

## Run once and recover

Place the preceding guide’s `window.mp4` and `window.json` in the Comfy input directory. Edit the opening frame on that same canvas and save it there as `edited-opening.png`. Put the six-section H3 editing prompt in `prompt.txt` in your working directory. This product edit reads its frame count, canvas and soundtrack directly from the preparation receipt.

The following integration fragment assumes a runtime owner that can stop and confirm its process. Give each execution a fresh journal directory. After a caller restart, `recover_completed_remake` can read a completed execution with the original request, directories and unchanged files. For a pending or uncertain execution, reconcile its recorded ID before further work.

```python
import json
from pathlib import Path
from vflash.pipeline.remake import RemakeReference, RemakeRequest
from vflash.pipeline.remake_execution import LocalRemakeRuntime, execute_remake

inputs = Path('/srv/remake/input')
window = json.loads((inputs / 'window.json').read_text())
request = RemakeRequest(
    source=inputs / 'window.mp4',
    prompt=Path('prompt.txt').read_text(),  # Original H3 six-section editing prompt.
    mode='edit', frames=window['model_frames'],
    width=window['width'], height=window['height'],
    references=(RemakeReference(inputs / 'edited-opening.png', frame=0),),
    soundtrack=window['soundtrack'], seed=0,
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

Pass the returned `artifact` path to the [finish command](./source-window). If the export is longer than the selected footage, check that the receipt belongs to this window and compare the exported frame count with its `content_frames`.
