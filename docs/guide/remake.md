# Edit a window of source footage

Use Remake when an existing shot supplies the action and timing. Prepare an `edit` window to change a scene or product surface, or a `swap` window for a replacement subject. The application selects the subject and prepares source-aligned references.

For a phone clip, screen recording or non-24-fps input, follow [prepare and finish footage](./source-window). Keep the preparation receipt to remove model padding after the edit.

## Prepare the environment

Install the wheel’s `remake-client` extra for the caller. Provision the local Comfy runtime separately using [Viggle’s pinned setup](https://huggingface.co/Viggle/Viggle-Remake/blob/7e5be143702dd53b9154482de673b8519ed2d4d6/COMFYUI.md), ComfyUI revision `355cd6b14c92c19ac3188fe528e47a3f21ef5178`, the premerged INT8 model for the selected mode and its original VAEs. Register the Vflash conditioning nodes in that environment.

```bash
git clone --branch v0.31.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install '.[remake-client]'
```

```python
from vflash.pipeline.remake_conditioning import NODE_CLASS_MAPPINGS
```

## Prepare the source and target

Use a local H.264 MP4 on a 24 fps clock with square pixels, zero start time and matching canvas. Normalize rotation first. Use a target pin at an exact model frame only for a complete edited picture at that moment. A catalog photo can guide appearance as an unpinned reference. For a preserved soundtrack, check the final audio stream; for silence, request `silent`.

To supply a portrait or product photograph independently of a frame, fit it to the selected video's model scale. The packaged command reads the display orientation and saves the selected region alongside its grid-aligned PNG.

```bash
python -m pip install "Pillow>=13,<14"
vflash prepare-remake-appearance portrait.jpg portrait-view --receipt prepared/window.json --box 250 40 740 650
```

Pass `RemakeReference(Path("portrait-view/appearance.png"))` for appearance guidance. Use `frame=n` for a complete edited picture aligned to a source instant. Version 0.30 also accepts a mixture of timed scenes and independent appearance pictures.

### Select a view of the portrait or location {#reference-view}

Choose left, top, right and bottom on the upright photo’s 0 to 1000 grid. Omit `--box` for the whole image and choose a new output directory. Give `portrait-view/view.png` to the image editor and use `portrait-view/appearance.png` as the independent reference for this video window. Both use the same original pixels. Frame the face and hair when retaining source clothes. For a setting, include enough of its layout and inspect bystanders at the edges.

### Compare products in the same shot

To make three complete versions, prepare the source window once. Give each version its own product photo and execution directory. Reuse the source, dimensions, frame count and seed. Paint a separate opening frame for each version if you need to control the object's position or its contact with a hand.

```bash
vflash prepare-remake-appearance tote.png tote-product --receipt prepared/window.json
vflash prepare-remake-appearance shoulder.png shoulder-product --receipt prepared/window.json
vflash prepare-remake-appearance flap.png flap-product --receipt prepared/window.json
```

In the Python request below, use the photos for **one** version at a time. For example, the first version can use `RemakeReference(Path("tote-product/appearance.png"))` by itself or after a tote-specific opening pinned to frame zero. Set separate output and execution directories before submitting the next version. Record the photo, request and finished file together so a resumed run retrieves the intended product.

Finish each full video against the same source receipt. Compare frames at the same time and inspect the product outline, the handle and the fingers around it. A common seed leaves room for movement and small hardware details to differ.

### Combine a scene frame with an appearance photo

For an edit changing a presenter and a setting, use the first picture for the desired scene at frame zero and a second picture for the presenter’s appearance. Picture numbers in the prompt follow tuple order. A window accepts up to four pictures.

```python
from pathlib import Path
from vflash.pipeline.remake import RemakeReference

references = (
    RemakeReference(Path("edited-opening.png"), frame=0),
    RemakeReference(Path("portrait-view/appearance.png")),
)
```

Pass this tuple to `RemakeRequest(references=references, ...)` in the execution example below. Each timed picture must match the prepared window canvas exactly. Frame zero starts at the beginning of that window. An independent portrait or product photo keeps its own composition. People in a location photo can influence the result. Use a photograph of the empty location when you want only its setting. Review the original gestures, people at the edges and product shape throughout the delivered clip.

### Budget the four reference pictures

A timed frame and an independent photo each occupy one slot. With one portrait you can guide the opening, a distinctive action and the end. With two photos you can guide two instants. Choose the interior instant by inspecting the source, especially when the visible side of a product changes briefly.

In the [three-second source-window example](./source-window), the active frames are 0–71. If the meaningful action occurs at frame 35, export the three source pictures together.

```bash
python -m vflash.media.remake_footage frames prepared/window.mp4 action-pins \
  --receipt prepared/window.json --frames 0 35 71
```

Edit each exported PNG on its own canvas. Keep the hand, contact point, camera angle and visible surface from **that instant**. Use the same new identity or finish across all three. Pass the three edited files with their original frame numbers and the portrait without a frame number. Verify the result just before and after the contact as well as at the pins.

The optional `keyframe_until` parameter is a sigma threshold between 0 and 1, default 0.9. Lower values keep the keyframes active into later denoising steps. Video timestamps still come from the frame numbers.


## Keep a held item or a brief graphic through an edit

For a background replacement, inspect a source frame where the hand and item overlap the old background. Edit that frame on the full model canvas and check that the fingertips, held item and contact point remain. For a short overlay, inspect frames before its appearance, late in the visible interval and after it disappears. Select up to four useful references within each window. Check both transitions in the finished film.

Use [the frame exporter](./source-window) to create `pins/frames.json`. Save your edited PNGs under the same filenames in `edited/`. The manifest carries the precise frame numbers and original order. Pass the resulting tuple as `references` in the request below.

```python
import json
from pathlib import Path
from vflash.pipeline.remake import RemakeReference

pins = Path("pins")
edited = Path("edited")
references = tuple(
    RemakeReference(edited / row["file"], frame=row["frame"])
    for row in json.loads((pins / "frames.json").read_text())
)
```

The [upstream prompt guide](https://huggingface.co/Viggle/Viggle-Remake/blob/main/PROMPTING.md) explains how the frame indices and six prompt sections work together.


<span id="extended-canvas-preview" aria-hidden="true"></span>

## Run once and recover

Place the preceding guide’s `prepared/window.mp4` and `prepared/window.json` in the Comfy input directory. Edit the opening frame on that same canvas and save it there as `edited-opening.png`. Put the six-section H3 editing prompt in `prompt.txt` in your working directory. This product edit reads its frame count, canvas and soundtrack directly from the preparation receipt.

The following integration fragment assumes a runtime owner that can stop and confirm its process. Give each execution a fresh journal directory. After a caller restart, `recover_completed_remake` can read a completed execution with the original request, directories and unchanged files. For a pending or uncertain execution, reconcile its recorded ID before further work.

Use the exported `frame-000.png` to paint the opening. For a later pin, pair that picture with its own zero-based index from the exported manifest.

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
