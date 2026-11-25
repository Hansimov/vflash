# Prepare footage and finish an edited window

Use the 0.29.0 base package with FFmpeg and FFprobe on your path. Times are relative to the source picture’s start. This example takes three seconds from an H.264 MP4 longer than four seconds, preserves its composition on a square model canvas, then removes padding for delivery.

## Try the workflow with a local test clip

If you have no footage to hand, generate this five-second test pattern with a tone. Its moving edges make padding, stretching and the end of the selected interval easy to inspect. You can also use your own H.264 file.

```bash
ffmpeg -n -f lavfi -i testsrc2=size=640x360:rate=30:duration=5 \
  -f lavfi -i sine=frequency=440:duration=5 -c:v libx264 -pix_fmt yuv420p -c:a aac source.mp4
```

## Prepare three seconds of footage

```bash
curl -fL https://github.com/Hansimov/vflash/releases/download/v0.29.0/remake_window.py -o remake_window.py
python remake_window.py prepare source.mp4 prepared \
  --width 512 --height 512 --start 1 --duration 3 --soundtrack preserve
```

The new directory contains `window.mp4` and `window.json`. Keep them together. Preparing again with the same directory leaves both files untouched. The saved SHA-256 identifies the original prepared file when you export reference frames or restore its soundtrack.

For a square-pixel 16:9 source, the picture fills a 512 × 288 rectangle at `[0,112,512,288]`. Three seconds contain 72 frames. The shortest model window needs 90 frames, so the final picture is repeated 18 times. Display rotation and sample aspect are accounted for before fitting the image. Model windows use lengths of `5 + 17k` between 90 and 345 frames.

Use `--soundtrack preserve` for footage with an audio track. The selected sound retains its timing relative to the picture. Model padding is silent. Use `silent` for sources without sound. Choose new output paths.

## Extract reference pictures on the model canvas

Use the same preparation to extract frames 0, 30 and 60 from the same prepared window. Indices start at zero on its 24 fps clock and must fall within the real content.

```bash
python remake_window.py frames prepared/window.mp4 pins --receipt prepared/window.json --frames 0 30 60
```

The resulting `frames.json` lists the pictures in the order requested. Each PNG includes the entire model canvas and its padding. Paint the desired appearance on these pictures and pass the recorded index as `RemakeReference(path, frame=n)`.

## Edit and deliver

Use `prepared/window.mp4` as the source for [Remake](./remake). Set the request’s `model_frames` value (the Python request field is `frames`), width and height from the receipt. Prepare pinned target pictures on the same full model canvas. After the executor writes its complete result, run:

```bash
vflash finish-remake-window edited-window.mp4 result.mp4 --receipt prepared/window.json
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames -of json result.mp4
```

The example delivers 512 × 288 pixels, 72 frames at 24 fps. To verify the conversion before running a model, give the finish command the prepared `window.mp4` as input. Finishing re-encodes the content rectangle. Keep the model output and receipt for later exports. The calling application owns ordering and joining multiple windows.

## Verify audio coverage or use the original sound

The example above has already prepared the test tone with `--soundtrack preserve`. The 0.29.0 wheel checks for an audio stream on export. That stream can end before the selected picture does. The companion decodes the audio and checks continuous sample coverage through the last content frame. A gap in the middle is detectable even when the reported duration is correct. Encoded silence counts as valid audio. If the checker reports an interruption, keep the generated picture and run the second command below. To use the original sound, provide the window from **the same preparation**; its audio will accompany the edited picture.

```bash
python remake_window.py finish edited-window.mp4 checked.mp4 --receipt prepared/window.json
python remake_window.py finish edited-window.mp4 original-sound.mp4 --receipt prepared/window.json --original-audio prepared/window.mp4
```

Earlier receipts from `vflash prepare-remake-source` remain usable. They record geometry and timing. To distinguish original files across similarly sized clips, create a paired directory with the `prepare` command above.
