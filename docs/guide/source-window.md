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
vflash prepare-remake-source source.mp4 window.mp4 \
  --width 512 --height 512 --start 1 --duration 3 --soundtrack silent > window.json
```

For a square-pixel 16:9 source, the picture fills a 512 × 288 rectangle at `[0,112,512,288]`. Three seconds contain 72 frames. The shortest model window needs 90 frames, so the final picture is repeated 18 times. Display rotation and sample aspect are accounted for before fitting the image. Model windows use lengths of `5 + 17k` between 90 and 345 frames.

Use `--soundtrack preserve` for footage with an audio track. The selected sound retains its timing relative to the picture. Model padding is silent. Use `silent` for sources without sound. Choose new output paths.

## Edit and deliver

Use the prepared file as the source for [Remake](./remake). Set the request’s `model_frames` value (the Python request field is `frames`), width and height from the receipt. Prepare pinned target pictures on the same full model canvas. After the executor writes its complete result, run:

```bash
vflash finish-remake-window edited-window.mp4 result.mp4 --receipt window.json
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames -of json result.mp4
```

The example delivers 512 × 288 pixels, 72 frames at 24 fps. To verify the conversion before running a model, give the finish command the prepared `window.mp4` as input. Finishing re-encodes the content rectangle. Keep the model output and receipt for later exports. The calling application owns ordering and joining multiple windows.
