# Prepare footage and finish an edited window

Use the released Python package with FFmpeg and FFprobe on your path. Times are relative to the source picture’s start. This example takes three seconds from an H.264 MP4 longer than four seconds, preserves its composition on a square model canvas, then removes padding for delivery.

## Try the workflow with a local test clip

If you have no footage to hand, generate this five-second test pattern with a tone. Its moving edges make padding, stretching and the end of the selected interval easy to inspect. You can also use your own H.264 file.

```bash
ffmpeg -n -f lavfi -i testsrc2=size=640x360:rate=30:duration=5 \
  -f lavfi -i sine=frequency=440:duration=5 -c:v libx264 -pix_fmt yuv420p -c:a aac source.mp4
```

## Prepare three seconds of footage

```bash
python -m vflash.media.remake_footage prepare source.mp4 prepared \
  --width 512 --height 512 --start 1 --duration 3 --soundtrack preserve
```

The new directory contains `window.mp4` and `window.json`. Keep them together. Preparing again with the same directory leaves both files untouched. The saved SHA-256 identifies the original prepared file when you export reference frames or restore its soundtrack.

For a square-pixel 16:9 source, the picture fills a 512 × 288 rectangle at `[0,112,512,288]`. Three seconds contain 72 frames. The shortest model window needs 90 frames, so the final picture is repeated 18 times. Display rotation and sample aspect are accounted for before fitting the image. Model windows use lengths of `5 + 17k` between 90 and 345 frames.

Use `--soundtrack preserve` for footage with an audio track. The selected sound retains its timing relative to the picture. Model padding is silent. Use `silent` for sources without sound. Choose new output paths.

### Deliver a replacement soundtrack

For a review cut with a separate music or narration file, first make one source with the picture and the chosen sound. This example assumes the soundtrack starts with the video and is at least as long as the picture. The video stream is copied unchanged.

```bash
ffmpeg -i performance.mp4 -i narration.wav -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -shortest with-narration.mp4
python -m vflash.media.remake_footage prepare with-narration.mp4 spoken-window \
  --width 512 --height 512 --duration 3 --soundtrack preserve
```

The prepared window owns the audio clock. After editing its picture, restore the sound from that same window and trim the model padding:

```bash
python -m vflash.media.remake_footage finish model-result.mp4 finished.mp4 \
  --receipt spoken-window/window.json --original-audio spoken-window/window.mp4
```

Listen to a distinct syllable or beat near each end of `finished.mp4`. For the three-second selection above, both picture and sound should end at three seconds. The intermediate model window can be longer.

## Extract reference pictures on the model canvas

Use the prepared video and its matching receipt to export frames 0, 30 and 60. Frame numbers start at zero on the 24 fps model clock and must fall inside the actual content.

```bash
python -m vflash.media.remake_footage frames prepared/window.mp4 pins --receipt prepared/window.json --frames 0 30 60
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

The footage module decodes the audio through the last content frame. Encoded silence counts as covered time. To restore a missing or truncated track, supply the window from the same preparation. Its digest ties the audio to the original selected interval.

```bash
python -m vflash.media.remake_footage finish edited-window.mp4 checked.mp4 --receipt prepared/window.json
python -m vflash.media.remake_footage finish edited-window.mp4 original-sound.mp4 --receipt prepared/window.json --original-audio prepared/window.mp4
```

Earlier receipts from `vflash prepare-remake-source` remain usable. They record geometry and timing. To distinguish original files across similarly sized clips, create a paired directory with the `prepare` command above.

## Use a delivered clip for a second edit

After replacing a presenter, you can use the delivered MP4 for a change of setting. Read the frame rate, frame count and duration of its video track first. Converting back to a source frame rate can move the end of the final displayed frame.

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate,nb_frames,duration -of json checked.mp4
python -m vflash.media.remake_footage prepare checked.mp4 second-window --width 512 --height 512 --soundtrack preserve
```

This prepares the next window from the beginning of the delivered clip. Use `--soundtrack silent` for a silent result. Use `second-window/window.mp4` and its new receipt for the next execution. Frames exported here belong to this second window. Compare both delivered edits with the original at the same playback times to check head turns, hands crossing the face and product positions.
