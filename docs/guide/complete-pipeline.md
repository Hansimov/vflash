# Generate a film from text or pictures

The fixed-profile CLI takes a prepared asset receipt and writes a playable MP4. Choose the input mode before compiling its transformer and schedule.

## Choose the input {#modes-and-limits}

| Mode | Supply | Effect |
| --- | --- | --- |
| T2VA | A prompt | Creates the whole shot |
| I2VA / L2VA | First / last image | Binds one endpoint |
| FL2VA | First and last image | Binds both endpoints in time order |
| Ref2VA | Ordered images | Guides appearance throughout the shot |
| Video reference | A supported short clip | Supplies motion or visual context |

The ordinary Ref4 recipe accepts one to three appearance images. Native keyframe profiles share first/last-frame weights. Check each canvas side on a 32-pixel grid and the mode’s joint pixel/frame budget before allocating a GPU.

## Prepare the runtime {#installation-and-assets}

Install the pipeline extra from the release checkout, then follow [weight compilation](./compile-weights). Keep the receipt, model snapshot and decoder at the paths used during preparation.

```bash
python -m pip install '.[pipeline]'
python -m vflash.install_sol   # only for the single-SM89 Base16 Sol default; no model downloads
```

```bash
vflash prepare-pipeline --assets pipeline-assets.json --receipt prepared-assets.json
```

## Animate a first frame {#first-frame}

Use a receipt prepared for a keyframe profile. Write what should move and what should be heard into a UTF-8 prompt file. This example uses a five-second landscape canvas.

```bash
vflash generate --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --first-frame first.png --width 928 --height 512 --duration 5 --seed 1234 \
  --gpu 0 --output video.mp4 --trust-local-code
```

## Compose appearance references {#generate-from-the-command-line}

Use a Ref4 receipt. The first file becomes `<Picture 1>`, the second `<Picture 2>`. State which supplies the subject and which supplies the setting. Check that the output opens, contains the intended subject and has the expected soundtrack.

```bash
vflash generate \
  --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --reference subject.png --reference setting.png \
  --width 928 --height 512 --seed 1234 --gpu 0 \
  --output video.mp4 --trust-local-code
```

## Reuse or extend a request {#generate-from-python}

Use [Python](./python) for repeated clips and [conditioning](./pipeline-conditioning) for the explicit extended hybrid. An existing shot that must retain its original motion belongs in [source editing](./remake).
