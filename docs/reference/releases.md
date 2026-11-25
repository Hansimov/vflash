# Upgrade to Vflash 0.29

## Changes from 0.28

[Prepare and finish footage](../guide/source-window) adds two CPU commands for source editing. Select an interval of a recorded H.264 clip, fit its displayed picture to the model canvas, and use the receipt to remove letterboxing and repeated frames from a generated window. Preserved audio retains its offset from the original picture.

When a Remake request uses a long generated take as history, its selected interval is checked against the presentation clock. Reference geometry follows EXIF display orientation. Each reference supplies one still image.

### Companion tool for 0.29.0

The release attachment `remake_footage_checked.py` runs with the original wheel. Extract full-canvas target PNGs, check audio coverage, or finish a generated picture with sound from the original prepared window. [Worked example and commands](../guide/source-window)

## Install the stable release

Install the wheel from [GitHub 0.29.0](https://github.com/Hansimov/vflash/releases/tag/v0.29.0) in a fresh environment and run `vflash prepare-remake-source --help`. Existing model snapshots and executor provisioning remain compatible. Docker image digests have a [separate inventory](https://github.com/Hansimov/vflash/blob/main/docker/images.json).

<span id="_0-16-1" aria-hidden="true"></span>
<span id="_0-17-0" aria-hidden="true"></span>
<span id="_0-18-0" aria-hidden="true"></span>
<span id="earlier" aria-hidden="true"></span>

[Earlier releases](https://github.com/Hansimov/vflash/releases)
