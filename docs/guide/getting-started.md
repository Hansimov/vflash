# Install and verify Vflash

Start with the command-line tools. You can check your hardware and select an execution profile before downloading weights.

## Install the release {#install}

Use Python 3.11 or later in a new environment. This command installs the wheel published with the GitHub release.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install https://github.com/Hansimov/vflash/releases/download/v0.30.0/vflash-0.30.0-py3-none-any.whl
python -c "import vflash; print(vflash.__version__)"
```

The last command should print `0.29.0`. The base wheel supplies the CLI. Install the dependencies for your chosen runtime when you prepare its model.

You can run the [source preparation and export example](./source-window) on a CPU to verify the base package, FFmpeg, picture clock and sound. For model inference, select a GPU below.

## Check the device {#check-your-gpu}

```bash
vflash doctor
vflash profiles
```

Read the GPU name, compute capability and available memory inside the process or container that will run inference. Select a matching row in [hardware requirements](./profiles). On a suitable SM89 device, for example, validate a four-step reference-image plan with `vflash plan ref2va-turbo4-exact-sm89 --gpu 0`.


<span id="native-latents" aria-hidden="true"></span>

## Make a first film {#first-video}

1. Choose the [input and result](./runtime-selection) you need. Source footage, an appearance photo and a first frame each constrain generation differently.
2. For the fixed-profile CLI, [compile and prepare assets](./compile-weights) and retain the resulting receipt at its original location.
3. Run [the first-frame example](./complete-pipeline#first-frame), then inspect the MP4 and its audio stream. For another clip with the same assets, use a [resident Python session](./python).
