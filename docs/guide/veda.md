# Select an attention backend

Attention selection belongs to a model recipe and a GPU architecture. Start with the profile’s default, then run a matched request with the alternative. A different sparse layout changes the computed picture.

## SM89 keyframe generation

For the original four-step LightX keyframe profile or its explicit FL/Ref hybrid, a single SM89 GPU can run `veda-sm89`. Install the pinned extension with `python -m vflash.install_veda` and supply the [H3 predictor](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) separately. Select both `--attention-backend veda-sm89` and `--veda-predictor FILE`. The installer checks the implementation revision. An incompatible predictor, dependency or GPU self-test stops the request.

## Configure the deployed recipe

Keep the sparse/dense partition, dense kernel, precision and model in the execution receipt. Hopper and Blackwell paths need their own kernels and qualification. A comparison should include a fine texture, a moving face and a hard edge across multiple frames as well as warm runtime and peak memory. See [published measurements](../reference/benchmarks).
