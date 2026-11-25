# Run Vflash in Docker

Choose the image by its output. The native service reads mounted conditioning bundles and returns latents. The pipeline image generates MP4 files from prepared model assets.

| Image | Published linux/amd64 tag |
| --- | --- |
| Native HTTP | `hansimov/vflash:0.21.0` |
| Pipeline | `hansimov/vflash:0.21.0-pipeline` |

Pin an image by its [published digest](images.json). For features from package 0.29.0 build from that source tag using the [deployment guide](https://hansimov.github.io/vflash/guide/docker). Mount models read-only and allocate a writable output directory. Verify readiness and the package version inside the container before submitting a job.
