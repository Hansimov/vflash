# How Vflash works

Vflash is a native H3 inference engine built with PyTorch and Triton. It owns the denoising execution path and uses compiled data instead of the LightX2V runtime or its model object graph.

## The current data path {#data-path}

```text
Compiled weights + schedule + auxiliary tensors
                      ↓
Conditioning bundle → Vflash session → Video and audio latents
```

A conditioning bundle supplies the encoded request and initial video/audio state. The session runs H3's packed multimodal transformer and updates both latent streams according to the selected schedule.

Encoding prompts and references happens before this boundary. Decoding the final latents and creating a playable media file happen after it. Those stages are not currently part of the public runtime.

## Two memory strategies {#memory-strategies}

On a **4090 with 48 GB**, compiled weights stay in GPU memory. Repeated requests reuse the loaded model, so they do not pay the loading cost again.

On a **3080 with 20 GB**, weights stay in pinned host memory and move through a small ring of device buffers. Transfer and compute overlap; a buffer is reused only after the GPU has finished reading its previous contents. This reduces the amount of VRAM needed at the cost of host memory and transfer work.

The hardware strategies share the same session interface. They do not imply identical performance or bitwise equality across GPU architectures.

## LoRA execution {#lora-execution}

The compiled resources include the selected adapter and schedule. The released profiles preserve the low-rank residual computations alongside the base weights and use exact attention. Changing the adapter changes the resources; it is not a per-request switch.

Vflash can use compatible LightX2V Turbo checkpoints without importing the LightX2V inference framework. Model compatibility and runtime implementation are separate concerns.

## Session and service ownership {#sessions}

The command-line runner creates one session for one call. The HTTP service starts one CUDA worker process, loads one fixed profile on its first job, and runs subsequent jobs serially in that process.

The worker owns its weights and CUDA state. Exiting it releases the device context. The HTTP process owns request validation, job status, and result downloads; it does not run model mathematics itself.

The service's job records live in memory. A larger application must provide durable task storage, user identity, resource scheduling, and any usage accounting it needs. See [Docker and API](../guide/docker) for the current operational behavior.
