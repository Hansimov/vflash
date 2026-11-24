---
search: false
---

# Retain and continue native latents

Pass a new latent_output path to H3Pipeline.generate to retain the completed audiovisual tensors before decoding. Keep this path distinct from the MP4. NativeAVPrefix also requires matching model tables, source frame count and context ownership. Stitch only the intended overlap once and check the audio boundary. Use [source editing](./remake) for edits to an encoded clip.

[Read the versioned recipe and measurements](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/guide/native-av-prefix.md)
