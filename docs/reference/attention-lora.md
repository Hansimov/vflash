---
search: false
---

# Load an attention adapter

AttentionAdapter applies a local FP32 DiT attention LoRA for a serial, single-SM89 Base16 keyframe session. Pass the local safetensors file, rank and scale together. Keep the adapter identity in the result record. The context restores base weights on exit. Other profiles and cooperating pairs require their own qualification.

[Read the versioned recipe and measurements](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/reference/attention-lora.md)
