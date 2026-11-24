---
search: false
---

# VDN keyframe compatibility

VDNEngineSession and VDNKeyframePipeline use a trained eight-step model. Existing SM89 integrations need its weights, rowwise FP8 loader and their original schedule. Recorded scope is 5–10 s I2VA and FL2VA on a 48 GB GPU. A VDN configuration cannot be selected by changing the attention backend of H3Pipeline. For new keyframe generation start with [model profiles](../reference/pipeline-profiles).

[Read the versioned recipe and measurements](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/guide/community-vdn.md)
