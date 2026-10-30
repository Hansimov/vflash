"""Owned reference-model components over declared original-base FP8 assets."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path


class ReferenceAVRuntime:
    """Load one original FL model and one Larry adapter for a segmented AV session.

    The text loader owns its declared quantized storage format; no application paths
    or environment variables are inferred here. The upstream reference source is the
    pinned VideoX-Fun checkout supplied explicitly by the caller.
    """

    def __init__(
        self,
        *,
        cache_root: Path,
        source_root: Path,
        tables: Path,
        adapter: Path,
        decoder_directory: Path,
        predictor: Path,
        text_loader,
        table_manifest: str = "manifest.json",
        trust_local_code: bool = False,
    ):
        import torch
        from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

        from vflash.adapters.h3_reference_assets import (
            SamplingClock,
            load_cached_trunk,
            validate_cached_structure,
        )
        from vflash.adapters.h3_reference_attention import H3ReferenceAttention
        from vflash.adapters.h3_reference_lora import H3ActivationAdapter
        from vflash.adapters.h3_reference_source import load_reference_modules
        from vflash.adapters.h3_text_encoder import retain_h3_text_encoder_prefix
        from vflash.media.runtime import OfficialMediaDecoder
        from vflash.pipeline.reference_av import SegmentedAVSession

        if not callable(text_loader):
            raise ValueError("Reference text weights require an explicit loader")
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError(
                "The reference FP8 runtime requires its supported SM120 allocation"
            )
        if torch.cuda.get_device_properties(0).total_memory < 90 * 1024**3:
            raise ValueError("The reference FP8 runtime requires a complete 96 GiB allocation")
        root, model_directory = cache_root.resolve(strict=True), cache_root / "model"
        self.closed = False
        self.adapter = self.attention = self.decoder = self.session = self.pipe = None
        self.clock = None
        self.initialization_stages = {}
        family = load_reference_modules(source_root, trust_local_code=trust_local_code)
        self.source_revision = family.revision
        try:
            started = time.monotonic()
            with torch.device("meta"):
                transformer = family.models.MiniMaxH3Transformer3DModel.from_config(
                    json.loads((model_directory / "transformer/config.json").read_text())
                )
                audio = family.models.AutoencoderKLMiniMaxH3Audio.from_config(
                    json.loads((model_directory / "audio_vae/config.json").read_text())
                )
            validate_cached_structure(transformer, root, tables, manifest_name=table_manifest)
            self.clock = SamplingClock(tables, manifest_name=table_manifest)
            load_cached_trunk(transformer, root, self.clock)
            self.adapter = H3ActivationAdapter(transformer, adapter, "larry")
            self.adapter.strength = 0
            self.attention = H3ReferenceAttention(family.models.MiniMaxH3Transformer3DModel)
            self.initialization_stages["denoise"] = time.monotonic() - started

            started = time.monotonic()
            text = text_loader()
            retain_h3_text_encoder_prefix(text)
            text.eval().to("cuda:0")
            tokenizer = Qwen2TokenizerFast.from_pretrained(
                model_directory / "tokenizer", local_files_only=True
            )
            processor = Qwen3VLProcessor.from_pretrained(
                model_directory / "processor", local_files_only=True
            )
            vae = family.models.AutoencoderKLMiniMaxH3.from_pretrained(
                model_directory,
                subfolder="vae",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).eval()
            vae.encoder.to("cuda:0")
            vae.quant_conv.to("cuda:0")
            self.pipe = family.pipeline.MiniMaxH3Pipeline(
                vae,
                audio,
                text,
                tokenizer,
                processor,
                transformer,
                family.scheduler(shift=12.0),
                family.scheduler(shift=3.0),
            )
            self.initialization_stages["conditioning"] = time.monotonic() - started

            started = time.monotonic()
            self.decoder = OfficialMediaDecoder(
                video_component=decoder_directory / "video_vae",
                audio_component=decoder_directory / "audio_vae",
                trust_local_code=trust_local_code,
            )
            self.decoder.resume_cuda()
            self.initialization_stages["decode"] = time.monotonic() - started
            self.session = SegmentedAVSession(
                self.pipe,
                self.clock,
                self.adapter,
                self.attention,
                self.decoder,
                predictor=predictor,
            )
        except BaseException:
            self.close()
            raise

    def generate(self, request, output, **kwargs):
        if self.closed or self.session is None:
            raise ValueError("The reference runtime is closed")
        return self.session.generate(request, output, **kwargs)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.adapter is not None:
            self.adapter.close()
        if self.attention is not None:
            self.attention.close()
        if self.decoder is not None:
            self.decoder.suspend_cuda()
        if self.clock is not None:
            self.clock.tensors.clear()
        self.session = self.pipe = self.adapter = self.attention = self.decoder = self.clock = (
            None
        )
        gc.collect()
