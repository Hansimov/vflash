"""Explicit local STCDiT-tiny loader for an isolated restoration process.

No third-party source or weights are vendored. The caller supplies and trusts a
checkout of the documented author revision and the original model artifacts.
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from typing import Any
from uuid import uuid4

from vflash.adapters.stcdit import StcditTinyRestorer

_WEIGHT_BYTES = {
    "diffusion_pytorch_model.safetensors": 5676070424,
    "models_t5_umt5-xxl-enc-bf16.pth": 11361920418,
    "Wan2.1_VAE.pth": 507609880,
    "tiny_8k.bin": 752596867,
    "google/umt5-xxl/tokenizer.json": 16837417,
    "google/umt5-xxl/tokenizer_config.json": 61728,
}


def validate_local_assets(weights: Path, source: Path) -> None:
    """Cheap inventory checks, not a content-digest or trust certification."""
    for name, size in _WEIGHT_BYTES.items():
        path = weights / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"missing or wrong-sized STCDiT artifact: {name}")
    for name in (
        "models/wan_video_dit_t2v_tiny.py",
        "models/wan_video_text_encoder.py",
        "models/wan_video_vae.py",
        "pipelines/wan_video_t2v_tiny.py",
    ):
        if not (source / "diffsynth" / name).is_file():
            raise ValueError(f"missing STCDiT runtime source: {name}")


def _provider_namespace(source: Path, prefix: str) -> Any:
    root = source / "diffsynth"
    for suffix in ("", ".models", ".pipelines", ".prompters", ".schedulers"):
        module = types.ModuleType(prefix + suffix)
        module.__path__ = [str(root / suffix.removeprefix("."))]
        sys.modules[module.__name__] = module
    # Avoid the upstream package initializer importing every unrelated model and
    # downloader. ModelManager is only a type annotation on this direct-load path.
    manager = types.ModuleType(prefix + ".models.model_manager")
    manager.ModelManager = object
    sys.modules[manager.__name__] = manager
    sys.modules[prefix + ".models"].ModelManager = object
    prompter = importlib.import_module(prefix + ".prompters.wan_prompter")
    sys.modules[prefix + ".prompters"].WanPrompter = prompter.WanPrompter
    for suffix in ("wan_video_dit_t2v_tiny", "wan_video_dit"):
        attention = importlib.import_module(prefix + ".models." + suffix)
        # Match the measured native PyTorch path, independent of H3 Sol defaults.
        attention.FLASH_ATTN_3_AVAILABLE = False
        attention.FLASH_ATTN_2_AVAILABLE = False
        attention.SAGE_ATTN_AVAILABLE = False
    name = prefix + ".pipelines.wan_video_t2v_tiny"
    module = types.ModuleType(name)
    module.__package__ = prefix + ".pipelines"
    module.__file__ = str(root / "pipelines/wan_video_t2v_tiny.py")
    tree = ast.parse(Path(module.__file__).read_text())
    # This plotting import is unused by the selected inference implementation.
    # Numerical definitions and model functions remain the author's code.
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Import)
            and len(node.names) == 1
            and node.names[0].name == "matplotlib.pyplot"
        )
    ]
    sys.modules[name] = module
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    return module


class LocalStcditTiny(StcditTinyRestorer):
    """Own one local BF16 restoration pipeline; not an H3 shared-process plugin."""

    def __init__(
        self,
        *,
        weights: Path,
        source: Path,
        trust_local_code: bool = False,
        device: str = "cuda:0",
    ) -> None:
        if trust_local_code is not True:
            raise ValueError("STCDiT requires explicit trust_local_code=True")
        validate_local_assets(weights, source)
        import torch
        from peft import LoraConfig
        from safetensors.torch import load_file

        if torch.device(device).type != "cuda":
            raise ValueError("the measured STCDiT runtime requires a CUDA device")
        self._prefix = "_vflash_stcdit_" + uuid4().hex
        self._closed = False
        self.pipeline = None
        self.device = device
        try:
            provider = _provider_namespace(source, self._prefix)
            models = self._prefix + ".models."
            initializer = importlib.import_module(models + "utils").init_weights_on_device

            def load(module_name: str, class_name: str, filename: str) -> Any:
                cls = getattr(importlib.import_module(models + module_name), class_name)
                state = (
                    load_file(weights / filename, device="cpu")
                    if filename.endswith(".safetensors")
                    else torch.load(
                        weights / filename, weights_only=True, map_location="cpu", mmap=True
                    )
                )
                converted = cls.state_dict_converter().from_civitai(state)
                values, kwargs = converted if isinstance(converted, tuple) else (converted, {})
                with initializer():
                    model = cls(**kwargs)
                model.load_state_dict(values, strict=True, assign=True)
                return model.to(dtype=torch.bfloat16, device="cpu").eval()

            pipe = provider.WanVideoPipeline_t2v_tiny(
                device=device,
                torch_dtype=torch.bfloat16,
                tokenizer_path=str(weights / "google/umt5-xxl"),
            )
            self.pipeline = pipe
            pipe.dit = load(
                "wan_video_dit_t2v_tiny",
                "WanModel_t2v_tiny",
                "diffusion_pytorch_model.safetensors",
            )
            pipe.text_encoder = load(
                "wan_video_text_encoder", "WanTextEncoder", "models_t5_umt5-xxl-enc-bf16.pth"
            )
            pipe.vae = load("wan_video_vae", "WanVideoVAE", "Wan2.1_VAE.pth")
            pipe.prompter.fetch_models(pipe.text_encoder)
            pipe.denoising_model_enable_lq_input()
            pipe.denoising_model_load_lora(
                LoraConfig(
                    r=128,
                    lora_alpha=128,
                    init_lora_weights=False,
                    target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
                )
            )
            adapter = torch.load(
                weights / "tiny_8k.bin", weights_only=True, map_location="cpu", mmap=True
            )
            if any(not key.startswith("pipe.dit.") for key in adapter):
                raise ValueError("unexpected STCDiT adapter namespace")
            adapter = {key.removeprefix("pipe.dit."): value for key, value in adapter.items()}
            expected = pipe.dit.state_dict()
            required = {
                key
                for key in expected
                if any(tag in key for tag in ("lora_", "lq_", "key_frame", "dw_conv"))
            }
            if not required <= adapter.keys() or adapter.keys() - expected.keys():
                raise ValueError("incomplete or incompatible STCDiT restoration adapter")
            pipe.dit.load_state_dict(adapter, strict=False)
            pipe.dit.to(dtype=torch.bfloat16)
            pipe.eval()
            pipe.enable_cpu_offload()
            super().__init__(pipe)
        except BaseException:
            self.close()
            raise

    def restore(self, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("the STCDiT runtime is closed")
        import torch

        with torch.inference_mode():
            return super().restore(*args, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        if self.pipeline is not None:
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize(self.device)
            self.pipeline = None
        for name in tuple(sys.modules):
            if name == self._prefix or name.startswith(self._prefix + "."):
                del sys.modules[name]
        self._closed = True

    def __enter__(self) -> LocalStcditTiny:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
