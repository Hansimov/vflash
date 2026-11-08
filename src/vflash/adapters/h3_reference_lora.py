"""Declared original activation-space adapters on the unpruned H3 model."""

import torch
from safetensors import safe_open

from vflash.adapters.h3_lora_layout import LAYOUTS, adapter_names, adapter_scale, converted


class H3ActivationAdapter:
    def __init__(
        self, model, path, kind, *, validate_header=None, manifest=None, device="cuda:0"
    ):
        if kind not in LAYOUTS:
            raise ValueError("An explicit supported adapter family is required")
        if kind == "pika":
            from vflash.pipeline.pika_flex import validate_adapter_manifest

            if not isinstance(manifest, dict):
                raise ValueError("The original Pika release manifest is required")
            validate_adapter_manifest(manifest, manifest.get("task"))
        self.handles = []
        self.calls = {}
        self.tensor_bytes = 0
        self.strength = 1.0
        self.skipped = 0
        if validate_header is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            handle.__enter__()
            keys = set(handle.keys())
            metadata = handle.metadata()

            def read(k):
                return handle.get_tensor(k)
        else:
            keys = set(validate_header["tensors"])
            metadata = validate_header["metadata"]
            handle = None

            def read(k):
                return torch.empty(
                    validate_header["tensors"][k]["shape"], device="meta", dtype=torch.bfloat16
                )

        try:
            layout = LAYOUTS[kind]
            suffix, other = layout.suffixes
            names = adapter_names(keys, kind)
            for name in names:
                down, up = read(name + suffix), read(name + other)
                if down.ndim != 2 or down.shape[0] < 1:
                    raise ValueError("Adapter A must be a nonempty matrix")
                scale = adapter_scale(kind, metadata, down.shape[0], manifest)
                if kind == "pika":
                    # Match original PEFT BF16 execution, retaining its alpha/rank.
                    down, up = down.to(torch.bfloat16), up.to(torch.bfloat16)
                for target, a, b in converted(name, down, up, kind):
                    module = model.get_submodule(target)
                    if not isinstance(module, torch.nn.Linear):
                        raise ValueError("Adapter target must be an unpruned Linear projection")
                    if list(a.shape) != [a.shape[0], module.in_features] or list(b.shape) != [
                        module.out_features,
                        a.shape[0],
                    ]:
                        raise ValueError("Adapter projection does not match the target model")
                    if target in self.calls:
                        raise ValueError("An adapter cannot target a projection twice")
                    self.calls[target] = 0
                    if validate_header is not None:
                        continue
                    a, b = a.to(device).contiguous(), b.to(device).contiguous()
                    self.tensor_bytes += (
                        a.numel() * a.element_size() + b.numel() * b.element_size()
                    )

                    def forward(module, args, result, *, a=a, b=b, scale=scale, target=target):
                        if self.strength == 0.0:
                            self.skipped += 1
                            return result
                        self.calls[target] += 1
                        delta = torch.nn.functional.linear(
                            torch.nn.functional.linear(args[0].to(a.dtype), a), b
                        )
                        delta = delta.to(result.dtype)
                        # Unit scaling is exact and needs no full-output tensor pass.
                        # Preserve the two rounding points for nonunit coefficients.
                        if scale != 1.0:
                            delta = delta * scale
                        if self.strength != 1.0:
                            delta = delta * self.strength
                        return result + delta

                    self.handles.append(module.register_forward_hook(forward))
            if len(self.calls) != layout.target_pairs:
                raise ValueError("Converted adapter coverage differs from its declared family")
        except BaseException:
            self.close()
            raise
        finally:
            if handle is not None:
                handle.__exit__(None, None, None)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
