"""Original LightX and Larry activation-space adapters on the unpruned H3 model."""

import torch
from safetensors import safe_open


def converted(name, down, up, kind):
    if kind == "lightx":
        return [(name, down, up)]
    name = (
        name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.").replace(
            "blocks.", "transformer_blocks."
        )
        if name.startswith("blocks.")
        else name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
    )
    name = name.replace("final_layer.adaln_proj.linear", "norm_out.linear")
    if name.endswith(".attn.qkv_proj"):
        return [
            (name.replace(".attn.qkv_proj", ".attn.to_" + k), down, b)
            for k, b in zip(["q", "k", "v"], up.chunk(3, dim=0), strict=True)
        ]
    if name.endswith(".mlp.fc1"):
        gate, value = up.chunk(2, dim=0)
        return [
            (name.replace(".mlp.fc1", ".ff.net.0.proj"), down, torch.cat([value, gate], dim=0))
        ]
    return [
        (
            name.replace(".attn.out_proj", ".attn.to_out.0").replace(".mlp.fc2", ".ff.net.2"),
            down,
            up,
        )
    ]


class H3ActivationAdapter:
    def __init__(self, model, path, kind, *, validate_header=None):
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
            suffix = ".lora_A.default.weight" if kind == "lightx" else ".lora_A.weight"
            other = ".lora_B.default.weight" if kind == "lightx" else ".lora_B.weight"
            names = sorted(k[: -len(suffix)] for k in keys if k.endswith(suffix))
            assert keys == {n + s for n in names for s in [suffix, other]}
            assert len(names) == (312 if kind == "lightx" else 259)
            for name in names:
                down, up = read(name + suffix), read(name + other)
                scale = (
                    float(metadata.get("alpha", 8.0)) / down.shape[0]
                    if kind == "lightx"
                    else 1.0
                )
                for target, a, b in converted(name, down, up, kind):
                    module = model.get_submodule(target)
                    assert isinstance(module, torch.nn.Linear)
                    assert list(a.shape) == [a.shape[0], module.in_features] and list(
                        b.shape
                    ) == [module.out_features, a.shape[0]]
                    assert target not in self.calls
                    self.calls[target] = 0
                    if validate_header is not None:
                        continue
                    a, b = a.to("cuda:0").contiguous(), b.to("cuda:0").contiguous()
                    self.tensor_bytes += (a.numel() + b.numel()) * 2

                    def forward(module, args, result, *, a=a, b=b, scale=scale, target=target):
                        if self.strength == 0.0:
                            self.skipped += 1
                            return result
                        self.calls[target] += 1
                        delta = torch.nn.functional.linear(
                            torch.nn.functional.linear(args[0].to(a.dtype), a), b
                        )
                        return result + delta.to(result.dtype) * scale * self.strength

                    self.handles.append(module.register_forward_hook(forward))
            assert len(self.calls) == (312 if kind == "lightx" else 363)
        finally:
            if handle is not None:
                handle.__exit__(None, None, None)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
