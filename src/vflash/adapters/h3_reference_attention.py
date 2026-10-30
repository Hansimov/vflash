"""Explicit SDPA, Sage or existing vflash 44/6 Veda on a reference-model layout."""

import sys
from types import SimpleNamespace


class H3ReferenceAttention:
    def __init__(self, model_type):
        self.module = sys.modules[model_type.__module__]
        self.original = self.module.attention
        self.backend = "sdpa"
        self.index = None
        self.veda = None
        self.handles = []
        self.reset()
        self.module.attention = self.attention

    def select(self, backend):
        assert backend in ("sdpa", "sage", "veda")
        self.backend = backend

    def reset(self):
        self.calls = {"sdpa": 0, "sage": 0, "veda": 0}
        self.prepared = False

    def configure(self, model, case, predictor):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.index = None
        self.profile = SimpleNamespace(
            frames=124,
            height=case["height"],
            width=case["width"],
            num_condition_video_rows=(case["height"] // 32) * (case["width"] // 32),
        )
        if self.backend == "veda" and self.veda is None:
            import torch

            from vflash.native.h3_veda_attention import VedaVideoAttention

            self.veda = VedaVideoAttention(
                torch.device("cuda:0"),
                predictor,
                backend="veda-triton",
                dense_backend="sageattention2",
                dense_edge_layers=3,
            )

        def before(_model, _args, kwargs):
            self.index = None
            if self.backend == "veda" and not self.prepared:
                self.veda.prepare(kwargs, self.profile)
                self.prepared = True

        self.handles.append(model.register_forward_pre_hook(before, with_kwargs=True))
        for index, block in enumerate(model.transformer_blocks):

            def enter(_block, _args, index=index):
                assert self.index is None
                self.index = index

            def leave(*_args):
                self.index = None

            self.handles.extend(
                (
                    block.register_forward_pre_hook(enter),
                    block.register_forward_hook(leave, always_call=True),
                )
            )

    def attention(self, query, key, value, *args, **kwargs):
        if self.backend == "veda" and self.index is not None:
            assert not args and kwargs == {"causal": False}
            self.calls["veda"] += 1
            return self.veda(
                self.index,
                lambda q, k, v: self.original(q, k, v, causal=False),
                query,
                key,
                value,
            )
        if self.backend == "sage" and tuple(query.shape[-2:]) == (56, 128):
            assert not args and kwargs == {"causal": False}
            from vflash.native.h3_veda_dense import sage_dense

            self.calls["sage"] += 1
            return sage_dense(query, key, value)
        self.calls["sdpa"] += 1
        return self.original(query, key, value, *args, **kwargs)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.module.attention == self.attention:
            self.module.attention = self.original
