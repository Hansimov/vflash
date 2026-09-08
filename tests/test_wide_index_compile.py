"""Compile the actual kernels for explicit CUDA targets without a driver/device."""

from __future__ import annotations

import importlib.util
import re
import unittest

from vflash.native import h3_fused_ops as fused


def kernels():
    _, modulate, gate = fused._strict_bf16_kernels()
    _, silu = fused._strict_bf16_silu_mul_kernel()
    _, merge, adapter_gate = fused._strict_bf16_adapter_kernels()
    _, rotary = fused._strict_bf16_rotary_kernel()
    return (
        modulate,
        gate,
        silu,
        fused._strict_bf16_ffn_adapter_silu_kernel(),
        merge,
        adapter_gate,
        rotary,
        fused._strict_bf16_qkv_direct_merge_kernel(),
    )


def compile_variant(kernel, arch, wide, *, large_elements=False):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = {}
    for name in kernel.arg_names:
        if name in {"BLOCK_SIZE", "WIDE_INDEX"}:
            continue
        if name.endswith("_ptr"):
            signature[name] = "*bf16"
        elif name.startswith("scaling"):
            signature[name] = "fp32"
        elif name in {"elements", "branch_elements"} and large_elements:
            signature[name] = "i64"
        else:
            signature[name] = "i32"
    return triton.compile(
        ASTSource(
            kernel,
            signature,
            constexprs={"BLOCK_SIZE": 1024, "WIDE_INDEX": wide},
        ),
        target=GPUTarget("cuda", arch, 32),
        options={"num_warps": 8},
    ).asm


def assert_address_width(assembly, wide):
    # Pointer-width widening after an overflowing i32 multiply is too late.
    # Check the type entering every actual Triton pointer addition.
    additions = [line for line in assembly["ttir"].splitlines() if "tt.addptr" in line]
    assert additions
    required = "i64" if wide else "i32"
    for line in additions:
        assert re.search(rf"tensor<\d+x{required}>", line), line
    if wide:
        assert re.search(r"mul\.wide\.[su]32|mul\.lo\.s64|shl\.b64", assembly["ptx"])


@unittest.skipUnless(importlib.util.find_spec("triton"), "requires the optional GPU compiler")
class OfflineAddressCompilation(unittest.TestCase):
    def test_wide_and_narrow_pointer_arithmetic_on_both_targets(self):
        for kernel in kernels():
            for arch in (86, 89):
                # Both strided views with an i32 logical size and contiguous
                # activations with an i64 logical size need the wide variant.
                for wide, large in ((False, False), (True, False), (True, True)):
                    with self.subTest(
                        kernel=kernel.__name__, arch=arch, wide=wide, large=large
                    ):
                        assembly = compile_variant(kernel, arch, wide, large_elements=large)
                        assert_address_width(assembly, wide)


if __name__ == "__main__":
    unittest.main()
