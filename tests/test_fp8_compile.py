"""Offline target compilation; no accelerator execution or quality claim."""

import re

import pytest

triton = pytest.importorskip("triton")
pytest.importorskip("torch")
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from vflash.native.h3_fp8_activation import _rows_fp8  # noqa: E402


@pytest.mark.parametrize("arch", [90, 120])
def test_large_strided_rows_widen_before_input_offset(arch):
    k = 14336
    stride = 2 * k
    m = (2**31 + stride - 1) // stride + 17
    code = triton.compile(
        ASTSource(
            _rows_fp8,
            {"X": "*bf16", "Q": "*fp8e4nv", "S": "*fp32"},
            constexprs={"M": m, "K": k, "STRIDE": stride, "BLOCK": 16384},
        ),
        target=GPUTarget("cuda", arch, 32),
        options={"num_warps": 8},
    ).asm
    assert (m - 1) * stride > 2**31
    # Row offsets must already be wide. The bounded within-row column may be i32.
    additions = [line for line in code["ttir"].splitlines() if "tt.addptr" in line]
    bases = [
        line
        for line in additions
        if any(f"tt.addptr %{name}," in line for name in ("X", "Q", "S"))
    ]
    assert len(bases) == 3 and all("i64" in line for line in bases)
    row_products = [line for line in code["ttir"].splitlines() if "arith.muli" in line]
    assert row_products and all("i64" in line for line in row_products)
    assert re.search(r"(?:mul|mad)\.wide\.[su]32|mul\.lo\.s64|shl\.b64", code["ptx"])
