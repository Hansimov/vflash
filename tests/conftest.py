"""The default suite must not claim an arbitrary visible serving GPU."""

import os

import pytest


def pytest_configure():
    if os.environ.get("VFLASH_TEST_CUDA") != "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise pytest.UsageError(
            "CUDA tests require an explicitly allocated CUDA_VISIBLE_DEVICES"
        )
