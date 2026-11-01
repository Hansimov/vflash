import hashlib
import json
import sys

import pytest

from vflash.adapters import pika_flex_source as source
from vflash.contracts import ContractError
from vflash.pipeline.pika_flex import PIKA_SOURCE_REVISION


@pytest.fixture
def pinned_source(tmp_path, monkeypatch):
    root = tmp_path / "upstream"
    package = root / "src/pika_h3"
    (package / "core").mkdir(parents=True)
    contents = {
        "__init__.py": "",
        "pipeline.py": "raise RuntimeError('the full reference pipeline was imported')\n",
        "core/__init__.py": "",
        **{
            f"core/{name}.py": f"role = {name!r}\n"
            for name in ("sampler", "wrapper", "numerics", "packing", "geometry")
        },
    }
    inventory = {}
    for name, text in contents.items():
        path = package / name
        path.write_text(text)
        inventory[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    metadata = tmp_path / "metadata"
    (metadata / "data").mkdir(parents=True)
    (metadata / "data/pika_flex_source.json").write_text(
        json.dumps({"source_revision": PIKA_SOURCE_REVISION, "files": inventory})
    )
    monkeypatch.setattr(source, "files", lambda _: metadata)
    old = {k: v for k, v in sys.modules.items() if k == "pika_h3" or k.startswith("pika_h3.")}
    for name in old:
        monkeypatch.delitem(sys.modules, name)
    paths = list(sys.path)
    yield root
    for name in list(sys.modules):
        if name == "pika_h3" or name.startswith("pika_h3."):
            del sys.modules[name]
    sys.modules.update(old)
    sys.path[:] = paths


def test_prepared_runtime_imports_pinned_sampler_without_reference_pipeline(pinned_source):
    value = source.load_pika_sampling(pinned_source, trust_local_code=True)
    assert value.source_revision == PIKA_SOURCE_REVISION
    assert value.sampler.role == "sampler" and value.packing.role == "packing"
    assert "pika_h3.pipeline" not in sys.modules
    with pytest.raises(RuntimeError, match="full reference pipeline"):
        source.load_pika_source(pinned_source, trust_local_code=True)


def test_modified_sampler_is_rejected_before_any_code_runs(pinned_source):
    (pinned_source / "src/pika_h3/core/sampler.py").write_text(
        "raise AssertionError('executed')"
    )
    with pytest.raises(ContractError, match="content differs"):
        source.load_pika_sampling(pinned_source, trust_local_code=True)
    assert "pika_h3" not in sys.modules


def test_sampling_requires_same_explicit_source_trust(pinned_source):
    with pytest.raises(ContractError, match="explicit trust"):
        source.load_pika_sampling(pinned_source)
