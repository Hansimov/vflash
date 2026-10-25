"""Pack prepared audio tables while preserving every BF16 table value."""

import json
import time
from pathlib import Path


def compact_audio_tables(source, output):
    from safetensors.torch import save_file

    from vflash.native.h3_audio_modulation_storage import pack_table
    from vflash.native.h3_refinement_schedule import load_tables

    source, output = Path(source), Path(output)
    manifest, _schedule, tables = load_tables(source, mode="audio")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    for record, table in zip(manifest["files"], tables, strict=True):
        path = output / record["file"]
        save_file(pack_table(table), str(path))
        records.append({"file": path.name, "size": path.stat().st_size})
    packed = {
        **manifest,
        "storage": "audio-table-dedup-v1",
        "files": records,
        "compaction_seconds": time.monotonic() - started,
        "source_table_bytes": sum(row["size"] for row in manifest["files"]),
        "stored_table_bytes": sum(row["size"] for row in records),
    }
    (output / "refinement.json").write_text(json.dumps(packed, indent=2))
    return packed
