"""Request-bound, portable text conditioning for the original 24 fps Remake graph."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

FORMAT = "minimax-remake-text-v1"
ENCODER_SHA256 = "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"
MAX_TOKENS = 32768
MAX_BYTES = MAX_TOKENS * (5120 * 4 + 8) + 65536


@dataclass(frozen=True)
class PreparedRemakeText:
    path: Path
    sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.path, Path)
            or not isinstance(self.sha256, str)
            or not re.fullmatch("[0-9a-f]{64}", self.sha256)
        ):
            raise ValueError("A prepared text file and its SHA256 are required")


def text_segments(prompt: str, frames: int, references: int) -> list[str]:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65536:
        raise ValueError("A complete Remake prompt is required")
    if type(frames) is not int or not 90 <= frames <= 345 or (frames - 5) % 17:
        raise ValueError("Invalid Remake frame grid")
    if type(references) is not int or not 1 <= references <= 4:
        raise ValueError("Expected one to four references")
    # The original node samples at 2 fps, pads the last pair and labels its midpoint.
    stamps = [frame / 24 for frame in range(0, frames, 12)]
    if len(stamps) % 2:
        stamps.append(stamps[-1])
    labels = [
        f"<{(left + right) / 2:.1f} seconds>"
        for left, right in zip(stamps[::2], stamps[1::2], strict=True)
    ]
    return [
        "<Video 1>: ",
        *labels,
        *[f"<Picture {i + 1}>: " for i in range(references)],
        prompt,
    ]


def segment_digest(segments: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(segments, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _digest(path: Path) -> str:
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def _check_tensors(embedding, tags):
    import torch

    if (
        embedding.ndim != 3
        or embedding.shape[0] != 1
        or embedding.shape[2] != 5120
        or not 1 <= embedding.shape[1] <= MAX_TOKENS
        or embedding.dtype != torch.float32
        or tuple(tags.shape) != (embedding.shape[1],)
        or tags.dtype != torch.int64
    ):
        raise ValueError("Prepared text dimensions or dtype differ from the MiniMax encoder")


def prepare_remake_text(
    clip, *, prompt: str, frames: int, references: int, destination: Path
) -> PreparedRemakeText:
    """Encode with the pinned CLIPLoader(type='minimax') and publish a new owned file."""
    import torch
    from safetensors.torch import save_file

    if destination.suffix != ".safetensors":
        raise ValueError("Use a safetensors destination")
    segments = text_segments(prompt, frames, references)
    tokens = [t for segment in segments for t in clip.tokenize(segment)["qwen3vl_32b"][0]]
    with torch.inference_mode():
        positive = clip.encode_from_tokens_scheduled({"qwen3vl_32b": [tokens]})
    if len(positive) != 1:
        raise ValueError("Expected one scheduled conditioning")
    embedding, extra = positive[0]
    if (
        set(extra) != {"pooled_output", "minimax_token_tags"}
        or extra["pooled_output"] is not None
    ):
        raise ValueError("Unsupported MiniMax conditioning fields")
    tags = extra["minimax_token_tags"]
    _check_tensors(embedding, tags)
    metadata = {
        "format": FORMAT,
        "encoder": ENCODER_SHA256,
        "segments": segment_digest(segments),
    }
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".remake-text-", delete=False
    ) as file:
        tmp = Path(file.name)
    try:
        save_file(
            {
                "embedding": embedding.detach().cpu().contiguous(),
                "tags": tags.detach().cpu().contiguous(),
            },
            str(tmp),
            metadata=metadata,
        )
        digest = _digest(tmp)
        os.link(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)
    return PreparedRemakeText(destination, digest)


def owned_text(path: Path, root: Path) -> str:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if (
        path.is_symlink()
        or not resolved.is_relative_to(root)
        or not resolved.is_file()
        or resolved.suffix != ".safetensors"
        or not 0 < resolved.stat().st_size <= MAX_BYTES
    ):
        raise ValueError("Prepared text must be a bounded file in the owned input directory")
    return resolved.relative_to(root).as_posix()


class _PreparedClip:
    def __init__(self, embedding, tags, segments):
        self.embedding, self.tags, self.segments = embedding, tags, segments

    def tokenize(self, segment):
        # Preserve segment boundaries while the upstream node builds its scheduled input.
        return {"qwen3vl_32b": [[segment]]}

    def encode_from_tokens_scheduled(self, tokens):
        if segment_digest(tokens["qwen3vl_32b"][0]) != self.segments:
            raise ValueError(
                "Prepared text does not match the prompt, frame clock and reference count"
            )
        return [[self.embedding, {"minimax_token_tags": self.tags, "pooled_output": None}]]


class VFlashRemakePreparedText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"file": ("STRING",), "sha256": ("STRING",), "segments": ("STRING",)}
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "Vflash/Remake"

    def load(self, file, sha256, segments):
        import folder_paths
        from safetensors.torch import load

        root = Path(folder_paths.get_input_directory())
        path = root / file
        owned_text(path, root)
        # Verify and decode the very same bytes, including after a concurrent replacement.
        with path.open("rb") as stream:
            blob = stream.read(MAX_BYTES + 1)
        if len(blob) > MAX_BYTES or hashlib.sha256(blob).hexdigest() != sha256:
            raise ValueError("Prepared text content differs from the accepted file")
        header_size = int.from_bytes(blob[:8], "little")
        if not 2 <= header_size <= min(65536, len(blob) - 8):
            raise ValueError("Invalid prepared text header")
        metadata = json.loads(blob[8 : 8 + header_size]).get("__metadata__")
        if metadata != {"format": FORMAT, "encoder": ENCODER_SHA256, "segments": segments}:
            raise ValueError("Prepared text identity differs from this request")
        data = load(blob)
        if set(data) != {"embedding", "tags"}:
            raise ValueError("Unsupported prepared text fields")
        _check_tensors(data["embedding"], data["tags"])
        return (_PreparedClip(data["embedding"], data["tags"], segments),)
