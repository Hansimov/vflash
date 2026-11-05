"""Immutable identities for the independent Ref-picture / FL-sound candidate."""

from dataclasses import dataclass

from vflash.pipeline.pika_av import PikaAVRecipe


@dataclass(frozen=True)
class PikaReferenceRecipe(PikaAVRecipe):
    overlap_decode: bool = False
    reference_short_edge: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if (
            self.overlap_decode
            or self.decode_tile_batch_size != 1
            or (
                self.reference_short_edge is not None
                and (
                    type(self.reference_short_edge) is not int
                    or self.reference_short_edge not in (1024, 2048)
                )
            )
        ):
            raise ValueError(
                "Ref execution requires its sequential decoder and explicit image budget"
            )

    @property
    def identity(self):
        budget = (
            "tiered" if self.reference_short_edge is None else str(self.reference_short_edge)
        )
        sound = f"larry{self.audio_steps}" if self.audio_steps else "joint"
        return f"pika-ref6-{sound}-r{budget}-v1"

    @property
    def reference_encoding_policy(self):
        return {
            "rule": "tiered" if self.reference_short_edge is None else "explicit",
            "short_edge": self.reference_short_edge,
            "allow_upscale": True,
        }
