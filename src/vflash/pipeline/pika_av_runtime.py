"""Explicit prepared-weight owner for Pika six-picture / Larry native-audio recipes."""

from pathlib import Path

from vflash.pipeline.reference_av_runtime import ReferenceAVRuntime


class PikaAVRuntime(ReferenceAVRuntime):
    """A resident SM120 FL allocation with pinned Pika and Larry components.

    All base, adapter, projection, text and decoder locations are caller supplied.
    This runtime does not infer application defaults or acquire a GPU. It replaces
    the inherited segmented session, so only one session owns the shared model.
    """

    def __init__(
        self,
        *,
        pika_source: Path,
        pika_adapter: Path,
        pika_manifest: dict,
        pika_tables: Path,
        trust_local_code: bool = False,
        **components,
    ):
        from vflash.adapters.h3_reference_lora import H3ActivationAdapter
        from vflash.adapters.pika_clock import PikaSamplingClock
        from vflash.adapters.pika_flex_source import load_pika_sampling
        from vflash.pipeline.pika_av import PikaAVSession

        self.pika_adapter = self.pika_clock = None
        sampling = load_pika_sampling(pika_source, trust_local_code=trust_local_code)
        super().__init__(trust_local_code=trust_local_code, **components)
        try:
            self.pika_adapter = H3ActivationAdapter(
                self.pipe.transformer, pika_adapter, "pika", manifest=pika_manifest
            )
            self.pika_adapter.strength = 0
            self.pika_clock = PikaSamplingClock(pika_tables)
            self.session = PikaAVSession(
                self,
                self.pika_adapter,
                self.pika_clock,
                sampling=sampling,
                predictor=components["predictor"],
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.pika_adapter is not None:
            self.pika_adapter.close()
            self.pika_adapter = None
        if self.pika_clock is not None:
            self.pika_clock.tensors.clear()
            self.pika_clock = None
        super().close()
