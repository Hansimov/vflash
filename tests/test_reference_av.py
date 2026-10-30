"""The reference session refuses stretched inputs and unwinds ownership."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vflash.pipeline import SegmentedAVRecipe, SegmentedAVSession, VideoRequest


class ReferenceAVContractTests(unittest.TestCase):
    def test_invalid_canvas_restores_session_for_the_next_request(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Run the reference session boundary in the pinned Torch image")

        def encode(*args, **kwargs):
            return None

        schedulers = (object(), object())
        pipe = SimpleNamespace(
            transformer=SimpleNamespace(training=False),
            encode_prompt=encode,
            scheduler=schedulers[0],
            audio_scheduler=schedulers[1],
        )
        adapter = SimpleNamespace(strength=0.5)
        attention = SimpleNamespace(backend="sage")
        attention.select = lambda value: setattr(attention, "backend", value)
        session = SegmentedAVSession(
            pipe, None, adapter, attention, None, predictor=Path("unused")
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "source.png"
            Image.new("RGB", (704, 352)).save(first)
            request = VideoRequest(
                "A person waves.",
                first_frame=first,
                width=704,
                height=704,
            )
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "aspect-preserving"):
                    session.generate(request, root / "output")
                self.assertFalse(session._owner.locked())
                self.assertIs(pipe.encode_prompt, encode)
                self.assertEqual((pipe.scheduler, pipe.audio_scheduler), schedulers)
                self.assertEqual(adapter.strength, 0.5)

    def test_recipe_does_not_silently_change_evaluation_count(self):
        with self.assertRaises(ValueError):
            SegmentedAVRecipe(visual_steps=12)
        self.assertEqual(SegmentedAVRecipe(visual_attention="veda").audio_steps, 12)
