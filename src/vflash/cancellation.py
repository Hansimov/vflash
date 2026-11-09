"""Cancellation of caller-owned complete media execution."""


class GenerationCancelled(InterruptedError):
    """The caller stopped its owned generation before confirmed completion."""
