class VflashNativeError(Exception):
    """Base exception for expected, user-actionable failures."""


class ConfigError(VflashNativeError):
    """Configuration is invalid or incomplete."""


class CommandError(VflashNativeError):
    """A child command failed."""
