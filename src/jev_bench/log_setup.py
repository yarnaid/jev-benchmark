"""Process-wide loguru configuration for CLI and web entry points.

Functions:
    configure_logging: replace the default sink with one safe, no-diagnose stderr sink.
"""

import os
import sys

from loguru import logger

__all__ = [
    "configure_logging",
]


def configure_logging(*, debug: bool = False) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
        diagnose=False,
        backtrace=False,
        colorize=_colorize(),
    )


def _colorize() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stderr.isatty()
