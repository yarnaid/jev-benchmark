"""Process-wide loguru configuration for CLI and web entry points.

Functions:
    configure_logging: replace the default sink with one safe, no-diagnose sink (stderr, or a
        caller's `write` callable such as a rich console that keeps a live progress bar intact).
"""

import os
import sys
from collections.abc import Callable

from loguru import logger

__all__ = [
    "configure_logging",
]


def configure_logging(*, debug: bool = False, write: Callable[[str], None] | None = None) -> None:
    logger.remove()
    logger.add(
        write or sys.stderr,
        level="DEBUG" if debug else "INFO",
        diagnose=False,
        backtrace=False,
        colorize=_colorize(),
    )


def _colorize() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stderr.isatty()
