"""Errors of the static snapshot export.

Classes:
    ExportError: the export cannot produce a correct snapshot; the message says why.
"""

__all__ = [
    "ExportError",
]


class ExportError(Exception):
    pass
