"""Identifier helpers for generations, runs and emails.

Functions:
    slugify: lowercase ASCII slug (at most 40 characters, never empty).
    new_id: `YYYYMMDD-HHMMSS-<slug>-<4hex>` from a timestamp converted to UTC.
    is_safe_id: whether a generation/run id is well-formed (safe as a path segment).
    split_email_id: `<generation_id>.<index>` -> (generation_id, index); KeyError when malformed.
"""

import re
import secrets
from datetime import UTC, datetime

__all__ = [
    "is_safe_id",
    "new_id",
    "slugify",
    "split_email_id",
]

_SAFE_ID = re.compile(r"\d{8}-\d{6}-[a-z0-9-]+")
_EMAIL_ID = re.compile(r"(?P<generation>\d{8}-\d{6}-[a-z0-9-]+)\.(?P<index>\d{4,})")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].strip("-")
    return slug or "x"


def new_id(label: str, now: datetime, suffix: str | None = None) -> str:
    stamp = now.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slugify(label)}-{suffix or secrets.token_hex(2)}"


def is_safe_id(value: str) -> bool:
    return _SAFE_ID.fullmatch(value) is not None


def split_email_id(email_id: str) -> tuple[str, int]:
    match = _EMAIL_ID.fullmatch(email_id)
    if match is None:
        raise KeyError(email_id)
    return match["generation"], int(match["index"])
