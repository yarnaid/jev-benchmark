"""Response models and helpers shared by route modules.

Classes:
    CancelView: whether a cancel request reached a running job.
Functions:
    load_or_404: load an item by id, turning a missing or unsafe id into HTTP 404.
"""

from collections.abc import Callable

from fastapi import HTTPException
from pydantic import BaseModel


class CancelView(BaseModel):
    cancelled: bool


def load_or_404[T](load: Callable[[str], T], item_id: str, label: str) -> T:
    try:
        return load(item_id)
    except KeyError as exc:
        detail = f"unknown {label} {item_id!r}"
        raise HTTPException(status_code=404, detail=detail) from exc
