"""Status route: whether the server holds an OpenRouter key and an HF token.

Classes:
    StatusView
Functions:
    status: GET /status
"""

from fastapi import APIRouter
from pydantic import BaseModel

from jev_bench.web.deps import ServicesDep

__all__ = [
    "StatusView",
    "router",
    "status",
]

router = APIRouter(tags=["status"])


class StatusView(BaseModel):
    server_key: bool
    server_hf_token: bool


@router.get("/status")
def status(services: ServicesDep) -> StatusView:
    settings = services.settings
    return StatusView(
        server_key=settings.server_api_key() is not None,
        server_hf_token=settings.server_hf_token() is not None,
    )
