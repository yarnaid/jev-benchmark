"""Status route: whether the server holds an OpenRouter key.

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


@router.get("/status")
def status(services: ServicesDep) -> StatusView:
    return StatusView(server_key=services.settings.server_api_key() is not None)
