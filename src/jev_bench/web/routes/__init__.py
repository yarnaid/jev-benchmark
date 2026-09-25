"""API routers mounted under /api."""

from fastapi import APIRouter

from jev_bench.web.routes import (
    analyses,
    catalog,
    compare,
    emails,
    generations,
    labels,
    questions,
    runs,
    status,
)

__all__ = [
    "ROUTERS",
]

ROUTERS: list[APIRouter] = [
    status.router,
    catalog.router,
    generations.router,
    runs.router,
    compare.router,
    emails.router,
    labels.router,
    questions.router,
    analyses.router,
]
