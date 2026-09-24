"""API routers mounted under /api."""

from fastapi import APIRouter

from jev_bench.web.routes import catalog, compare, emails, generations, labels, runs, status

ROUTERS: list[APIRouter] = [
    status.router,
    catalog.router,
    generations.router,
    runs.router,
    compare.router,
    emails.router,
    labels.router,
]
