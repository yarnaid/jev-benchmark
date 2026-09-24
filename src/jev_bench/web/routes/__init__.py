"""API routers mounted under /api."""

from fastapi import APIRouter

from jev_bench.web.routes import catalog, status

ROUTERS: list[APIRouter] = [status.router, catalog.router]
