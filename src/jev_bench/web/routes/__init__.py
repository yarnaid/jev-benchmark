"""API routers mounted under /api."""

from fastapi import APIRouter

from jev_bench.web.routes import catalog, generations, runs, status

ROUTERS: list[APIRouter] = [status.router, catalog.router, generations.router, runs.router]
