"""FastAPI application factory: services lifespan, security headers, API routers and the static UI.

Constants:
    STATIC_DIR
Functions:
    create_app: app for the given Settings (optionally with an injected HTTP client, for tests).
    create_default_app: uvicorn factory reading Settings from the environment.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from jev_bench.log_setup import configure_logging
from jev_bench.openrouter import build_http_client
from jev_bench.services import Services
from jev_bench.settings import Settings, load_settings
from jev_bench.web.routes import ROUTERS
from jev_bench.web.security import security_headers

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings, *, http: httpx2.AsyncClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = http or build_http_client(settings)
        services = Services(settings, client)
        services.sweep_interrupted()
        app.state.services = services
        try:
            yield
        finally:
            await services.jobs.shutdown()
            if http is None:
                await client.aclose()

    app = FastAPI(title="jev-bench", lifespan=lifespan)
    app.middleware("http")(security_headers)
    for router in ROUTERS:
        app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def create_default_app() -> FastAPI:
    configure_logging()
    return create_app(load_settings())
