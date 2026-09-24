"""FastAPI application factory: services lifespan, security headers, API routers and the static UI.

Constants:
    STATIC_DIR
Functions:
    create_app: app for the given Settings (optionally with an injected HTTP client, for tests).
        An unhandled exception from a route still gets the same security headers and never
        reveals its detail in the response body.
    create_default_app: uvicorn factory reading Settings from the environment.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from jev_bench.log_setup import configure_logging
from jev_bench.openrouter import build_http_client
from jev_bench.services import Services
from jev_bench.settings import Settings, load_settings
from jev_bench.web.routes import ROUTERS
from jev_bench.web.security import apply_security_headers, security_headers

STATIC_DIR = Path(__file__).parent / "static"


async def _unhandled(request: Request, exc: Exception) -> Response:
    body = {"detail": "internal server error"}
    return apply_security_headers(JSONResponse(body, status_code=500))


def create_app(settings: Settings, *, http: httpx2.AsyncClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = http or build_http_client(settings)
        services: Services | None = None
        try:
            services = Services(settings, client)
            services.sweep_interrupted()
            app.state.services = services
            yield
        finally:
            if services is not None:
                await services.jobs.shutdown()
            if http is None:
                await client.aclose()

    app = FastAPI(title="jev-bench", lifespan=lifespan)
    app.middleware("http")(security_headers)
    app.add_exception_handler(Exception, _unhandled)
    for router in ROUTERS:
        app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def create_default_app() -> FastAPI:
    configure_logging()
    return create_app(load_settings())
