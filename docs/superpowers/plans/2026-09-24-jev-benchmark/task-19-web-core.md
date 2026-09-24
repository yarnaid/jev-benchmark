### Task 19: Web core — app factory, security headers, dependencies, status and catalog routes

**Files:**
- Create: `src/jev_bench/web/__init__.py`, `src/jev_bench/web/app.py`, `src/jev_bench/web/security.py`,
  `src/jev_bench/web/deps.py`
- Create: `src/jev_bench/web/routes/__init__.py`, `src/jev_bench/web/routes/status.py`, `src/jev_bench/web/routes/catalog.py`
- Create: `src/jev_bench/web/static/index.html` (minimal shell; Task 23 replaces it)
- Modify: `tests/factories.py` (`AppFactory`), `tests/conftest.py` (`make_app` fixture)
- Test: `tests/test_web_app.py`, `tests/test_web_security.py`, `tests/test_web_deps.py`,
  `tests/test_web_routes_status.py`, `tests/test_web_routes_catalog.py`

**Interfaces:**
- Consumes: `Settings`, `load_settings` (Task 1); `build_http_client`, `OpenRouterError` (Task 5);
  `ModelInfo`, `ColumnKind` (Task 6); `Services` (Task 16); `mini_settings`, `FakeOpenRouter`,
  `seed_generation` (Tasks 16–17).
- Produces:
  - `jev_bench.web.app`:
    - `STATIC_DIR`;
    - `create_app(settings, *, http=None) -> FastAPI`. The lifespan builds `Services`, runs
      `sweep_interrupted()`, stores the services on `app.state.services`, and on exit calls
      `jobs.shutdown()` and closes the HTTP client if it owns it;
    - `create_default_app() -> FastAPI` (the uvicorn factory).
  - `jev_bench.web.security`: `CSP` and the middleware `security_headers(request, call_next)`.
  - `jev_bench.web.deps`:
    - `NO_KEY_DETAIL`;
    - `get_services(request) -> Services`;
    - `ServicesDep`;
    - `require_api_key(services, x_openrouter_key)` (HTTP 400 when there is no key);
    - `ApiKeyDep`.
  - `jev_bench.web.routes.ROUTERS: list[APIRouter]`, every router mounted under `/api`.
  - `GET /api/status -> {"server_key": bool}`.
  - `GET /api/catalog -> list[CatalogColumn]`:
    - `CatalogColumn` fields: `id`, `title`, `kind`, `default_model`, `models`, `error`,
      `embedding_temperature`, `emails_per_request`;
    - `CatalogModel` fields: `id`, `name`, `prompt_price_per_m`, `completion_price_per_m`,
      `context_length`, `max_completion_tokens`.
  - Tests: fixture `make_app(handler, *, api_key=None) -> TestClient` (lifespan entered; closed at
    teardown) and `tests.factories.services_of(client) -> Services` (typed access to
    `app.state.services`).

- [ ] **Step 1: Add the `make_app` fixture**

Append to `tests/factories.py` (docstring + code):
```python
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

type AppFactory = Callable[..., TestClient]


def services_of(client: TestClient) -> Services:
    return cast(FastAPI, client.app).state.services
```

Append to `tests/conftest.py` (imports + fixture + docstring entry `make_app: TestClient factory with the lifespan entered`):
```python
from collections.abc import Iterator

from fastapi.testclient import TestClient

from jev_bench.web.app import create_app
from tests.factories import AppFactory


@pytest.fixture
def make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppFactory]:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    clients: list[TestClient] = []

    def build(handler: Handler, *, api_key: str | None = None) -> TestClient:
        http = httpx2.AsyncClient(base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler))
        client = TestClient(create_app(mini_settings(tmp_path, api_key), http=http))
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_web_security.py`:
```python
"""Tests for jev_bench.web.security."""

import pytest

from jev_bench.web.security import CSP
from tests.factories import AppFactory, FakeOpenRouter


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("/api/status", id="api"),
        pytest.param("/", id="static-index"),
        pytest.param("/api/missing", id="api-404"),
    ],
)
def test_security_headers_on_every_response(make_app: AppFactory, path: str) -> None:
    response = make_app(FakeOpenRouter()).get(path)
    assert response.headers["Content-Security-Policy"] == CSP
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_csp_forbids_inline_scripts_and_foreign_connections() -> None:
    directives = dict(part.split(" ", 1) for part in CSP.split("; "))
    assert directives["script-src"] == "'self' https://cdn.jsdelivr.net"
    assert directives["connect-src"] == "'self'"
    assert directives["object-src"] == "'none'"
    assert "'unsafe-inline'" not in directives["script-src"]
```

`tests/test_web_deps.py`:
```python
"""Tests for jev_bench.web.deps."""

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jev_bench.web.deps import NO_KEY_DETAIL, ApiKeyDep
from tests.factories import ServicesFactory


def _unused(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(500)


@pytest.mark.parametrize(
    ("server_key", "header", "status", "expected"),
    [
        pytest.param("sk-server", "sk-browser", 200, "sk-server", id="server-wins"),
        pytest.param(None, "sk-browser", 200, "sk-browser", id="header-used"),
        pytest.param(None, None, 400, NO_KEY_DETAIL, id="missing"),
        pytest.param(None, "   ", 400, NO_KEY_DETAIL, id="blank-header"),
    ],
)
async def test_require_api_key(make_services: ServicesFactory, server_key: str | None, header: str | None, status: int, expected: str) -> None:
    app = FastAPI()
    app.state.services = make_services(_unused, api_key=server_key)

    @app.get("/key")
    def key(api_key: ApiKeyDep) -> dict[str, str]:
        return {"key": api_key}

    headers = {"X-OpenRouter-Key": header} if header is not None else {}
    response = TestClient(app).get("/key", headers=headers)
    assert response.status_code == status
    body = response.json()
    assert (body.get("key") or body.get("detail")) == expected
```

`tests/test_web_app.py`:
```python
"""Tests for jev_bench.web.app."""

from datetime import UTC, datetime
from pathlib import Path

import httpx2
from fastapi.testclient import TestClient

from jev_bench.benchmark_config import JevParams
from jev_bench.questions import load_question_set
from jev_bench.store.runs import RunMeta, RunStore
from jev_bench.web.app import STATIC_DIR, create_app, create_default_app
from tests.factories import AppFactory, FakeOpenRouter, mini_settings


def test_static_index_is_served(make_app: AppFactory) -> None:
    response = make_app(FakeOpenRouter()).get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert (STATIC_DIR / "index.html").exists()


def test_unknown_api_route_is_404(make_app: AppFactory) -> None:
    assert make_app(FakeOpenRouter()).get("/api/does-not-exist").status_code == 404


def test_lifespan_sweeps_interrupted_runs(tmp_path: Path) -> None:
    settings = mini_settings(tmp_path)
    run = RunMeta(
        id="20260924-100000-jev-x-0001", column="jev", kind="decisions", model="m", generation_ids=("g",), mode="per_email",
        emails_per_request=1, question_set=load_question_set(settings.config_dir / "questions.toml"), params=JevParams(),
        concurrency=1, created_at=datetime(2026, 9, 24, tzinfo=UTC), n_emails=0,
    )
    RunStore(settings.data_dir / "runs").save(run)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(FakeOpenRouter()))
    with TestClient(create_app(settings, http=http)):
        assert RunStore(settings.data_dir / "runs").get(run.id).status == "interrupted"


def test_create_default_app_builds_an_app() -> None:
    assert create_default_app().title == "jev-bench"
```

`tests/test_web_routes_status.py`:
```python
"""Tests for jev_bench.web.routes.status."""

import pytest

from tests.factories import AppFactory, FakeOpenRouter


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        pytest.param("sk-server", True, id="server-key"),
        pytest.param(None, False, id="no-key"),
    ],
)
def test_status(make_app: AppFactory, api_key: str | None, expected: bool) -> None:
    assert make_app(FakeOpenRouter(), api_key=api_key).get("/api/status").json() == {"server_key": expected}
```

`tests/test_web_routes_catalog.py`:
```python
"""Tests for jev_bench.web.routes.catalog."""

import pytest

from tests.factories import AppFactory, FakeOpenRouter


def test_catalog_lists_columns_with_models_and_prices(make_app: AppFactory) -> None:
    columns = make_app(FakeOpenRouter()).get("/api/catalog").json()
    by_id = {column["id"]: column for column in columns}
    assert list(by_id) == ["jev", "anthropic", "embeddings"]
    anthropic = by_id["anthropic"]
    assert anthropic["default_model"] == "anthropic/claude-sonnet-5"
    assert [model["id"] for model in anthropic["models"]] == ["anthropic/claude-sonnet-5"]
    assert anthropic["models"][0]["prompt_price_per_m"] == pytest.approx(2.0)
    assert anthropic["models"][0]["max_completion_tokens"] == 128000
    assert anthropic["error"] is None
    assert (by_id["embeddings"]["embedding_temperature"], by_id["embeddings"]["emails_per_request"]) == (0.05, 2)
    assert by_id["jev"]["embedding_temperature"] is None


def test_catalog_outage_keeps_the_default_model(make_app: AppFactory) -> None:
    columns = make_app(FakeOpenRouter(models_status=503)).get("/api/catalog").json()
    jev = columns[0]
    assert "503" in jev["error"]
    assert [model["id"] for model in jev["models"]] == ["typesafe/jev-1.13"]
    assert jev["models"][0]["name"] == "typesafe/jev-1.13 (default)"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_app.py tests/test_web_security.py tests/test_web_deps.py tests/test_web_routes_status.py tests/test_web_routes_catalog.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.web'`.

- [ ] **Step 4: Write the implementation**

`src/jev_bench/web/__init__.py`:
```python
"""Web layer: FastAPI app, JSON API routes and the static vanilla-JS UI."""
```

`src/jev_bench/web/static/index.html` (a minimal shell that Task 23 replaces):
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark · jev-bench</title>
</head>
<body>
  <main id="app"></main>
</body>
</html>
```

`src/jev_bench/web/security.py`:
```python
"""Security headers for every response (CSP tuned for the vanilla UI and its jsDelivr assets).

Constants:
    CSP: Content-Security-Policy value.
Functions:
    security_headers: HTTP middleware adding CSP, nosniff and no-referrer headers.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response

CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
        "font-src 'self' https://cdn.jsdelivr.net https://fonts.gstatic.com",
        "img-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
    ]
)


async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response
```

`src/jev_bench/web/deps.py`:
```python
"""FastAPI dependencies: the shared Services and the API key a job should use.

Constants:
    NO_KEY_DETAIL
Types:
    ServicesDep, ApiKeyDep
Functions:
    get_services: Services stored on the app by the lifespan.
    require_api_key: server key, else the X-OpenRouter-Key header, else HTTP 400.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from jev_bench.services import Services

NO_KEY_DETAIL = "OpenRouter API key is not configured: set OPENROUTER_API_KEY on the server or enter a key in the UI."


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def require_api_key(services: ServicesDep, x_openrouter_key: Annotated[str | None, Header()] = None) -> str:
    key = services.api_key(x_openrouter_key)
    if key is None:
        raise HTTPException(status_code=400, detail=NO_KEY_DETAIL)
    return key


ApiKeyDep = Annotated[str, Depends(require_api_key)]
```

`src/jev_bench/web/routes/status.py`:
```python
"""Status route: whether the server holds an OpenRouter key.

Classes:
    StatusView
Functions:
    status: GET /status
"""

from fastapi import APIRouter
from pydantic import BaseModel

from jev_bench.web.deps import ServicesDep

router = APIRouter(tags=["status"])


class StatusView(BaseModel):
    server_key: bool


@router.get("/status")
def status(services: ServicesDep) -> StatusView:
    return StatusView(server_key=services.settings.server_api_key() is not None)
```

`src/jev_bench/web/routes/catalog.py`:
```python
"""Catalog route: every benchmark column with its selectable OpenRouter models.

Classes:
    CatalogModel, CatalogColumn
Functions:
    catalog: GET /catalog
"""

from fastapi import APIRouter
from pydantic import BaseModel

from jev_bench.benchmark_config import BenchmarkConfig, ColumnConfig, ColumnKind
from jev_bench.catalog import ModelInfo
from jev_bench.openrouter import OpenRouterError
from jev_bench.services import Services
from jev_bench.web.deps import ServicesDep

router = APIRouter(tags=["catalog"])


class CatalogModel(BaseModel):
    id: str
    name: str
    prompt_price_per_m: float
    completion_price_per_m: float
    context_length: int | None
    max_completion_tokens: int | None


class CatalogColumn(BaseModel):
    id: str
    title: str
    kind: ColumnKind
    default_model: str
    models: list[CatalogModel]
    error: str | None = None
    embedding_temperature: float | None = None
    emails_per_request: int | None = None


@router.get("/catalog")
async def catalog(services: ServicesDep) -> list[CatalogColumn]:
    config = services.benchmark_config()
    return [await _column(services, column, config) for column in config.columns]


async def _column(services: Services, column: ColumnConfig, config: BenchmarkConfig) -> CatalogColumn:
    models, error = await _models(services, column)
    embeddings = column.kind == "embeddings"
    return CatalogColumn(
        id=column.id,
        title=column.title,
        kind=column.kind,
        default_model=column.default_model,
        models=_with_default(models, column.default_model),
        error=error,
        embedding_temperature=config.embeddings.temperature if embeddings else None,
        emails_per_request=config.embeddings.emails_per_request if embeddings else None,
    )


async def _models(services: Services, column: ColumnConfig) -> tuple[list[ModelInfo], str | None]:
    try:
        return await services.catalog.for_column(column), None
    except OpenRouterError as exc:
        return [], str(exc)


def _with_default(models: list[ModelInfo], default: str) -> list[CatalogModel]:
    views = [_view(model) for model in models]
    if all(view.id != default for view in views):
        views.insert(0, _view(ModelInfo(id=default, name=f"{default} (default)")))
    return views


def _view(model: ModelInfo) -> CatalogModel:
    return CatalogModel(
        id=model.id,
        name=model.name,
        prompt_price_per_m=model.prompt_price * 1_000_000,
        completion_price_per_m=model.completion_price * 1_000_000,
        context_length=model.context_length,
        max_completion_tokens=model.max_completion_tokens,
    )
```

`src/jev_bench/web/routes/__init__.py`:
```python
"""API routers mounted under /api."""

from fastapi import APIRouter

from jev_bench.web.routes import catalog, status

ROUTERS: list[APIRouter] = [status.router, catalog.router]
```

`src/jev_bench/web/app.py`:
```python
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
    return create_app(load_settings())
```

- [ ] **Step 5: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_web_app.py tests/test_web_security.py tests/test_web_deps.py tests/test_web_routes_status.py tests/test_web_routes_catalog.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/web tests
git commit -m "feat(web): app factory with CSP, key dependency, status and catalog routes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
