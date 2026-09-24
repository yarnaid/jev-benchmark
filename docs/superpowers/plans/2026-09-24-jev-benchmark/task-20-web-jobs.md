### Task 20: Generation and run routes

**Files:**
- Create: `src/jev_bench/web/routes/generations.py`, `src/jev_bench/web/routes/runs.py`
- Modify: `src/jev_bench/web/deps.py` (add `split_ids`), `src/jev_bench/web/routes/__init__.py`
  (register routers), `tests/factories.py` (add `poll`)
- Test: `tests/test_web_routes_generations.py`, `tests/test_web_routes_runs.py`; extend `tests/test_web_deps.py`

**Interfaces:**
- Consumes: `GenerationRequest`, `launch_generation` (Task 16); `RunRequest`, `RunLaunchError`,
  `launch_run` (Task 17); `ProgressView` (Task 13); `GenerationMeta`, `RunMeta` (Task 11);
  `ServicesDep`, `ApiKeyDep` (Task 19).
- Produces:
  - `jev_bench.web.deps.split_ids(value: str | None) -> list[str]`: comma-separated, trimmed,
    de-duplicated, order kept.
  - Generations:
    - `GenerationView(meta: GenerationMeta, progress: ProgressView | None)`; responses exclude
      `meta.question_set` and `meta.config`;
    - `GET /api/generations -> list[GenerationView]`;
    - `POST /api/generations` (body `GenerationRequest`, needs a key) → 202 `GenerationView`, or 400 on
      a launch `ValueError`;
    - `GET /api/generations/{id} -> GenerationView` (404 if unknown or unsafe);
    - `POST /api/generations/{id}/cancel -> {"cancelled": bool}` (404 if unknown).
  - Runs:
    - `RunView(meta: RunMeta, progress: ProgressView | None)`; responses exclude `meta.question_set`
      and `meta.params`;
    - `GET /api/runs?generations=a,b -> list[RunView]` (the filter is an exact set match);
    - `POST /api/runs` (body `RunRequest`, needs a key) → 202 `RunView`, or 400 on `RunLaunchError`;
    - `GET /api/runs/{id}` and `POST /api/runs/{id}/cancel`, as for generations.
  - Tests: `tests.factories.poll(client, path, *, until, attempts=100, delay_s=0.002) -> dict` (returns
    the first JSON body satisfying `until`; fails the test after `attempts`).

This task pins the key-never-persisted guarantee (spec §3). A sentinel key sent as `X-OpenRouter-Key`
reaches OpenRouter as `Authorization: Bearer <key>`, but it never appears in any file under the data
directory or in any captured log line.

- [ ] **Step 1: Add `split_ids` and `poll`**

Append to `src/jev_bench/web/deps.py` and list it in the docstring:
```python
def split_ids(value: str | None) -> list[str]:
    parts = (part.strip() for part in (value or "").split(","))
    return list(dict.fromkeys(part for part in parts if part))
```

Append to `tests/factories.py` (docstring + imports `import time`):
```python
def poll(client: TestClient, path: str, *, until: Callable[[dict[str, Any]], bool], attempts: int = 100, delay_s: float = 0.002) -> dict[str, Any]:
    for _ in range(attempts):
        body = client.get(path).json()
        if until(body):
            return body
        time.sleep(delay_s)
    raise AssertionError(f"{path} never satisfied the condition")
```

Append to `tests/test_web_deps.py`:
```python
from jev_bench.web.deps import split_ids


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, [], id="none"),
        pytest.param("", [], id="empty"),
        pytest.param(" a, b ,,a ", ["a", "b"], id="trim-dedupe"),
    ],
)
def test_split_ids(value: str | None, expected: list[str]) -> None:
    assert split_ids(value) == expected
```

- [ ] **Step 2: Write the failing tests**

`tests/test_web_routes_generations.py`:
```python
"""Tests for jev_bench.web.routes.generations."""

from pathlib import Path

import pytest

from jev_bench.web.deps import NO_KEY_DETAIL
from tests.factories import AppFactory, FakeOpenRouter, poll

_KEY = {"X-OpenRouter-Key": "sk-browser"}


def _done(body: dict[str, object]) -> bool:
    meta = body["meta"]
    return isinstance(meta, dict) and meta["status"] != "running"


def test_create_generation_runs_in_the_background(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    response = client.post("/api/generations", json={"name": "Spam", "count": 2, "seed": 3}, headers=_KEY)
    assert response.status_code == 202
    created = response.json()
    assert created["meta"]["status"] == "running"
    assert "config" not in created["meta"]
    assert "question_set" not in created["meta"]
    final = poll(client, f"/api/generations/{created['meta']['id']}", until=_done)
    assert (final["meta"]["status"], final["meta"]["done"], final["meta"]["seed"]) == ("completed", 2, 3)
    listed = client.get("/api/generations").json()
    assert [view["meta"]["id"] for view in listed] == [created["meta"]["id"]]
    assert "config" not in listed[0]["meta"]


def test_create_generation_without_key_is_rejected(make_app: AppFactory) -> None:
    response = make_app(FakeOpenRouter()).post("/api/generations", json={"count": 1})
    assert (response.status_code, response.json()["detail"]) == (400, NO_KEY_DETAIL)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"count": 0}, id="zero"),
        pytest.param({"count": 1, "bogus": 1}, id="extra-field"),
    ],
)
def test_invalid_generation_payload_is_422(make_app: AppFactory, payload: dict[str, object]) -> None:
    assert make_app(FakeOpenRouter()).post("/api/generations", json=payload, headers=_KEY).status_code == 422


def test_launch_value_error_is_400(make_app: AppFactory, tmp_path: Path) -> None:
    client = make_app(FakeOpenRouter())
    config = tmp_path / "config" / "generation.toml"
    config.write_text(config.read_text(encoding="utf-8").replace('question = "category"', 'question = "missing"'), encoding="utf-8")
    response = client.post("/api/generations", json={"count": 1}, headers=_KEY)
    assert response.status_code == 400
    assert "unknown question" in response.json()["detail"]


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("/api/generations/20260924-100000-missing-0001", id="unknown"),
        pytest.param("/api/generations/..%2F..%2Fetc", id="traversal"),
    ],
)
def test_unknown_generation_is_404(make_app: AppFactory, path: str) -> None:
    client = make_app(FakeOpenRouter())
    assert client.get(path).status_code == 404
    assert client.post(f"{path}/cancel").status_code == 404


def test_cancel_finished_generation_reports_false(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    created = client.post("/api/generations", json={"count": 1}, headers=_KEY).json()
    poll(client, f"/api/generations/{created['meta']['id']}", until=_done)
    assert client.post(f"/api/generations/{created['meta']['id']}/cancel").json() == {"cancelled": False}
```

`tests/test_web_routes_runs.py`:
```python
"""Tests for jev_bench.web.routes.runs."""

from pathlib import Path

import pytest
from loguru import logger

from tests.factories import AppFactory, FakeOpenRouter, poll, seed_generation, services_of

_SENTINEL = "sk-or-v1-SENTINEL-4242"


def _done(body: dict[str, object]) -> bool:
    meta = body["meta"]
    return isinstance(meta, dict) and meta["status"] != "running"


def test_create_run_and_poll_to_completion(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    generation_id = seed_generation(services_of(client))
    response = client.post("/api/runs", json={"column": "anthropic", "generation_ids": [generation_id], "mode": "all_in_one"})
    assert response.status_code == 202
    run_id = response.json()["meta"]["id"]
    final = poll(client, f"/api/runs/{run_id}", until=_done)
    assert (final["meta"]["status"], final["meta"]["n_done"], final["meta"]["mode"]) == ("completed", 2, "all_in_one")
    assert "question_set" not in final["meta"]
    assert "params" not in final["meta"]
    assert final["progress"] is None


def test_list_runs_filters_by_exact_generation_set(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    services = services_of(client)
    first = seed_generation(services)
    second = seed_generation(services, generation_id="20260924-110000-seed-bcde")
    one = client.post("/api/runs", json={"column": "jev", "generation_ids": [first]}).json()["meta"]["id"]
    both = client.post("/api/runs", json={"column": "jev", "generation_ids": [first, second]}).json()["meta"]["id"]
    poll(client, f"/api/runs/{both}", until=_done)
    poll(client, f"/api/runs/{one}", until=_done)
    assert [view["meta"]["id"] for view in client.get("/api/runs", params={"generations": first}).json()] == [one]
    assert [view["meta"]["id"] for view in client.get("/api/runs", params={"generations": f"{second},{first}"}).json()] == [both]
    assert len(client.get("/api/runs").json()) == 2


@pytest.mark.parametrize(
    ("payload", "headers", "status", "detail"),
    [
        pytest.param({"column": "jev"}, {"X-OpenRouter-Key": "k"}, 422, None, id="no-generations-field"),
        pytest.param({"column": "missing", "generation_ids": ["20260924-100000-seed-abcd"]}, {"X-OpenRouter-Key": "k"}, 400, "unknown column", id="unknown-column"),
        pytest.param({"column": "jev", "generation_ids": ["20260924-100000-seed-abcd"], "mode": "all_in_one"}, {"X-OpenRouter-Key": "k"}, 400, "only accepted for chat", id="mode-on-jev"),
        pytest.param({"column": "jev", "generation_ids": ["20260924-100000-seed-abcd"]}, {}, 400, "API key", id="no-key"),
    ],
)
def test_invalid_run_requests(make_app: AppFactory, payload: dict[str, object], headers: dict[str, str], status: int, detail: str | None) -> None:
    client = make_app(FakeOpenRouter())
    seed_generation(services_of(client))
    response = client.post("/api/runs", json=payload, headers=headers)
    assert response.status_code == status
    if detail is not None:
        assert detail in response.json()["detail"]


def test_unknown_run_is_404(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    assert client.get("/api/runs/20260924-100000-missing-0001").status_code == 404
    assert client.post("/api/runs/20260924-100000-missing-0001/cancel").status_code == 404


def test_browser_key_is_used_but_never_persisted_or_logged(make_app: AppFactory, tmp_path: Path) -> None:
    upstream = FakeOpenRouter()
    client = make_app(upstream)
    generation_id = seed_generation(services_of(client))
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        created = client.post("/api/runs", json={"column": "jev", "generation_ids": [generation_id]}, headers={"X-OpenRouter-Key": _SENTINEL})
        final = poll(client, f"/api/runs/{created.json()['meta']['id']}", until=_done)
    finally:
        logger.remove(handler)
    assert final["meta"]["status"] == "completed"
    decisions = [request for request in upstream.requests if request.url.path.endswith("/alpha/decisions")]
    assert decisions
    assert all(request.headers["Authorization"] == f"Bearer {_SENTINEL}" for request in decisions)
    stored = [path.read_text(encoding="utf-8") for path in (tmp_path / "data").rglob("*") if path.is_file()]
    assert stored
    assert all(_SENTINEL not in text for text in stored)
    assert all(_SENTINEL not in message for message in messages)


def test_cancel_running_run(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    generation_id = seed_generation(services_of(client), emails=1)
    run_id = client.post("/api/runs", json={"column": "jev", "generation_ids": [generation_id]}).json()["meta"]["id"]
    result = client.post(f"/api/runs/{run_id}/cancel").json()
    final = poll(client, f"/api/runs/{run_id}", until=_done)
    assert final["meta"]["status"] in ({"cancelled"} if result["cancelled"] else {"completed"})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_routes_generations.py tests/test_web_routes_runs.py tests/test_web_deps.py -v`
Expected: the new route tests FAIL with 404 (routes not registered); `test_split_ids` fails with `ImportError`.

- [ ] **Step 4: Write the implementation**

`src/jev_bench/web/routes/generations.py`:
```python
"""Generation routes: list, create (background job), read with live progress, cancel.

Constants:
    LIGHT: response fields excluded from generation views.
Classes:
    GenerationView: meta + live progress.
    CancelView: whether a cancel request reached a running job.
Functions:
    list_generations, create_generation, get_generation, cancel_generation: route handlers.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jev_bench.generation.launcher import GenerationRequest, launch_generation
from jev_bench.jobs import ProgressView
from jev_bench.services import Services
from jev_bench.store.generations import GenerationMeta
from jev_bench.web.deps import ApiKeyDep, ServicesDep

router = APIRouter(tags=["generations"])
LIGHT = {"meta": {"question_set", "config"}}


class GenerationView(BaseModel):
    meta: GenerationMeta
    progress: ProgressView | None


class CancelView(BaseModel):
    cancelled: bool


def _view(services: Services, meta: GenerationMeta) -> GenerationView:
    return GenerationView(meta=meta, progress=services.jobs.progress(meta.id))


def _meta_or_404(services: Services, generation_id: str) -> GenerationMeta:
    try:
        return services.generations.get(generation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown generation {generation_id!r}") from exc


@router.get("/generations", response_model_exclude={"__all__": LIGHT})
def list_generations(services: ServicesDep) -> list[GenerationView]:
    return [_view(services, meta) for meta in services.generations.list_metas()]


@router.post("/generations", status_code=202, response_model_exclude=LIGHT)
async def create_generation(request: GenerationRequest, services: ServicesDep, api_key: ApiKeyDep) -> GenerationView:
    try:
        meta, _ = await launch_generation(request, api_key, services)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _view(services, meta)


@router.get("/generations/{generation_id}", response_model_exclude=LIGHT)
def get_generation(generation_id: str, services: ServicesDep) -> GenerationView:
    return _view(services, _meta_or_404(services, generation_id))


@router.post("/generations/{generation_id}/cancel")
def cancel_generation(generation_id: str, services: ServicesDep) -> CancelView:
    _meta_or_404(services, generation_id)
    return CancelView(cancelled=services.jobs.cancel(generation_id))
```

`src/jev_bench/web/routes/runs.py`:
```python
"""Run routes: list (optionally by exact generation set), create (background job), read, cancel.

Constants:
    LIGHT: response fields excluded from run views.
Classes:
    RunView: meta + live progress.
Functions:
    list_runs, create_run, get_run, cancel_run: route handlers.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jev_bench.jobs import ProgressView
from jev_bench.run_launcher import RunLaunchError, RunRequest, launch_run
from jev_bench.services import Services
from jev_bench.store.runs import RunMeta
from jev_bench.web.deps import ApiKeyDep, ServicesDep, split_ids
from jev_bench.web.routes.generations import CancelView

router = APIRouter(tags=["runs"])
LIGHT = {"meta": {"question_set", "params"}}


class RunView(BaseModel):
    meta: RunMeta
    progress: ProgressView | None


def _view(services: Services, meta: RunMeta) -> RunView:
    return RunView(meta=meta, progress=services.jobs.progress(meta.id))


def _meta_or_404(services: Services, run_id: str) -> RunMeta:
    try:
        return services.runs.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}") from exc


@router.get("/runs", response_model_exclude={"__all__": LIGHT})
def list_runs(services: ServicesDep, generations: str | None = None) -> list[RunView]:
    metas = services.runs.list_metas()
    wanted = set(split_ids(generations))
    if wanted:
        metas = [meta for meta in metas if set(meta.generation_ids) == wanted]
    return [_view(services, meta) for meta in metas]


@router.post("/runs", status_code=202, response_model_exclude=LIGHT)
async def create_run(request: RunRequest, services: ServicesDep, api_key: ApiKeyDep) -> RunView:
    try:
        meta, _ = await launch_run(request, api_key, services)
    except RunLaunchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _view(services, meta)


@router.get("/runs/{run_id}", response_model_exclude=LIGHT)
def get_run(run_id: str, services: ServicesDep) -> RunView:
    return _view(services, _meta_or_404(services, run_id))


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, services: ServicesDep) -> CancelView:
    _meta_or_404(services, run_id)
    return CancelView(cancelled=services.jobs.cancel(run_id))
```

Replace `src/jev_bench/web/routes/__init__.py`:
```python
"""API routers mounted under /api."""

from fastapi import APIRouter

from jev_bench.web.routes import catalog, generations, runs, status

ROUTERS: list[APIRouter] = [status.router, catalog.router, generations.router, runs.router]
```

- [ ] **Step 5: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_web_routes_generations.py tests/test_web_routes_runs.py tests/test_web_deps.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/web tests
git commit -m "feat(web): generation and run routes with live progress and cancel

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
