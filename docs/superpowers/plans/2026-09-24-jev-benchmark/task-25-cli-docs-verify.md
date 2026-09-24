### Task 25: CLI, CLAUDE.md, integration smoke test and final verification

**Files:**
- Create: `src/jev_bench/cli.py`, `src/jev_bench/cli_jobs.py`, `CLAUDE.md`
- Test: `tests/test_cli.py`, `tests/test_cli_jobs.py`, `tests/test_integration_openrouter.py`

**Interfaces:**
- Consumes:
  - Task 1: `Settings`, `load_settings`;
  - Task 5: `build_http_client`;
  - Task 16: `Services`, `GenerationRequest`, `launch_generation`;
  - Task 17: `RunRequest`, `RunLaunchError`, `launch_run`;
  - test helpers: `mini_settings`, `FakeOpenRouter`, `seed_generation`.
- Produces:
  - `jev_bench.cli.app` (typer), registered as the `jev-bench` script:
    - `jev-bench serve [--host 127.0.0.1] [--port 8000] [--reload]`. It prints a yellow warning when the
      host is not loopback.
    - `jev-bench generate [--count 200] [--name generation] [--seed N] [--model M ...]`.
    - `jev-bench run COLUMN --generation G [--generation G2 ...] [--model M] [--mode per_email|all_in_one]`.
  - `jev_bench.cli_jobs`:
    - `await generate_and_wait(request, *, settings=None, http=None) -> int`;
    - `await run_and_wait(column, generations, model, mode, *, settings=None, http=None) -> int`;
    - exit codes: 0 completed, 1 failed/cancelled/interrupted, 2 invalid request / no key.

`cli.py` imports only `typer` at module level; every heavy import lives inside a command. Gate:
`uv run python -X importtime -c "import jev_bench.cli"` shows < 200 ms cumulative for `jev_bench.cli`, and
`time uv run jev-bench --help` < 500 ms.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
"""Tests for jev_bench.cli."""

from typing import Any

import pytest
from typer.testing import CliRunner

from jev_bench import cli_jobs
from jev_bench.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("serve", "generate", "run"):
        assert command in result.output


@pytest.mark.parametrize(
    ("host", "warned"),
    [
        pytest.param("127.0.0.1", False, id="loopback"),
        pytest.param("0.0.0.0", True, id="exposed"),
    ],
)
def test_serve_starts_uvicorn(monkeypatch: pytest.MonkeyPatch, host: str, warned: bool) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda target, **kwargs: calls.append({"target": target, **kwargs}))
    result = runner.invoke(app, ["serve", "--host", host, "--port", "8123"])
    assert result.exit_code == 0
    assert calls == [{"target": "jev_bench.web.app:create_default_app", "factory": True, "host": host, "port": 8123, "reload": False}]
    assert ("plain HTTP" in result.output) is warned


def test_generate_builds_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    async def fake(request: Any, **_: Any) -> int:
        seen.append(request)
        return 0

    monkeypatch.setattr(cli_jobs, "generate_and_wait", fake)
    result = runner.invoke(app, ["generate", "--count", "5", "--name", "t", "--seed", "3", "--model", "a/b", "--model", "c/d"])
    assert result.exit_code == 0
    assert (seen[0].count, seen[0].name, seen[0].seed, seen[0].models) == (5, "t", 3, ("a/b", "c/d"))


def test_run_passes_arguments_and_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, ...]] = []

    async def fake(column: str, generations: list[str], model: str | None, mode: str | None, **_: Any) -> int:
        seen.append((column, generations, model, mode))
        return 1

    monkeypatch.setattr(cli_jobs, "run_and_wait", fake)
    result = runner.invoke(app, ["run", "anthropic", "-g", "g1", "-g", "g2", "--mode", "all_in_one"])
    assert result.exit_code == 1
    assert seen == [("anthropic", ["g1", "g2"], None, "all_in_one")]


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["generate", "--count", "0"], id="count-too-small"),
        pytest.param(["run", "jev"], id="missing-generation"),
    ],
)
def test_invalid_arguments(args: list[str]) -> None:
    assert runner.invoke(app, args).exit_code == 2
```

`tests/test_cli_jobs.py`:
```python
"""Tests for jev_bench.cli_jobs."""

from pathlib import Path

import httpx2
import pytest

from jev_bench.cli_jobs import generate_and_wait, run_and_wait
from jev_bench.generation.launcher import GenerationRequest
from jev_bench.services import Services
from tests.factories import FakeOpenRouter, mini_settings, seed_generation


def _http(handler: FakeOpenRouter) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


async def test_generate_and_wait_completes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    http = _http(FakeOpenRouter())
    code = await generate_and_wait(GenerationRequest(count=2), settings=mini_settings(tmp_path, "sk-test"), http=http)
    assert code == 0
    assert "2/2 emails" in capsys.readouterr().err
    await http.aclose()


async def test_generate_without_key_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    http = _http(FakeOpenRouter())
    assert await generate_and_wait(GenerationRequest(count=1), settings=mini_settings(tmp_path), http=http) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    await http.aclose()


async def test_generate_launch_error_is_exit_2(tmp_path: Path) -> None:
    settings = mini_settings(tmp_path, "sk-test")
    config = settings.config_dir / "generation.toml"
    config.write_text(config.read_text(encoding="utf-8").replace('question = "category"', 'question = "missing"'), encoding="utf-8")
    http = _http(FakeOpenRouter())
    assert await generate_and_wait(GenerationRequest(count=1), settings=settings, http=http) == 2
    await http.aclose()


@pytest.mark.parametrize(
    ("column", "mode", "api_key", "expected"),
    [
        pytest.param("jev", None, "sk-test", 0, id="completed"),
        pytest.param("anthropic", "all_in_one", "sk-test", 0, id="chat-all-in-one"),
        pytest.param("jev", "batched", "sk-test", 2, id="invalid-mode"),
        pytest.param("missing", None, "sk-test", 2, id="unknown-column"),
        pytest.param("jev", None, None, 2, id="no-key"),
    ],
)
async def test_run_and_wait(tmp_path: Path, column: str, mode: str | None, api_key: str | None, expected: int) -> None:
    settings = mini_settings(tmp_path, api_key)
    http = _http(FakeOpenRouter())
    generation_id = seed_generation(Services(settings, http))
    assert await run_and_wait(column, [generation_id], None, mode, settings=settings, http=http) == expected
    await http.aclose()


async def test_failed_run_is_exit_1(tmp_path: Path) -> None:
    settings = mini_settings(tmp_path, "sk-test")

    def unauthorized(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/v1/models"):
            return FakeOpenRouter()(request)
        return httpx2.Response(401, json={"error": {"message": "no auth"}})

    http = httpx2.AsyncClient(base_url="https://openrouter.test/api", transport=httpx2.MockTransport(unauthorized))
    generation_id = seed_generation(Services(settings, http))
    assert await run_and_wait("jev", [generation_id], None, None, settings=settings, http=http) == 1
    await http.aclose()
```

`tests/test_integration_openrouter.py`:
```python
"""Opt-in smoke test against the real OpenRouter API: one hand-written email per column and mode.

Run with: uv run pytest -m integration tests/test_integration_openrouter.py (needs OPENROUTER_API_KEY).
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from jev_bench.emails import Email, Party
from jev_bench.generation.launcher import GenerationRequest, launch_generation
from jev_bench.openrouter import build_http_client
from jev_bench.run_launcher import RunRequest, launch_run
from jev_bench.services import Services
from jev_bench.settings import Settings, load_settings
from jev_bench.store.generations import GenerationMeta

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
GENERATION_ID = "20260924-000000-smoke-0001"


@pytest.fixture
async def live(tmp_path: Path) -> AsyncIterator[tuple[Services, str]]:
    key = load_settings().server_api_key()
    if not key:
        pytest.skip("OPENROUTER_API_KEY is not set")
    settings = Settings(_env_file=None, openrouter_api_key=SecretStr(key), data_dir=tmp_path / "data", config_dir=CONFIG_DIR)
    http = build_http_client(settings)
    try:
        yield Services(settings, http), key
    finally:
        await http.aclose()


def _seed(services: Services) -> None:
    meta = GenerationMeta(
        id=GENERATION_ID, name="smoke", created_at=datetime.now(UTC), status="completed", requested=1, done=1, seed=0,
        models=("manual",), question_set=services.question_set(), config=services.generation_config(),
    )
    services.generations.save(meta)
    email = Email(
        id=f"{GENERATION_ID}.0001",
        sent_at=datetime(2026, 9, 24, 8, 15, tzinfo=UTC),
        sender=Party(name="Security Team", address="security@paypa1-support.test"),
        to=(Party(name="Ann Lee", address="ann.lee@example.org"),),
        subject="Urgent: your account will be suspended in 24 hours",
        body="We detected unusual activity. Confirm your password within 24 hours at http://paypa1-support.test/verify or your account will be closed.",
        generator_model="manual",
    )
    services.generations.append_email(GENERATION_ID, email)


@pytest.mark.parametrize(
    ("column", "mode"),
    [
        pytest.param("jev", None, id="jev"),
        pytest.param("anthropic", "per_email", id="anthropic-per-email"),
        pytest.param("anthropic", "all_in_one", id="anthropic-all-in-one"),
        pytest.param("openai", "per_email", id="openai-per-email"),
        pytest.param("embeddings", None, id="embeddings"),
    ],
)
async def test_one_email_per_column(live: tuple[Services, str], column: str, mode: str | None) -> None:
    services, key = live
    _seed(services)
    request = RunRequest.model_validate({"column": column, "generation_ids": [GENERATION_ID], "mode": mode})
    meta, task = await launch_run(request, key, services)
    await asyncio.wait_for(task, timeout=240)
    final = services.runs.get(meta.id)
    assert final.status == "completed", final.error
    assert final.n_errors == 0, services.runs.predictions(meta.id)
    assert final.total_cost > 0


async def test_generation_of_one_email(live: tuple[Services, str]) -> None:
    services, key = live
    meta, task = await launch_generation(GenerationRequest(name="smoke", count=1, seed=1), key, services)
    await asyncio.wait_for(task, timeout=240)
    final = services.generations.get(meta.id)
    assert (final.status, final.done, final.errors) == ("completed", 1, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py tests/test_cli_jobs.py -v`
Expected: `ModuleNotFoundError` for `jev_bench.cli` / `jev_bench.cli_jobs`.
Run: `uv run pytest tests/test_integration_openrouter.py -v`
Expected: every test deselected by the default `-m 'not integration and not slow'` (0 selected).

- [ ] **Step 3: Write the implementation**

`src/jev_bench/cli.py`:
```python
"""Command-line entry point: `serve`, `generate`, `run`.

Heavy modules (uvicorn, FastAPI, numpy, httpx2) are imported inside the commands so `--help` stays fast.

Constants:
    LOOPBACK
Functions:
    serve: start the web UI and API.
    generate: create one generation and wait for it.
    run: run one benchmark column on generations and wait for it.
"""

from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Jev benchmark: generate emails, run columns, serve the UI.")

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind; keep loopback unless you know why.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on code changes (development).")] = False,
) -> None:
    """Serve the web UI and the JSON API."""
    import uvicorn
    from rich.console import Console

    if host not in LOOPBACK:
        Console(stderr=True).print(f"[yellow]Warning:[/] binding to {host}: API keys entered in the UI travel over plain HTTP.")
    uvicorn.run("jev_bench.web.app:create_default_app", factory=True, host=host, port=port, reload=reload)


@app.command()
def generate(
    count: Annotated[int, typer.Option(min=1, max=2000, help="Number of emails.")] = 200,
    name: Annotated[str, typer.Option(help="Generation name.")] = "generation",
    seed: Annotated[int | None, typer.Option(min=0, help="Seed; random when omitted.")] = None,
    model: Annotated[list[str] | None, typer.Option("--model", help="Generator model (repeatable); defaults to the config mix.")] = None,
) -> None:
    """Generate a new set of synthetic emails with reference answers."""
    import asyncio

    from jev_bench import cli_jobs
    from jev_bench.generation.launcher import GenerationRequest

    request = GenerationRequest(name=name, count=count, seed=seed, models=tuple(model) if model else None)
    raise typer.Exit(asyncio.run(cli_jobs.generate_and_wait(request)))


@app.command()
def run(
    column: Annotated[str, typer.Argument(help="Column id from config/benchmark.toml.")],
    generations: Annotated[list[str], typer.Option("--generation", "-g", help="Generation id (repeatable).")],
    model: Annotated[str | None, typer.Option(help="Model id; defaults to the column default.")] = None,
    mode: Annotated[str | None, typer.Option(help="per_email or all_in_one (chat columns only).")] = None,
) -> None:
    """Run one benchmark column on the given generations."""
    import asyncio

    from jev_bench import cli_jobs

    raise typer.Exit(asyncio.run(cli_jobs.run_and_wait(column, generations, model, mode)))
```

`src/jev_bench/cli_jobs.py`:
```python
"""Async bodies of the CLI commands: build services, launch a job, render live progress, report the outcome.

Constants:
    NO_KEY
Functions:
    generate_and_wait: run one generation to completion; returns the exit code.
    run_and_wait: run one benchmark column to completion; returns the exit code.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from jev_bench.generation.launcher import GenerationRequest, launch_generation
from jev_bench.openrouter import build_http_client
from jev_bench.run_launcher import RunLaunchError, RunRequest, launch_run
from jev_bench.services import Services
from jev_bench.settings import Settings, load_settings

NO_KEY = "OPENROUTER_API_KEY is not set (environment or .env)."
console = Console(stderr=True)


async def generate_and_wait(
    request: GenerationRequest, *, settings: Settings | None = None, http: httpx2.AsyncClient | None = None
) -> int:
    async with _services(settings, http) as services:
        key = services.api_key(None)
        if key is None:
            return _fail(NO_KEY)
        try:
            meta, task = await launch_generation(request, key, services)
        except ValueError as exc:
            return _fail(str(exc))
        await _watch(services, meta.id, task, f"generation {meta.name}")
        final = services.generations.get(meta.id)
        summary = f"generation {final.id}: {final.done}/{final.requested} emails, {final.errors} errors, ${final.total_cost:.4f}"
        return _report(final.status, summary, final.error)


async def run_and_wait(
    column: str,
    generations: list[str],
    model: str | None,
    mode: str | None,
    *,
    settings: Settings | None = None,
    http: httpx2.AsyncClient | None = None,
) -> int:
    try:
        request = RunRequest.model_validate({"column": column, "model": model, "generation_ids": generations, "mode": mode})
    except ValidationError as exc:
        return _fail(f"invalid run request: {exc.errors()[0]['msg']}")
    async with _services(settings, http) as services:
        key = services.api_key(None)
        if key is None:
            return _fail(NO_KEY)
        try:
            meta, task = await launch_run(request, key, services)
        except RunLaunchError as exc:
            return _fail(str(exc))
        await _watch(services, meta.id, task, f"{meta.column} · {meta.model}")
        final = services.runs.get(meta.id)
        summary = f"run {final.id}: {final.n_done}/{final.n_emails} emails, {final.n_errors} errors, ${final.total_cost:.4f}, {final.duration_s or 0:.1f} s"
        return _report(final.status, summary, final.error)


@asynccontextmanager
async def _services(settings: Settings | None, http: httpx2.AsyncClient | None) -> AsyncIterator[Services]:
    resolved = settings or load_settings()
    client = http or build_http_client(resolved)
    try:
        yield Services(resolved, client)
    finally:
        if http is None:
            await client.aclose()


async def _watch(services: Services, job_id: str, task: asyncio.Task[None], label: str) -> None:
    columns = (TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.fields[cost]}"), TimeElapsedColumn())
    with Progress(*columns, console=console, transient=True) as progress:
        bar = progress.add_task(label, total=None, cost="$0.0000")
        while not task.done():
            view = services.jobs.progress(job_id)
            if view is not None:
                progress.update(bar, total=view.total, completed=view.done, cost=f"${view.cost:.4f}")
            await asyncio.wait({task}, timeout=0.5)
    await task


def _report(status: str, summary: str, error: str | None) -> int:
    if status == "completed":
        console.print(f"[green]✓[/] {summary}")
        return 0
    detail = f" — {error}" if error else ""
    console.print(f"[red]✗ {status}:[/] {summary}{detail}")
    return 1


def _fail(message: str) -> int:
    console.print(f"[red]error:[/] {message}")
    return 2
```

- [ ] **Step 4: Run tests to verify they pass, then measure startup**

Run: `uv run pytest tests/test_cli.py tests/test_cli_jobs.py -v`
Expected: all PASS.

Run: `uv run python -X importtime -c "import jev_bench.cli" 2>&1 | sort -t'|' -k2 -n -r | head -5`
Expected: the `jev_bench.cli` cumulative figure < 200000 µs; no `fastapi`, `numpy` or `httpx2` in the list.

Run: `time uv run jev-bench --help`
Expected: < 0.5 s wall (take a warm median of 3 runs).

- [ ] **Step 5: Write `CLAUDE.md`**

````markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmark for **Jev** (`typesafe/jev-1.13`, TypeSafe's decision model on OpenRouter) against an Anthropic
chat model, an OpenAI chat model and an embedding-similarity baseline on synthetic email triage. Everything
goes through OpenRouter.

Design: `docs/superpowers/specs/2026-09-24-jev-benchmark-design.md` (§14 records revisions made while
planning). Implementation plan: `docs/superpowers/plans/2026-09-24-jev-benchmark.md`.

**All repository content is in English** (code, docs, UI copy, prompts, commits).

## Commands

```bash
uv sync                                   # install (Python >= 3.14)
uv run jev-bench serve                    # UI + API on http://127.0.0.1:8000
uv run jev-bench generate --count 200     # new generation (needs OPENROUTER_API_KEY in env/.env)
uv run jev-bench run anthropic -g <generation-id> [--mode all_in_one] [--model anthropic/claude-sonnet-5]

uv run pytest                             # default suite (no network; integration + slow excluded)
uv run pytest tests/test_compare.py -k fleiss -v    # one module / one test
uv run pytest --cov --cov-fail-under=95   # coverage gate
uv run pytest -m integration tests/test_integration_openrouter.py   # real OpenRouter (costs cents)
uv run ruff check --fix && uv run ruff format && uv run pyright     # after every change
```

## Architecture (read these together)

- **The question set is the single source of truth** (`config/questions.toml` → `questions.py`).
  - Types mirror Jev's primitives (`choice` / `score` / `noul`).
  - Every answer from every source is normalized to a distribution `{option_id: p}`; `noul` becomes
    `{yes, no}`.
  - Jev payloads (`classifiers/jev.py`), LLM JSON schemas (`classifiers/llm_schema.py`), embedding option
    texts, generator schemas, metrics and the UI are all derived from it.
- **Columns** (`config/benchmark.toml` → `benchmark_config.py`) are `decisions` / `chat` / `embeddings`.
  - Each kind has a classifier implementing the `Classifier` protocol (`classifiers/base.py`): `prepare()`
    once, then `classify(batch)` per request.
  - `run_launcher.py` validates a request and builds the classifier.
  - `runner.py` plans requests under the token budget (`tokens.py` + `request_plan.py`: minimal equal
    contiguous split), runs them under a semaphore, and appends predictions and responses.
- **Generation** (`generation/`): a seeded trait plan (`plan.py`), then prompts and strict schema
  (`prompt.py`), then the job (`generator.py`). The generator's own answers become `reference_answers`.
- **Persistence**: JSON/JSONL under `data/` (`store/`). `data/embeddings/` is a gitignored per-model vector
  cache keyed by sha256 of the exact input text.
  - Runs and generations left `running` become `interrupted` at startup (`Services.sweep_interrupted`).
- **Comparison** (`compare.py`, pure numpy via `metrics/`):
  - raters are runs, the generator reference and optional human labels;
  - pairwise agreement / κ (quadratic for score) / JSD / Pearson / Brier with bootstrap CIs, plus Fleiss'
    κ over runs;
  - a per-email disagreement index.
- **Web**: `web/app.py` (lifespan builds `Services`) exposes JSON routes under `/api` (`web/routes/`) and a
  build-free UI in `web/static/` (Bootstrap 5.3 + Chart.js from jsDelivr with SRI).
- **CLI**: `cli.py` (typer, lazy imports) delegates to `cli_jobs.py`, which uses the same launchers and
  `Services` as the web app.

## Invariants you must not break

- **Models see only** `Email.to_state()`: `{sent_at, from, to, cc, subject, body}`. They never see email
  ids (these contain generation name slugs), traits, reference answers or the generator model.
  All-in-one prompts address emails as `e001…`.
- **Generator answers never feed the benchmark.** For example, τ for embeddings is fixed in config, not
  tuned on references.
- **API keys:**
  - the server key (`OPENROUTER_API_KEY`) wins;
  - otherwise the browser sends `X-OpenRouter-Key`, and only on `POST /api/runs` and
    `POST /api/generations`;
  - keys are passed per call (never set on the shared client) and are never persisted or logged. A test
    greps `data/` and the logs for a sentinel key.
- **UI:**
  - never use `innerHTML` / `outerHTML` / `insertAdjacentHTML` / `document.write`: build DOM with `h()`
    from `js/dom.js`. Email bodies are hostile by design (prompt injections);
  - no inline scripts. `tests/test_web_static.py` enforces both, along with SRI and pinned CDN versions.
- **Config** TOMLs are re-read on every run/generation start, and runs snapshot what they used. Edit
  `config/*.toml` instead of code to change prompts, questions, traits or default models.

## Conventions

- `httpx2` (Pydantic's maintained httpx continuation, same API), never `httpx`; no OpenAI SDK.
- Every Python file starts with a docstring listing its classes and functions; no inline comments; full type
  annotations; pydantic models for known shapes; `loguru` for logs, `rich` for CLI output.
- Tests:
  - one file per module;
  - tables via `pytest.param(..., id=...)`;
  - OpenRouter is always mocked (`httpx2.MockTransport`). Use `tests.factories.FakeOpenRouter`, the
    `make_client` / `make_services` / `make_app` fixtures and the mini configs in `tests/factories.py`;
  - each test < 50 ms (global `timeout = 1`).
- Default suite target < 5 s. Keep `jev-bench --help` < 500 ms: no heavy imports at the top of `cli.py`.
````

- [ ] **Step 6: Full verification**

Run: `uv run ruff check && uv run ruff format --check && uv run pyright`
Expected: no issues, `0 errors`.

Run: `uv run pytest --cov --cov-fail-under=95 --durations=10`
Expected: all pass; coverage ≥ 95 %; the slowest test < 50 ms (any test above 20 ms goes into
`tests/slow_tests.txt` via `~/.claude/scripts/test-profile.sh`, or gets fixed).

Run the browser smoke test (the controller does this with the Playwright MCP tools):
1. `uv run jev-bench serve --port 8765` in the background.
2. Open `http://127.0.0.1:8765/`, `/generations.html`, `/explorer.html` and `/runs.html`. The console must
   show no errors and no CSP violations. The navbar key badge shows "set API key" when the server has no
   key.
3. With a key available: create a 5-email generation on the Generations page and wait for `completed`.
   Then on the Benchmark page select it and run Jev, Anthropic (per email and all in one), OpenAI and
   Embeddings. Check the live timer, cost and progress, the comparison cards and charts, the Explorer
   detail panel and saving a human label.
4. Resize to 375 px width: no horizontal page scroll; the navbar labels collapse to icons.

Optionally, with a key: `uv run pytest -m integration tests/test_integration_openrouter.py -v` (a few cents).

- [ ] **Step 7: Commit**

```bash
git add src/jev_bench/cli.py src/jev_bench/cli_jobs.py CLAUDE.md tests
git commit -m "feat(cli): serve/generate/run commands, CLAUDE.md and live smoke test

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
