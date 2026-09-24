### Task 16: Generator job, generation launcher and the Services container

**Files:**
- Create: `src/jev_bench/generation/generator.py`, `src/jev_bench/generation/launcher.py`, `src/jev_bench/services.py`
- Modify: `tests/factories.py` (mini TOML configs, `write_mini_config`, `generator_output`),
  `tests/conftest.py` (`make_services` fixture)
- Test: `tests/test_generation_generator.py`, `tests/test_generation_launcher.py`, `tests/test_services.py`

**Interfaces:**
- Consumes:
  - Task 1: `Settings`, `new_id`;
  - Task 2: `Email`, `email_id`, `QuestionSet`, `load_question_set`;
  - Task 5: `OpenRouterClient`, `OpenRouterError`, `ApiResponse`, `CHAT_PATH`, `chat_content`,
    `json_schema_format`;
  - Task 6: `Catalog`, `load_benchmark_config`, `BenchmarkConfig`;
  - Task 11: `GenerationConfig`, `load_generation_config`, `GenerationMeta`, `GenerationStore`,
    `RunStore`, `LabelStore`;
  - Task 12: `EmbeddingCaches`;
  - Task 13: `JobRegistry`, `JobProgress`, `cancel_status`, `describe_error`;
  - Task 14: `mark_interrupted_runs`;
  - Task 15: `build_plan`, `resolve_traits`, `render_prompts`, `generation_schema`,
    `parse_generator_output`, `count_mismatches`.
- Produces:
  - `jev_bench.generation.generator`:
    - `CHECKPOINT_EVERY = 10`;
    - `GeneratorDeps(client, api_key, store, questions, config)`;
    - `await execute_generation(meta, plan, deps, progress)`;
    - `mark_interrupted_generations(store, is_live) -> list[str]`.
  - `jev_bench.generation.launcher`:
    - `GenerationRequest(name="generation", count, seed=None, models=None)`, with `count` in [1, 2000];
    - `await launch_generation(request, api_key, services, *, now=None) -> tuple[GenerationMeta,
      asyncio.Task[None]]`.
  - `jev_bench.services.Services(settings, http)`:
    - attributes `settings`, `client`, `catalog`, `jobs`, `generations`, `runs`, `labels`,
      `embedding_caches`;
    - methods `question_set()`, `benchmark_config()`, `generation_config()` (re-read on every call),
      `api_key(supplied) -> str | None` and `sweep_interrupted() -> list[str]`.
  - Tests:
    - `tests.factories.MINI_QUESTIONS_TOML`, `MINI_BENCHMARK_TOML`, `MINI_GENERATION_TOML`;
    - `write_mini_config(config_dir)`;
    - `mini_settings(root, api_key=None) -> Settings` (writes the mini config under `root/config`,
      with data under `root/data`, zero retries);
    - `generator_output(category="spam") -> str`;
    - fixture `make_services(handler, *, api_key=None) -> Services` (tmp data/config dirs; clears
      `OPENROUTER_API_KEY` from the environment).

- [ ] **Step 1: Add mini configs and the `make_services` fixture**

Append to `tests/factories.py` and list the new names in its docstring:
```python
from pathlib import Path

from pydantic import SecretStr

from jev_bench.services import Services
from jev_bench.settings import Settings

type ServicesFactory = Callable[..., Services]

MINI_QUESTIONS_TOML = """
name = "mini"

[[questions]]
id = "category"
type = "choice"
instructions = "What kind of email?"

[questions.options]
spam = "Junk"
personal = "From a friend"
work = "From a colleague"

[[questions]]
id = "urgency"
type = "score"
instructions = "How urgent?"

[questions.options]
low = "Whenever"
today = "Within a day"
now = "Immediately"

[[questions]]
id = "needs_reply"
type = "noul"
instructions = "Needs a reply?"

[questions.options]
yes = "Reply expected"
no = "No reply expected"
"""

MINI_BENCHMARK_TOML = """
[jev]
concurrency = 2

[llm]
system_prompt = "Classify.\\n$questions"
system_prompt_all_in_one = "Batch.\\n$questions"
concurrency = 2

[embeddings]
email_template = "$subject"
option_template = "$option"
emails_per_request = 2
concurrency = 2

[[columns]]
id = "jev"
title = "Jev"
kind = "decisions"
modality = "decisions"
default_model = "typesafe/jev-1.13"

[[columns]]
id = "anthropic"
title = "Anthropic"
kind = "chat"
modality = "text"
prefix = "anthropic/"
default_model = "anthropic/claude-sonnet-5"
cache_system_prompt = true

[[columns]]
id = "embeddings"
title = "Embeddings"
kind = "embeddings"
modality = "embeddings"
default_model = "openai/text-embedding-3-large"
"""

MINI_GENERATION_TOML = """
models = ["gen/a", "gen/b"]
concurrency = 2
system_prompt = "Write.\\n$questions"
user_prompt = "Category $category ($category_prompt) at $sent_at."

[[traits]]
name = "category"
question = "category"
stratify = true
"""


def write_mini_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "questions.toml").write_text(MINI_QUESTIONS_TOML, encoding="utf-8")
    (config_dir / "benchmark.toml").write_text(MINI_BENCHMARK_TOML, encoding="utf-8")
    (config_dir / "generation.toml").write_text(MINI_GENERATION_TOML, encoding="utf-8")


def mini_settings(root: Path, api_key: str | None = None) -> Settings:
    write_mini_config(root / "config")
    return Settings(
        _env_file=None,
        data_dir=root / "data",
        config_dir=root / "config",
        openrouter_api_key=SecretStr(api_key) if api_key else None,
        max_retries=0,
        retry_base_delay_s=0.0,
    )


def generator_output(category: str = "spam") -> str:
    return json.dumps(
        {
            "email": {
                "sender": {"name": "Deals Team", "address": "deals@promo.test"},
                "to": [{"name": "Ann", "address": "ann@mail.test"}],
                "cc": [],
                "subject": "You won a prize",
                "body": "Claim it now.",
            },
            "answers": {"category": category, "urgency": "now", "needs_reply": "no"},
        }
    )
```

Append to `tests/conftest.py`: add the imports at the top, the fixture at the end, and
`make_services: async Services factory on tmp dirs` to the docstring.
```python
from jev_bench.services import Services
from tests.factories import ServicesFactory, mini_settings


@pytest.fixture
async def make_services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ServicesFactory]:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    opened: list[httpx2.AsyncClient] = []

    def build(handler: Handler, *, api_key: str | None = None) -> Services:
        http = httpx2.AsyncClient(base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler))
        opened.append(http)
        return Services(mini_settings(tmp_path, api_key), http)

    yield build
    for http in opened:
        await http.aclose()
```

- [ ] **Step 2: Write the failing tests**

`tests/test_generation_generator.py`:
```python
"""Tests for jev_bench.generation.generator."""

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest

from jev_bench.generation import generator
from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.generator import GeneratorDeps, execute_generation, mark_interrupted_generations
from jev_bench.generation.plan import PlanItem, build_plan
from jev_bench.jobs import JobProgress, JobRegistry
from jev_bench.openrouter import OpenRouterClient
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta, GenerationStore
from tests.factories import ClientFactory, EmailFactory, chat_body, generator_output

_GEN_ID = "20260924-100000-gen-abcd"
_CONFIG = GenerationConfig.model_validate(
    {
        "models": ["gen/a", "gen/b"],
        "concurrency": 2,
        "system_prompt": "Write.\n$questions",
        "user_prompt": "Category $category ($category_prompt).",
        "traits": [{"name": "category", "question": "category", "stratify": True}],
    }
)


class GeneratorServer:
    def __init__(self, failures: dict[int, httpx2.Response] | None = None, content: str | None = None) -> None:
        self.failures = failures or {}
        self.content = content
        self.bodies: list[dict[str, object]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        call = len(self.bodies)
        if call in self.failures:
            return self.failures[call]
        return httpx2.Response(200, json=chat_body(self.content or generator_output("spam"), model="gen/resolved"))


def _meta(questions: QuestionSet, requested: int = 3) -> GenerationMeta:
    return GenerationMeta(
        id=_GEN_ID,
        name="gen",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        requested=requested,
        seed=5,
        models=_CONFIG.models,
        question_set=questions,
        config=_CONFIG,
    )


def _plan(questions: QuestionSet, count: int = 3) -> list[PlanItem]:
    return build_plan(_CONFIG, questions, count=count, seed=5, models=_CONFIG.models, now=datetime(2026, 9, 24, tzinfo=UTC))


async def _generate(tmp_path: Path, client: OpenRouterClient, questions: QuestionSet, plan: list[PlanItem]) -> tuple[GenerationMeta, GenerationStore]:
    store = GenerationStore(tmp_path)
    meta = _meta(questions, len(plan))
    store.save(meta)
    deps = GeneratorDeps(client=client, api_key="sk-test", store=store, questions=questions, config=_CONFIG)
    await execute_generation(meta, plan, deps, JobProgress(len(plan), lambda: 0.0))
    return store.get(meta.id), store


async def test_generation_persists_emails_and_totals(tmp_path: Path, make_client: ClientFactory, questions: QuestionSet) -> None:
    server = GeneratorServer()
    plan = _plan(questions)
    final, store = await _generate(tmp_path, make_client(server), questions, plan)
    emails = store.emails(_GEN_ID)
    assert final.status == "completed"
    assert (final.done, final.errors) == (3, 0)
    assert final.total_cost == pytest.approx(0.003)
    assert final.trait_mismatches == sum(item.traits["category"] != "spam" for item in plan)
    assert sorted(email.id for email in emails) == [f"{_GEN_ID}.0001", f"{_GEN_ID}.0002", f"{_GEN_ID}.0003"]
    by_id = {email.id: email for email in emails}
    first = by_id[f"{_GEN_ID}.0001"]
    assert first.sent_at == plan[0].sent_at
    assert first.traits == plan[0].traits
    assert first.reference_answers == {"category": "spam", "urgency": "now", "needs_reply": "no"}
    assert first.generator_model == "gen/resolved"
    assert [body["model"] for body in server.bodies] == ["gen/a", "gen/b", "gen/a"]
    assert server.bodies[0]["response_format"]["json_schema"]["name"] == "email_generation"


@pytest.mark.parametrize(
    ("server", "errors"),
    [
        pytest.param(GeneratorServer(failures={2: httpx2.Response(500, json={"error": {"message": "boom"}})}), 1, id="http-500"),
        pytest.param(GeneratorServer(content='{"email": {}}'), 3, id="invalid-output"),
        pytest.param(GeneratorServer(failures={1: httpx2.Response(200, json=chat_body("{}", finish_reason="length"))}), 1, id="truncated"),
    ],
)
async def test_item_failures_do_not_stop_the_generation(
    tmp_path: Path, make_client: ClientFactory, questions: QuestionSet, server: GeneratorServer, errors: int
) -> None:
    final, store = await _generate(tmp_path, make_client(server), questions, _plan(questions))
    assert final.status == "completed"
    assert final.errors == errors
    assert len(store.emails(_GEN_ID)) == 3 - errors


async def test_fatal_error_fails_the_generation(tmp_path: Path, make_client: ClientFactory, questions: QuestionSet) -> None:
    server = GeneratorServer(failures={1: httpx2.Response(401, json={"error": {"message": "no auth"}})})
    final, _ = await _generate(tmp_path, make_client(server), questions, _plan(questions, 1))
    assert final.status == "failed"
    assert final.error is not None
    assert "401" in final.error


async def test_checkpoints_are_written(tmp_path: Path, make_client: ClientFactory, questions: QuestionSet, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator, "CHECKPOINT_EVERY", 1)
    final, _ = await _generate(tmp_path, make_client(GeneratorServer()), questions, _plan(questions, 2))
    assert (final.status, final.done) == ("completed", 2)


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        pytest.param("cancel", "cancelled", id="cancel"),
        pytest.param("shutdown", "interrupted", id="shutdown"),
    ],
)
async def test_cancellation_is_persisted(tmp_path: Path, make_client: ClientFactory, questions: QuestionSet, stop: str, expected: str) -> None:
    async def hang(request: httpx2.Request) -> httpx2.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    store = GenerationStore(tmp_path)
    meta = _meta(questions, 1)
    store.save(meta)
    deps = GeneratorDeps(client=make_client(hang), api_key="k", store=store, questions=questions, config=_CONFIG)
    registry = JobRegistry()
    task = registry.start(meta.id, 1, lambda progress: execute_generation(meta, _plan(questions, 1), deps, progress))
    for _ in range(5):
        await asyncio.sleep(0)
    if stop == "cancel":
        registry.cancel(meta.id)
    else:
        await registry.shutdown(timeout_s=0.05)
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert store.get(meta.id).status == expected


def test_mark_interrupted_generations(tmp_path: Path, questions: QuestionSet) -> None:
    store = GenerationStore(tmp_path)
    orphan = _meta(questions)
    live = orphan.model_copy(update={"id": "20260924-110000-live-abcd"})
    store.save(orphan)
    store.save(live)
    store.append_email(orphan.id, EmailFactory(id=f"{orphan.id}.0001"))
    assert mark_interrupted_generations(store, lambda generation_id: generation_id == live.id) == [orphan.id]
    assert (store.get(orphan.id).status, store.get(orphan.id).done) == ("interrupted", 1)
    assert store.get(live.id).status == "running"
```

`tests/test_generation_launcher.py`:
```python
"""Tests for jev_bench.generation.launcher."""

from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from jev_bench.generation.launcher import GenerationRequest, launch_generation
from tests.factories import ServicesFactory, chat_body, generator_output


def _server(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=chat_body(generator_output()))


async def test_launch_generation_runs_to_completion(make_services: ServicesFactory) -> None:
    services = make_services(_server)
    now = datetime(2026, 9, 24, 15, 30, 12, tzinfo=UTC)
    meta, task = await launch_generation(GenerationRequest(name="Spam test", count=4, seed=9), "sk-test", services, now=now)
    await task
    final = services.generations.get(meta.id)
    assert meta.id.startswith("20260924-153012-spam-test-")
    assert (final.status, final.done, final.seed, final.models) == ("completed", 4, 9, ("gen/a", "gen/b"))
    assert len(services.generations.emails(meta.id)) == 4
    assert final.question_set.name == "mini"


async def test_request_models_override_config_and_seed_is_recorded(make_services: ServicesFactory) -> None:
    services = make_services(_server)
    meta, task = await launch_generation(GenerationRequest(count=1, models=("custom/model",)), "sk-test", services)
    await task
    assert meta.models == ("custom/model",)
    assert 0 <= meta.seed < 2**31


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"count": 0}, id="zero"),
        pytest.param({"count": 2001}, id="too-many"),
        pytest.param({"count": 1, "unknown": True}, id="extra-field"),
        pytest.param({"count": 1, "name": ""}, id="empty-name"),
        pytest.param({"count": 1, "seed": -1}, id="negative-seed"),
    ],
)
def test_invalid_requests(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate(payload)
```

`tests/test_services.py`:
```python
"""Tests for jev_bench.services."""

from datetime import UTC, datetime

import httpx2
import pytest

from jev_bench.benchmark_config import JevParams
from jev_bench.store.generations import GenerationMeta
from jev_bench.store.runs import RunMeta
from tests.factories import ServicesFactory


def _unused(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(500)


@pytest.mark.parametrize(
    ("server_key", "supplied", "expected"),
    [
        pytest.param("sk-server", "sk-browser", "sk-server", id="server-wins"),
        pytest.param(None, "  sk-browser  ", "sk-browser", id="browser-stripped"),
        pytest.param(None, None, None, id="none"),
        pytest.param(None, "   ", None, id="blank"),
    ],
)
async def test_api_key_resolution(make_services: ServicesFactory, server_key: str | None, supplied: str | None, expected: str | None) -> None:
    assert make_services(_unused, api_key=server_key).api_key(supplied) == expected


async def test_configs_are_read_from_the_config_dir(make_services: ServicesFactory) -> None:
    services = make_services(_unused)
    assert services.question_set().name == "mini"
    assert [column.id for column in services.benchmark_config().columns] == ["jev", "anthropic", "embeddings"]
    assert services.generation_config().models == ("gen/a", "gen/b")


async def test_sweep_interrupted_marks_orphans(make_services: ServicesFactory) -> None:
    services = make_services(_unused)
    questions = services.question_set()
    created = datetime(2026, 9, 24, tzinfo=UTC)
    run = RunMeta(
        id="20260924-100000-jev-x-0001", column="jev", kind="decisions", model="m", generation_ids=("g",), mode="per_email",
        emails_per_request=1, question_set=questions, params=JevParams(), concurrency=1, created_at=created, n_emails=0,
    )
    generation = GenerationMeta(
        id="20260924-100000-gen-0001", name="g", created_at=created, requested=1, seed=1, models=("m",),
        question_set=questions, config=services.generation_config(),
    )
    services.runs.save(run)
    services.generations.save(generation)
    assert sorted(services.sweep_interrupted()) == sorted([run.id, generation.id])
    assert services.runs.get(run.id).status == "interrupted"
    assert services.generations.get(generation.id).status == "interrupted"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_generation_generator.py tests/test_generation_launcher.py tests/test_services.py -v`
Expected: `ModuleNotFoundError` for `jev_bench.generation.generator` / `jev_bench.services`.

- [ ] **Step 4: Write the implementation**

`src/jev_bench/generation/generator.py`:
```python
"""Executes one generation: prompts per plan item, calls the generator mix, persists emails and totals.

Constants:
    CHECKPOINT_EVERY: persist running totals after this many completed emails.
Classes:
    GeneratorDeps: collaborators of a generation job.
Functions:
    execute_generation: job body (always finalizes the meta; re-raises only CancelledError).
    mark_interrupted_generations: startup sweep for generations left `running`.
"""

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, NamedTuple

from loguru import logger

from jev_bench.emails import Email, email_id
from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.plan import PlanItem, ResolvedTrait, resolve_traits
from jev_bench.generation.prompt import count_mismatches, generation_schema, parse_generator_output, render_prompts
from jev_bench.jobs import JobProgress, cancel_status, describe_error
from jev_bench.openrouter import CHAT_PATH, ApiResponse, JsonObject, OpenRouterClient, OpenRouterError, chat_content, json_schema_format
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta, GenerationStore
from jev_bench.store.status import JobStatus

CHECKPOINT_EVERY = 10


class GeneratorDeps(NamedTuple):
    client: OpenRouterClient
    api_key: str
    store: GenerationStore
    questions: QuestionSet
    config: GenerationConfig


class _Tally:
    def __init__(self) -> None:
        self.mismatches = 0


async def execute_generation(
    meta: GenerationMeta, plan: Sequence[PlanItem], deps: GeneratorDeps, progress: JobProgress
) -> None:
    started = time.perf_counter()
    tally = _Tally()
    try:
        await _generate_all(meta, plan, deps, progress, tally)
    except asyncio.CancelledError as exc:
        _finish(deps.store, meta, progress, tally, cancel_status(exc), None, started)
        raise
    except Exception as exc:
        logger.bind(generation=meta.id).opt(exception=exc).warning("generation failed")
        _finish(deps.store, meta, progress, tally, "failed", describe_error(exc), started)
        return
    _finish(deps.store, meta, progress, tally, "completed", None, started)


async def _generate_all(
    meta: GenerationMeta, plan: Sequence[PlanItem], deps: GeneratorDeps, progress: JobProgress, tally: _Tally
) -> None:
    traits = resolve_traits(deps.config, deps.questions)
    response_format = json_schema_format("email_generation", generation_schema(deps.questions))
    semaphore = asyncio.Semaphore(deps.config.concurrency)
    async with asyncio.TaskGroup() as group:
        for item in plan:
            group.create_task(_generate_one(meta, item, traits, response_format, deps, semaphore, progress, tally))


async def _generate_one(
    meta: GenerationMeta,
    item: PlanItem,
    traits: Sequence[ResolvedTrait],
    response_format: JsonObject,
    deps: GeneratorDeps,
    semaphore: asyncio.Semaphore,
    progress: JobProgress,
    tally: _Tally,
) -> None:
    try:
        async with semaphore:
            response = await _call(item, traits, response_format, deps)
        progress.cost += _cost(response.body)
        email, mismatched = _to_email(meta.id, item, traits, response, deps.questions)
    except OpenRouterError as exc:
        if exc.fatal:
            raise
        _item_failed(meta.id, item, str(exc), progress)
        return
    except ValueError as exc:
        _item_failed(meta.id, item, str(exc), progress)
        return
    _store_email(meta, email, mismatched, deps.store, progress, tally)


async def _call(
    item: PlanItem, traits: Sequence[ResolvedTrait], response_format: JsonObject, deps: GeneratorDeps
) -> ApiResponse:
    system, user = render_prompts(deps.config, traits, item, deps.questions)
    body = {
        "model": item.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": deps.config.temperature,
        "response_format": response_format,
        "provider": {"require_parameters": True},
    }
    return await deps.client.post_json(CHAT_PATH, body, api_key=deps.api_key)


def _to_email(
    generation_id: str, item: PlanItem, traits: Sequence[ResolvedTrait], response: ApiResponse, questions: QuestionSet
) -> tuple[Email, int]:
    output = parse_generator_output(chat_content(response.body), questions)
    email = Email(
        id=email_id(generation_id, item.index),
        sent_at=item.sent_at,
        sender=output.email.sender,
        to=output.email.to,
        cc=output.email.cc,
        subject=output.email.subject,
        body=output.email.body,
        generator_model=response.body.get("model") or item.model,
        traits=dict(item.traits),
        reference_answers=output.answers,
    )
    return email, count_mismatches(item, traits, output.answers)


def _store_email(
    meta: GenerationMeta, email: Email, mismatched: int, store: GenerationStore, progress: JobProgress, tally: _Tally
) -> None:
    store.append_email(meta.id, email)
    progress.done += 1
    tally.mismatches += mismatched
    if progress.done % CHECKPOINT_EVERY == 0:
        store.save(_totals(meta, progress, tally))


def _item_failed(generation_id: str, item: PlanItem, message: str, progress: JobProgress) -> None:
    progress.errors += 1
    logger.bind(generation=generation_id, item=item.index, model=item.model).warning("generation item failed: {}", message)


def _cost(body: Mapping[str, Any]) -> float:
    usage = body.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    return float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else 0.0


def _totals(meta: GenerationMeta, progress: JobProgress, tally: _Tally) -> GenerationMeta:
    update = {"done": progress.done, "errors": progress.errors, "total_cost": progress.cost, "trait_mismatches": tally.mismatches}
    return meta.model_copy(update=update)


def _finish(
    store: GenerationStore,
    meta: GenerationMeta,
    progress: JobProgress,
    tally: _Tally,
    status: JobStatus,
    error: str | None,
    started: float,
) -> None:
    finished = {"status": status, "error": error, "finished_at": datetime.now(UTC), "duration_s": time.perf_counter() - started}
    store.save(_totals(meta, progress, tally).model_copy(update=finished))


def mark_interrupted_generations(store: GenerationStore, is_live: Callable[[str], bool]) -> list[str]:
    orphaned = [meta for meta in store.list_metas() if meta.status == "running" and not is_live(meta.id)]
    for meta in orphaned:
        store.save(meta.model_copy(update={"status": "interrupted", "done": len(store.emails(meta.id))}))
    return [meta.id for meta in orphaned]
```

`src/jev_bench/generation/launcher.py`:
```python
"""Validates a generation request, builds its plan and meta, and starts it as a background job.

Classes:
    GenerationRequest: what the UI / CLI asks for.
Functions:
    launch_generation: persist the initial meta and start the job; returns (meta, task).
"""

import asyncio
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.generator import GeneratorDeps, execute_generation
from jev_bench.generation.plan import build_plan
from jev_bench.ids import new_id
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta

if TYPE_CHECKING:
    from jev_bench.services import Services


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="generation", min_length=1, max_length=80)
    count: int = Field(ge=1, le=2000)
    seed: int | None = Field(default=None, ge=0)
    models: tuple[str, ...] | None = None


async def launch_generation(
    request: GenerationRequest, api_key: str, services: "Services", *, now: datetime | None = None
) -> tuple[GenerationMeta, asyncio.Task[None]]:
    config = services.generation_config()
    questions = services.question_set()
    models = request.models or config.models
    seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
    started = now or datetime.now(UTC)
    plan = build_plan(config, questions, count=request.count, seed=seed, models=models, now=started)
    meta = _new_meta(request, config, questions, seed, models, started)
    services.generations.save(meta)
    deps = GeneratorDeps(client=services.client, api_key=api_key, store=services.generations, questions=questions, config=config)
    task = services.jobs.start(meta.id, request.count, lambda progress: execute_generation(meta, plan, deps, progress))
    return meta, task


def _new_meta(
    request: GenerationRequest,
    config: GenerationConfig,
    questions: QuestionSet,
    seed: int,
    models: Sequence[str],
    started: datetime,
) -> GenerationMeta:
    return GenerationMeta(
        id=new_id(request.name, started),
        name=request.name,
        created_at=started,
        requested=request.count,
        seed=seed,
        models=tuple(models),
        question_set=questions,
        config=config,
    )
```

`src/jev_bench/services.py`:
```python
"""Long-lived services shared by the web app and the CLI.

Classes:
    Services: stores, OpenRouter client, catalog, job registry and config loaders built from Settings.
"""

import httpx2

from jev_bench.benchmark_config import BenchmarkConfig, load_benchmark_config
from jev_bench.catalog import Catalog
from jev_bench.generation.config import GenerationConfig, load_generation_config
from jev_bench.generation.generator import mark_interrupted_generations
from jev_bench.jobs import JobRegistry
from jev_bench.openrouter import OpenRouterClient
from jev_bench.questions import QuestionSet, load_question_set
from jev_bench.runner import mark_interrupted_runs
from jev_bench.settings import Settings
from jev_bench.store.embeddings import EmbeddingCaches
from jev_bench.store.generations import GenerationStore
from jev_bench.store.labels import LabelStore
from jev_bench.store.runs import RunStore


class Services:
    def __init__(self, settings: Settings, http: httpx2.AsyncClient) -> None:
        self.settings = settings
        self.client = OpenRouterClient(http, max_retries=settings.max_retries, retry_base_delay_s=settings.retry_base_delay_s)
        self.catalog = Catalog(self.client)
        self.jobs = JobRegistry()
        self.generations = GenerationStore(settings.data_dir / "generations")
        self.runs = RunStore(settings.data_dir / "runs")
        self.labels = LabelStore(settings.data_dir / "labels")
        self.embedding_caches = EmbeddingCaches(settings.data_dir / "embeddings")

    def question_set(self) -> QuestionSet:
        return load_question_set(self.settings.config_dir / "questions.toml")

    def benchmark_config(self) -> BenchmarkConfig:
        return load_benchmark_config(self.settings.config_dir / "benchmark.toml")

    def generation_config(self) -> GenerationConfig:
        return load_generation_config(self.settings.config_dir / "generation.toml")

    def api_key(self, supplied: str | None) -> str | None:
        server = self.settings.server_api_key()
        if server:
            return server
        cleaned = (supplied or "").strip()
        return cleaned or None

    def sweep_interrupted(self) -> list[str]:
        runs = mark_interrupted_runs(self.runs, self.jobs.is_running)
        generations = mark_interrupted_generations(self.generations, self.jobs.is_running)
        return [*runs, *generations]
```

- [ ] **Step 5: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_generation_generator.py tests/test_generation_launcher.py tests/test_services.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/generation/generator.py src/jev_bench/generation/launcher.py src/jev_bench/services.py tests
git commit -m "feat(generation): generator job, launcher and shared Services container

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
