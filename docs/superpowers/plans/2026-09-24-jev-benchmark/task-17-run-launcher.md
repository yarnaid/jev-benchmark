### Task 17: Run launcher

**Files:**
- Create: `src/jev_bench/run_launcher.py`
- Modify: `tests/factories.py` (add `FakeOpenRouter` and `seed_generation`)
- Test: `tests/test_run_launcher.py`

**Interfaces:**
- Consumes:
  - Task 1: `new_id`;
  - Task 6: `BenchmarkConfig`, `ColumnConfig`, `ChatMode`, `JevParams`, `LlmParams`, `EmbeddingParams`,
    `ModelInfo`, `Catalog`;
  - Task 8: `JevClassifier`, `Classifier`;
  - Task 10: `LlmClassifier`;
  - Task 11: `RunMeta` (including `column_config`), `RunMode`;
  - Task 12: `EmbeddingClassifier`;
  - Task 14: `execute_run`;
  - Task 16: `Services`.
- Produces:
  - `jev_bench.run_launcher`:
    - `RunRequest(column, model=None, generation_ids (≥ 1), mode: ChatMode | None = None)`;
    - `RunLaunchError(ValueError)`;
    - `await launch_run(request, api_key, services, *, now=None) -> tuple[RunMeta, asyncio.Task[None]]`;
    - `build_classifier(column, model, mode, model_info, questions, config, services, api_key) ->
      Classifier`.
  - Tests:
    - `tests.factories.FakeOpenRouter`: a `MockTransport` handler serving `/v1/models` (per modality),
      `/alpha/decisions`, `/v1/chat/completions` and `/v1/embeddings` for the mini question set. Chat
      requests get JSON classification answers, SSE all-in-one results, or `generator_output()` when the
      schema name is `email_generation`. Attributes: `models_status: int = 200`,
      `requests: list[httpx2.Request]`;
    - `tests.factories.seed_generation(services, *, emails=2, generation_id="20260924-100000-seed-abcd")
      -> str`.

Validation, where every failure raises `RunLaunchError`:
- the column exists;
- `mode` is only allowed for chat columns;
- the generations exist and contain at least one email;
- the model is in the catalog for the column. When the catalog itself is unreachable, the run proceeds
  with fallback limits and a warning.

- [ ] **Step 1: Add the fake OpenRouter and generation seeding to `tests/factories.py`**

Add the imports next to the existing ones, list the names in the docstring, and append:
```python
from datetime import UTC, datetime

import httpx2

from jev_bench.store.generations import GenerationMeta

CHAT_ANSWER: dict[str, Any] = {
    "category": {"spam": 0.7, "personal": 0.2, "work": 0.1},
    "urgency": {"low": 0.1, "today": 0.3, "now": 0.6},
    "needs_reply": 0.2,
}
JEV_ANSWERS: dict[str, Any] = {
    "category": {"type": "choice", "choice": "spam", "probabilities": {"spam": 0.8, "personal": 0.1, "work": 0.1}},
    "urgency": {"type": "score", "score": 1.8, "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85}},
    "needs_reply": {"type": "noul", "noul": 0.15},
}
_CATALOG: dict[str, list[dict[str, Any]]] = {
    "text": [
        {"id": "anthropic/claude-sonnet-5", "name": "Claude Sonnet 5", "pricing": {"prompt": "0.000002", "completion": "0.00001"}, "context_length": 1000000, "top_provider": {"max_completion_tokens": 128000}, "supported_parameters": ["structured_outputs"]},
        {"id": "openai/gpt-5.6-terra", "name": "GPT-5.6 Terra", "pricing": {"prompt": "0.000002", "completion": "0.000012"}, "context_length": 1050000, "top_provider": {"max_completion_tokens": 128000}, "supported_parameters": ["structured_outputs"]},
    ],
    "decisions": [{"id": "typesafe/jev-1.13", "name": "Jev 1.13", "pricing": {"prompt": "0.000000042"}, "context_length": 32000}],
    "embeddings": [{"id": "openai/text-embedding-3-large", "name": "Embedding 3 Large", "pricing": {"prompt": "0.00000013"}, "context_length": 8192}],
}


def _vector(text: str) -> list[float]:
    return [float(len(text)), 1.0, float(sum(map(ord, text)) % 7)]


class FakeOpenRouter:
    def __init__(self, *, models_status: int = 200) -> None:
        self.models_status = models_status
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/v1/models"):
            return self._models(request)
        body = json.loads(request.content)
        if path.endswith("/alpha/decisions"):
            usage = {"input_tokens": 100, "output_tokens": 10, "cost": 0.00001}
            return httpx2.Response(200, json={"model": "typesafe/jev-1.13", "answers": JEV_ANSWERS, "usage": usage})
        if path.endswith("/v1/embeddings"):
            data = [{"index": i, "embedding": _vector(text)} for i, text in enumerate(body["input"])]
            usage = {"prompt_tokens": 5 * len(data), "cost": 0.00001 * len(data)}
            return httpx2.Response(200, json={"model": body["model"], "data": data, "usage": usage})
        return self._chat(body)

    def _models(self, request: httpx2.Request) -> httpx2.Response:
        if self.models_status != 200:
            return httpx2.Response(self.models_status, json={"error": {"message": "catalog down"}})
        modality = request.url.params.get("output_modalities") or "text"
        return httpx2.Response(200, json={"data": _CATALOG[modality]})

    def _chat(self, body: dict[str, Any]) -> httpx2.Response:
        if body["response_format"]["json_schema"]["name"] == "email_generation":
            return httpx2.Response(200, json=chat_body(generator_output(), model=body["model"]))
        if not body.get("stream"):
            return httpx2.Response(200, json=chat_body(json.dumps(CHAT_ANSWER), model=body["model"]))
        items = body["response_format"]["json_schema"]["schema"]["properties"]["results"]["items"]
        refs = items["properties"]["ref"]["enum"]
        content = json.dumps({"results": [{"ref": ref, **CHAT_ANSWER} for ref in refs]})
        headers = {"content-type": "text/event-stream"}
        return httpx2.Response(200, content=sse_body([content], model=body["model"]), headers=headers)


def seed_generation(services: Services, *, emails: int = 2, generation_id: str = "20260924-100000-seed-abcd") -> str:
    meta = GenerationMeta(
        id=generation_id,
        name="seed",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        status="completed",
        requested=emails,
        done=emails,
        seed=1,
        models=("gen/a",),
        question_set=services.question_set(),
        config=services.generation_config(),
    )
    services.generations.save(meta)
    for index in range(1, emails + 1):
        services.generations.append_email(generation_id, EmailFactory(id=f"{generation_id}.{index:04d}"))
    return generation_id
```

- [ ] **Step 2: Write the failing tests**

`tests/test_run_launcher.py`:
```python
"""Tests for jev_bench.run_launcher."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.run_launcher import RunLaunchError, RunRequest, launch_run
from tests.factories import FakeOpenRouter, ServicesFactory, seed_generation

_NOW = datetime(2026, 9, 24, 15, 30, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("column", "mode", "expected_mode", "per_request", "requests"),
    [
        pytest.param("jev", None, "per_email", 1, 2, id="jev"),
        pytest.param("anthropic", None, "per_email", 1, 2, id="chat-per-email"),
        pytest.param("anthropic", "all_in_one", "all_in_one", None, 1, id="chat-all-in-one"),
        pytest.param("embeddings", None, "batched", 2, 1, id="embeddings"),
    ],
)
async def test_launch_run_for_every_column_kind(
    make_services: ServicesFactory, column: str, mode: str | None, expected_mode: str, per_request: int | None, requests: int
) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services)
    request = RunRequest.model_validate({"column": column, "generation_ids": [generation_id], "mode": mode})
    meta, task = await launch_run(request, "sk-test", services, now=_NOW)
    await task
    final = services.runs.get(meta.id)
    assert meta.id.startswith(f"20260924-153012-{column}-")
    assert (meta.mode, meta.emails_per_request) == (expected_mode, per_request)
    assert meta.column_config is not None
    assert meta.column_config.id == column
    assert final.status == "completed"
    assert (final.n_done, final.n_errors, final.n_requests) == (2, 0, requests)
    assert final.generation_ids == (generation_id,)


async def test_catalog_outage_falls_back_to_default_limits(make_services: ServicesFactory) -> None:
    services = make_services(FakeOpenRouter(models_status=503))
    generation_id = seed_generation(services)
    meta, task = await launch_run(RunRequest(column="jev", generation_ids=(generation_id,)), "sk-test", services)
    await task
    assert services.runs.get(meta.id).status == "completed"


def _request(**fields: Any) -> RunRequest:
    return RunRequest.model_validate({"column": "jev", "generation_ids": ["20260924-100000-seed-abcd"], **fields})


@pytest.mark.parametrize(
    ("request_fields", "seed", "message"),
    [
        pytest.param({"column": "missing"}, True, "unknown column", id="unknown-column"),
        pytest.param({"mode": "all_in_one"}, True, "only accepted for chat", id="mode-on-jev"),
        pytest.param({"generation_ids": ["20260924-100000-none-abcd"]}, True, "unknown generation", id="unknown-generation"),
        pytest.param({}, False, "unknown generation", id="nothing-seeded"),
        pytest.param({"model": "typesafe/not-a-model"}, True, "not available", id="model-not-in-catalog"),
    ],
)
async def test_invalid_requests_raise_launch_errors(
    make_services: ServicesFactory, request_fields: dict[str, Any], seed: bool, message: str
) -> None:
    services = make_services(FakeOpenRouter())
    if seed:
        seed_generation(services)
    with pytest.raises(RunLaunchError, match=message):
        await launch_run(_request(**request_fields), "sk-test", services)


async def test_generation_without_emails_is_rejected(make_services: ServicesFactory) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services, emails=0)
    with pytest.raises(RunLaunchError, match="contain no emails"):
        await launch_run(_request(generation_ids=[generation_id]), "sk-test", services)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"column": "jev", "generation_ids": []}, id="no-generations"),
        pytest.param({"column": "jev", "generation_ids": ["g"], "extra": 1}, id="extra-field"),
        pytest.param({"column": "jev", "generation_ids": ["g"], "mode": "batched"}, id="bad-mode"),
    ],
)
def test_run_request_validation(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunRequest.model_validate(payload)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_launcher.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.run_launcher'`.

- [ ] **Step 4: Write the implementation**

`src/jev_bench/run_launcher.py`:
```python
"""Validates a run request, builds its classifier and meta, and starts it as a background job.

Classes:
    RunRequest: what the UI / CLI asks for.
    RunLaunchError: invalid request (unknown column/generation/model, empty email set, bad mode).
Functions:
    launch_run: persist the initial RunMeta and start the job; returns (meta, task).
    build_classifier: column + model + mode -> Classifier.
"""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from jev_bench.benchmark_config import BenchmarkConfig, ChatMode, ColumnConfig, EmbeddingParams, JevParams, LlmParams
from jev_bench.catalog import Catalog, ModelInfo
from jev_bench.classifiers.base import Classifier
from jev_bench.classifiers.embeddings import EmbeddingClassifier
from jev_bench.classifiers.jev import JevClassifier
from jev_bench.classifiers.llm import LlmClassifier
from jev_bench.emails import Email
from jev_bench.ids import new_id
from jev_bench.openrouter import OpenRouterError
from jev_bench.questions import QuestionSet
from jev_bench.runner import execute_run
from jev_bench.store.runs import RunMeta, RunMode

if TYPE_CHECKING:
    from jev_bench.services import Services


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    model: str | None = None
    generation_ids: tuple[str, ...] = Field(min_length=1)
    mode: ChatMode | None = None


class RunLaunchError(ValueError):
    pass


async def launch_run(
    request: RunRequest, api_key: str, services: "Services", *, now: datetime | None = None
) -> tuple[RunMeta, asyncio.Task[None]]:
    config = services.benchmark_config()
    column = _column(config, request.column)
    mode = _mode(column, request.mode)
    model = request.model or column.default_model
    emails = _emails(services, request.generation_ids)
    info = await _model_info(services.catalog, column, model)
    questions = services.question_set()
    classifier = build_classifier(column, model, mode, info, questions, config, services, api_key)
    meta = _new_meta(column, model, mode, request.generation_ids, questions, config, classifier, len(emails), now)
    services.runs.save(meta)
    task = services.jobs.start(meta.id, len(emails), lambda progress: execute_run(meta, emails, classifier, services.runs, progress))
    return meta, task


def _column(config: BenchmarkConfig, column_id: str) -> ColumnConfig:
    try:
        return config.column(column_id)
    except KeyError as exc:
        raise RunLaunchError(f"unknown column {column_id!r}") from exc


def _mode(column: ColumnConfig, requested: ChatMode | None) -> RunMode:
    if column.kind == "chat":
        return requested or "per_email"
    if requested is not None:
        raise RunLaunchError(f"mode is only accepted for chat columns, not {column.id!r}")
    return "batched" if column.kind == "embeddings" else "per_email"


def _emails(services: "Services", generation_ids: Sequence[str]) -> list[Email]:
    try:
        emails = services.generations.emails_for(generation_ids)
    except KeyError as exc:
        raise RunLaunchError(f"unknown generation {exc.args[0]!r}") from exc
    if not emails:
        raise RunLaunchError("the selected generations contain no emails")
    return emails


async def _model_info(catalog: Catalog, column: ColumnConfig, model: str) -> ModelInfo | None:
    try:
        models = await catalog.for_column(column)
    except OpenRouterError as exc:
        logger.bind(column=column.id).warning("catalog unavailable, using fallback limits: {}", exc)
        return None
    found = next((candidate for candidate in models if candidate.id == model), None)
    if found is None:
        raise RunLaunchError(f"model {model!r} is not available for column {column.id!r}")
    return found


def build_classifier(
    column: ColumnConfig,
    model: str,
    mode: RunMode,
    model_info: ModelInfo | None,
    questions: QuestionSet,
    config: BenchmarkConfig,
    services: "Services",
    api_key: str,
) -> Classifier:
    client = services.client
    if column.kind == "decisions":
        return JevClassifier(client=client, api_key=api_key, model=model, model_info=model_info, questions=questions, params=config.jev, tokens=config.tokens)
    if column.kind == "embeddings":
        cache = services.embedding_caches.for_model(model)
        return EmbeddingClassifier(client=client, api_key=api_key, model=model, model_info=model_info, questions=questions, params=config.embeddings, tokens=config.tokens, cache=cache)
    chat_mode: ChatMode = "all_in_one" if mode == "all_in_one" else "per_email"
    return LlmClassifier(client=client, api_key=api_key, model=model, model_info=model_info, questions=questions, params=config.llm, tokens=config.tokens, mode=chat_mode, cache_system_prompt=column.cache_system_prompt)


def _params(column: ColumnConfig, config: BenchmarkConfig) -> JevParams | LlmParams | EmbeddingParams:
    if column.kind == "decisions":
        return config.jev
    return config.embeddings if column.kind == "embeddings" else config.llm


def _new_meta(
    column: ColumnConfig,
    model: str,
    mode: RunMode,
    generation_ids: Sequence[str],
    questions: QuestionSet,
    config: BenchmarkConfig,
    classifier: Classifier,
    n_emails: int,
    now: datetime | None,
) -> RunMeta:
    created = now or datetime.now(UTC)
    return RunMeta(
        id=new_id(f"{column.id}-{model}", created),
        column=column.id,
        column_config=column,
        kind=column.kind,
        model=model,
        generation_ids=tuple(dict.fromkeys(generation_ids)),
        mode=mode,
        emails_per_request=classifier.emails_per_request,
        question_set=questions,
        params=_params(column, config),
        concurrency=classifier.concurrency,
        created_at=created,
        n_emails=n_emails,
    )
```

- [ ] **Step 5: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_run_launcher.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/run_launcher.py tests/factories.py tests/test_run_launcher.py
git commit -m "feat: run launcher validating column, mode, model and generations

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
