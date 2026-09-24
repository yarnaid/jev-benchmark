"""Tests for jev_bench.generation.generator."""

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr
from tests.factories import ClientFactory, EmailFactory, chat_body, generator_output

from jev_bench.generation import generator
from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.generator import (
    GeneratorDeps,
    execute_generation,
    mark_interrupted_generations,
)
from jev_bench.generation.plan import PlanItem, build_plan
from jev_bench.jobs import JobProgress, JobRegistry
from jev_bench.openrouter import OpenRouterClient
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta, GenerationStore

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
    def __init__(
        self, failures: dict[int, httpx2.Response] | None = None, content: str | None = None
    ) -> None:
        self.failures = failures or {}
        self.content = content
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        call = len(self.bodies)
        if call in self.failures:
            return self.failures[call]
        return httpx2.Response(
            200, json=chat_body(self.content or generator_output("spam"), model="gen/resolved")
        )


def _meta(
    questions: QuestionSet, requested: int = 3, config: GenerationConfig = _CONFIG
) -> GenerationMeta:
    return GenerationMeta(
        id=_GEN_ID,
        name="gen",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        requested=requested,
        seed=5,
        models=config.models,
        question_set=questions,
        config=config,
    )


def _plan(
    questions: QuestionSet, count: int = 3, config: GenerationConfig = _CONFIG
) -> list[PlanItem]:
    return build_plan(
        config,
        questions,
        count=count,
        seed=5,
        models=config.models,
        now=datetime(2026, 9, 24, tzinfo=UTC),
    )


async def _generate(
    tmp_path: Path, client: OpenRouterClient, questions: QuestionSet, plan: list[PlanItem]
) -> tuple[GenerationMeta, GenerationStore]:
    store = GenerationStore(tmp_path)
    meta = _meta(questions, len(plan))
    store.save(meta)
    deps = GeneratorDeps(
        client=client,
        api_key=SecretStr("sk-test"),
        store=store,
        questions=questions,
        config=_CONFIG,
    )
    await execute_generation(meta, plan, deps, JobProgress(len(plan), lambda: 0.0))
    return store.get(meta.id), store


async def test_generation_persists_emails_and_totals(
    tmp_path: Path, make_client: ClientFactory, questions: QuestionSet
) -> None:
    server = GeneratorServer()
    plan = _plan(questions)
    final, store = await _generate(tmp_path, make_client(server), questions, plan)
    emails = store.emails(_GEN_ID)
    assert final.status == "completed"
    assert (final.done, final.errors) == (3, 0)
    assert final.total_cost == pytest.approx(0.003)
    assert final.trait_mismatches == sum(item.traits["category"] != "spam" for item in plan)
    assert sorted(email.id for email in emails) == [
        f"{_GEN_ID}.0001",
        f"{_GEN_ID}.0002",
        f"{_GEN_ID}.0003",
    ]
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
        pytest.param(
            GeneratorServer(
                failures={2: httpx2.Response(500, json={"error": {"message": "boom"}})}
            ),
            1,
            id="http-500",
        ),
        pytest.param(GeneratorServer(content='{"email": {}}'), 3, id="invalid-output"),
        pytest.param(
            GeneratorServer(
                failures={1: httpx2.Response(200, json=chat_body("{}", finish_reason="length"))}
            ),
            1,
            id="truncated",
        ),
    ],
)
async def test_item_failures_do_not_stop_the_generation(
    tmp_path: Path,
    make_client: ClientFactory,
    questions: QuestionSet,
    server: GeneratorServer,
    errors: int,
) -> None:
    final, store = await _generate(tmp_path, make_client(server), questions, _plan(questions))
    assert final.status == "completed"
    assert final.errors == errors
    assert len(store.emails(_GEN_ID)) == 3 - errors


async def test_fatal_error_fails_the_generation(
    tmp_path: Path, make_client: ClientFactory, questions: QuestionSet
) -> None:
    server = GeneratorServer(
        failures={1: httpx2.Response(401, json={"error": {"message": "no auth"}})}
    )
    final, _ = await _generate(tmp_path, make_client(server), questions, _plan(questions, 1))
    assert final.status == "failed"
    assert final.error is not None
    assert "401" in final.error


async def test_fatal_failure_never_logs_the_api_key(
    tmp_path: Path, make_client: ClientFactory, questions: QuestionSet, log_records: list[Any]
) -> None:
    secret = "sk-or-v1-TOPSECRET-4242"
    server = GeneratorServer(
        failures={1: httpx2.Response(402, json={"error": {"message": "no credit"}})}
    )
    store = GenerationStore(tmp_path)
    meta = _meta(questions, 1)
    store.save(meta)
    deps = GeneratorDeps(
        client=make_client(server),
        api_key=SecretStr(secret),
        store=store,
        questions=questions,
        config=_CONFIG,
    )
    await execute_generation(meta, _plan(questions, 1), deps, JobProgress(1, lambda: 0.0))
    final = store.get(meta.id)
    assert final.status == "failed"
    assert secret not in repr(deps)
    assert not any(secret in record for record in log_records)


async def test_non_provider_job_failure_is_logged_at_error_with_type_prefix(
    tmp_path: Path,
    make_client: ClientFactory,
    questions: QuestionSet,
    monkeypatch: pytest.MonkeyPatch,
    log_records: list[Any],
) -> None:
    def _boom(self: GenerationStore, generation_id: str, email: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(GenerationStore, "append_email", _boom)
    store = GenerationStore(tmp_path)
    meta = _meta(questions, 1)
    store.save(meta)
    deps = GeneratorDeps(
        client=make_client(GeneratorServer()),
        api_key=SecretStr("k"),
        store=store,
        questions=questions,
        config=_CONFIG,
    )
    await execute_generation(meta, _plan(questions, 1), deps, JobProgress(1, lambda: 0.0))
    final = store.get(meta.id)
    assert final.status == "failed"
    assert final.error is not None
    assert final.error.startswith("OSError: ")
    error_records = [record for record in log_records if record.record["level"].name == "ERROR"]
    assert error_records


async def test_checkpoints_are_written(
    tmp_path: Path,
    make_client: ClientFactory,
    questions: QuestionSet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator, "CHECKPOINT_EVERY", 2)
    config = _CONFIG.model_copy(update={"concurrency": 1})
    gate = asyncio.Event()
    calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 3:
            await gate.wait()
        return httpx2.Response(200, json=chat_body(generator_output("spam"), model="gen/resolved"))

    store = GenerationStore(tmp_path)
    plan = _plan(questions, 5, config)
    meta = _meta(questions, 5, config)
    store.save(meta)
    deps = GeneratorDeps(
        client=make_client(handler),
        api_key=SecretStr("k"),
        store=store,
        questions=questions,
        config=config,
    )
    registry = JobRegistry()
    task = registry.start(
        meta.id, 5, lambda progress: execute_generation(meta, plan, deps, progress)
    )
    for _ in range(500):
        if len(store.emails(meta.id)) >= 2:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("checkpoint condition never reached")
    on_disk = store.get(meta.id)
    assert (on_disk.status, on_disk.done) == ("running", 2)
    gate.set()
    await task
    assert store.get(meta.id).status == "completed"


async def test_concurrency_is_bounded_by_the_semaphore(
    tmp_path: Path, make_client: ClientFactory, questions: QuestionSet
) -> None:
    in_flight = 0
    max_in_flight = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return httpx2.Response(200, json=chat_body(generator_output("spam"), model="gen/resolved"))

    await _generate(tmp_path, make_client(handler), questions, _plan(questions, 5))
    assert max_in_flight == 2


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        pytest.param("cancel", "cancelled", id="cancel"),
        pytest.param("shutdown", "interrupted", id="shutdown"),
    ],
)
async def test_cancellation_is_persisted(
    tmp_path: Path, make_client: ClientFactory, questions: QuestionSet, stop: str, expected: str
) -> None:
    async def hang(request: httpx2.Request) -> httpx2.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    store = GenerationStore(tmp_path)
    meta = _meta(questions, 1)
    store.save(meta)
    deps = GeneratorDeps(
        client=make_client(hang),
        api_key=SecretStr("k"),
        store=store,
        questions=questions,
        config=_CONFIG,
    )
    registry = JobRegistry()
    task = registry.start(
        meta.id, 1, lambda progress: execute_generation(meta, _plan(questions, 1), deps, progress)
    )
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
    store.append_email(orphan.id, EmailFactory(id=f"{orphan.id}.0001", traits={"category": "spam"}))
    assert mark_interrupted_generations(store, lambda generation_id: generation_id == live.id) == [
        orphan.id
    ]
    assert (store.get(orphan.id).status, store.get(orphan.id).done) == ("interrupted", 1)
    assert store.get(live.id).status == "running"


def test_mark_interrupted_generations_recomputes_trait_mismatches(
    tmp_path: Path, questions: QuestionSet
) -> None:
    store = GenerationStore(tmp_path)
    meta = _meta(questions, 6)
    store.save(meta)
    categories = ["personal", "personal", "work", "work", "spam", "spam"]
    for index, category in enumerate(categories, start=1):
        store.append_email(
            meta.id,
            EmailFactory(
                id=f"{meta.id}.{index:04d}",
                traits={"category": "spam"},
                reference_answers={"category": category, "urgency": "now", "needs_reply": "no"},
            ),
        )
    assert mark_interrupted_generations(store, lambda _: False) == [meta.id]
    final = store.get(meta.id)
    assert (final.status, final.done, final.trait_mismatches) == ("interrupted", 6, 4)
