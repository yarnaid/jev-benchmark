"""Tests for jev_bench.runner."""

import asyncio
import contextlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from tests.factories import EmailFactory

from jev_bench.benchmark_config import ColumnKind, EmbeddingParams, JevParams
from jev_bench.classifiers.base import (
    Classifier,
    EmailOutcome,
    PrepareResult,
    ProgressCallback,
    RequestResult,
    Usage,
)
from jev_bench.emails import Email
from jev_bench.jobs import JobProgress, JobRegistry
from jev_bench.openrouter import OpenRouterError
from jev_bench.questions import QuestionSet
from jev_bench.request_plan import Sizing
from jev_bench.runner import OVERSIZE_ERROR, execute_run, mark_interrupted_runs, summarize
from jev_bench.store.runs import Prediction, RunMeta, RunStore
from jev_bench.tokens import Budget

_ANSWER = {"needs_reply": {"yes": 0.9, "no": 0.1}}


class FakeClassifier:
    def __init__(
        self,
        *,
        emails_per_request: int | None = 1,
        budget: Budget | None = None,
        concurrency: int = 4,
        fail: dict[str, Exception] | None = None,
        resolved: dict[str, EmailOutcome] | None = None,
        sizes: dict[str, int] | None = None,
        gate: asyncio.Event | None = None,
        prepare_error: Exception | None = None,
    ) -> None:
        self.emails_per_request = emails_per_request
        self.concurrency = concurrency
        self.budget = budget if budget is not None else Budget(total=10_000)
        self.sizing = Sizing()
        self.calls: list[list[str]] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._fail = fail or {}
        self._resolved = resolved or {}
        self._sizes = sizes or {}
        self._gate = gate
        self._prepare_error = prepare_error

    def input_tokens(self, email: Email) -> int:
        return self._sizes.get(email.subject, 10)

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        if self._prepare_error is not None:
            raise self._prepare_error
        pending = tuple(email for email in emails if email.id not in self._resolved)
        return PrepareResult(
            usage=Usage(cost=0.01), cached_cost=0.02, resolved=self._resolved, pending=pending
        )

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        self.calls.append([email.id for email in emails])
        latency_ms = 10.0 * len(self.calls)
        if on_progress is not None:
            on_progress(len(emails))
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0)
            if self._gate is not None:
                await self._gate.wait()
            for email in emails:
                if email.id in self._fail:
                    raise self._fail[email.id]
            outcomes = {email.id: EmailOutcome(answers=_ANSWER) for email in emails}
            usage = Usage(input_tokens=10 * len(emails), output_tokens=4, cost=0.001 * len(emails))
            return RequestResult(
                outcomes=outcomes,
                usage=usage,
                latency_ms=latency_ms,
                resolved_model="test/resolved",
                raw={"ok": True},
            )
        finally:
            self._in_flight -= 1


def _meta(questions: QuestionSet, kind: ColumnKind = "decisions", n_emails: int = 3) -> RunMeta:
    params = (
        EmbeddingParams(email_template="$body", option_template="$description")
        if kind == "embeddings"
        else JevParams()
    )
    return RunMeta(
        id="20260924-100000-test-run-0001",
        column="test",
        kind=kind,
        model="test/model",
        generation_ids=("20260924-100000-gen-abcd",),
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=params,
        concurrency=4,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=n_emails,
    )


async def _run(
    tmp_path: Path,
    questions: QuestionSet,
    classifier: Classifier,
    emails: list[Email],
    kind: ColumnKind = "decisions",
) -> tuple[RunMeta, JobProgress, RunStore]:
    store = RunStore(tmp_path)
    meta = _meta(questions, kind, len(emails))
    store.save(meta)
    progress = JobProgress(len(emails), lambda: 0.0)
    await execute_run(meta, emails, classifier, store, progress)
    return store.get(meta.id), progress, store


async def test_per_email_run_completes_and_summarizes(
    tmp_path: Path, questions: QuestionSet
) -> None:
    emails: list[Email] = [EmailFactory() for _ in range(3)]
    final, progress, store = await _run(tmp_path, questions, FakeClassifier(), emails)
    assert final.status == "completed"
    assert (final.n_done, final.n_errors, final.n_requests, final.n_splits) == (3, 0, 3, 0)
    assert final.total_cost == pytest.approx(0.01 + 0.003)
    assert final.resolved_models == ("test/resolved",)
    assert final.latency_p50_ms == pytest.approx(20.0)
    assert final.duration_s is not None
    assert final.finished_at is not None
    assert len(store.responses(final.id)) == 3
    assert (progress.done, progress.streaming) == (3, {})


async def test_concurrency_is_bounded_by_the_semaphore(
    tmp_path: Path, questions: QuestionSet
) -> None:
    emails: list[Email] = [EmailFactory() for _ in range(5)]
    classifier = FakeClassifier(concurrency=2)
    await _run(tmp_path, questions, classifier, emails)
    assert classifier.max_in_flight == 2


async def test_all_in_one_is_split_by_the_budget_and_costs_are_shared(
    tmp_path: Path, questions: QuestionSet
) -> None:
    emails: list[Email] = [EmailFactory() for _ in range(3)]
    classifier = FakeClassifier(emails_per_request=None, budget=Budget(total=25))
    final, _, store = await _run(tmp_path, questions, classifier, emails)
    assert [len(call) for call in classifier.calls] == [2, 1]
    assert (final.n_requests, final.n_splits) == (2, 1)
    shared = [p for p in store.predictions(final.id) if p.batch_size == 2]
    assert [p.cost for p in shared] == pytest.approx([0.001, 0.001])
    assert [p.input_tokens for p in shared] == [10.0, 10.0]


async def test_non_fatal_error_is_recorded_per_email(
    tmp_path: Path, questions: QuestionSet
) -> None:
    emails: list[Email] = [EmailFactory() for _ in range(3)]
    classifier = FakeClassifier(
        fail={emails[1].id: OpenRouterError("HTTP 500: boom", status=500, retryable=True)}
    )
    final, progress, store = await _run(tmp_path, questions, classifier, emails)
    errors = {p.email_id: p.error for p in store.predictions(final.id) if p.error}
    assert final.status == "completed"
    assert errors == {emails[1].id: "HTTP 500: boom"}
    assert (final.n_errors, progress.errors) == (1, 1)


@pytest.mark.parametrize(
    ("fatal", "prepare_error", "message"),
    [
        pytest.param(
            OpenRouterError("HTTP 401: no auth", status=401), None, "401", id="fatal-http"
        ),
        pytest.param(
            None,
            OpenRouterError("expected 2 embeddings, got 1"),
            "expected 2 embeddings",
            id="prepare-error",
        ),
        pytest.param(None, RuntimeError("bug"), "bug", id="unexpected-exception"),
        pytest.param(
            KeyError("needs_reply"),
            None,
            "KeyError: 'needs_reply'",
            id="bug-inside-task-group",
        ),
    ],
)
async def test_failures_finalize_as_failed(
    tmp_path: Path,
    questions: QuestionSet,
    fatal: Exception | None,
    prepare_error: Exception | None,
    message: str,
) -> None:
    emails: list[Email] = [EmailFactory() for _ in range(2)]
    fail = {emails[0].id: fatal} if fatal is not None else None
    classifier = FakeClassifier(fail=fail, prepare_error=prepare_error)
    final, _, _ = await _run(tmp_path, questions, classifier, emails)
    assert final.status == "failed"
    assert final.error is not None
    assert message in final.error


async def test_cached_emails_are_recorded_without_requests(
    tmp_path: Path, questions: QuestionSet
) -> None:
    emails: list[Email] = [EmailFactory() for _ in range(3)]
    cached = EmailOutcome(answers=_ANSWER, cached=True, cached_cost=0.0005)
    classifier = FakeClassifier(resolved={emails[0].id: cached})
    final, _, store = await _run(tmp_path, questions, classifier, emails, kind="embeddings")
    first = next(p for p in store.predictions(final.id) if p.email_id == emails[0].id)
    assert classifier.calls == [[emails[1].id], [emails[2].id]]
    assert (first.cached, first.request_index) == (True, None)
    assert (final.cache_hits, final.cache_misses) == (1, 2)
    assert final.total_cost == pytest.approx(0.01 + 0.002)
    assert final.cold_cost == pytest.approx(0.01 + 0.02 + 0.002 + 0.0005)


async def test_oversize_email_is_never_sent(tmp_path: Path, questions: QuestionSet) -> None:
    emails: list[Email] = [EmailFactory(subject="small"), EmailFactory(subject="huge")]
    classifier = FakeClassifier(budget=Budget(total=50), sizes={"huge": 100})
    final, _, store = await _run(tmp_path, questions, classifier, emails)
    errors = {p.email_id: p.error for p in store.predictions(final.id)}
    assert classifier.calls == [[emails[0].id]]
    assert errors == {emails[0].id: None, emails[1].id: OVERSIZE_ERROR}


async def test_earlier_requests_are_persisted_before_a_fatal_failure(
    tmp_path: Path, questions: QuestionSet
) -> None:
    emails: list[Email] = [EmailFactory() for _ in range(3)]
    classifier = FakeClassifier(
        concurrency=1,
        fail={emails[2].id: OpenRouterError("HTTP 402: no credit", status=402)},
    )
    final, _, store = await _run(tmp_path, questions, classifier, emails)
    assert final.status == "failed"
    assert {p.email_id for p in store.predictions(final.id)} == {emails[0].id, emails[1].id}
    assert [r.request_index for r in store.responses(final.id)] == [0, 1]


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        pytest.param("cancel", "cancelled", id="cancel"),
        pytest.param("shutdown", "interrupted", id="shutdown"),
    ],
)
async def test_cancellation_is_persisted(
    tmp_path: Path, questions: QuestionSet, stop: str, expected: str
) -> None:
    store = RunStore(tmp_path)
    meta = _meta(questions)
    store.save(meta)
    registry = JobRegistry()
    classifier = FakeClassifier(gate=asyncio.Event())
    emails = [EmailFactory()]
    task = registry.start(
        meta.id, 1, lambda progress: execute_run(meta, emails, classifier, store, progress)
    )
    for _ in range(5):
        await asyncio.sleep(0)
    if stop == "cancel":
        registry.cancel(meta.id)
    else:
        await registry.shutdown(timeout_s=0.05)
    with contextlib.suppress(asyncio.CancelledError):
        await task
    final = store.get(meta.id)
    assert final.status == expected
    assert final.finished_at is not None
    assert final.duration_s is not None


def test_summarize_percentiles_use_distinct_requests(questions: QuestionSet) -> None:
    predictions = [
        Prediction(email_id="a", request_index=0, batch_size=2, latency_ms=100.0, cost=0.5),
        Prediction(email_id="b", request_index=0, batch_size=2, latency_ms=100.0, cost=0.5),
        Prediction(
            email_id="c", request_index=1, batch_size=1, latency_ms=300.0, cost=1.0, error="x"
        ),
    ]
    summary = summarize(_meta(questions).model_copy(update={"setup_cost": 0.25}), predictions)
    assert (summary.n_done, summary.n_errors) == (3, 1)
    assert summary.total_cost == pytest.approx(2.25)
    assert summary.latency_p50_ms == pytest.approx(200.0)


def test_mark_interrupted_runs(tmp_path: Path, questions: QuestionSet) -> None:
    store = RunStore(tmp_path)
    orphan = _meta(questions)
    live = orphan.model_copy(update={"id": "20260924-110000-live-run-0002"})
    done = orphan.model_copy(update={"id": "20260924-120000-done-run-0003", "status": "completed"})
    for meta in (orphan, live, done):
        store.save(meta)
    store.append_predictions(
        orphan.id, [Prediction(email_id="a", request_index=0, batch_size=1, cost=0.1)]
    )
    marked = mark_interrupted_runs(store, lambda run_id: run_id == live.id)
    assert marked == [orphan.id]
    interrupted = store.get(orphan.id)
    assert (interrupted.status, interrupted.n_done) == ("interrupted", 1)
    assert interrupted.total_cost == pytest.approx(0.1)
    assert store.get(live.id).status == "running"
    assert store.get(done.id).status == "completed"


class _KeyLeakingClassifier:
    def __init__(self, api_key: str, *, unexpected: bool = False) -> None:
        self._api_key = SecretStr(api_key)
        self._unexpected = unexpected
        self.emails_per_request = 1
        self.concurrency = 1
        self.budget = Budget(total=10_000)
        self.sizing = Sizing()

    def input_tokens(self, email: Email) -> int:
        return 10

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        return PrepareResult(pending=tuple(emails))

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        if self._unexpected:
            return await _unexpected_request(self._api_key.get_secret_value())
        return await _fatal_request(self._api_key.get_secret_value())


async def _fatal_request(api_key: str) -> RequestResult:
    raise OpenRouterError("HTTP 402: insufficient credits", status=402)


async def _unexpected_request(api_key: str) -> RequestResult:
    raise RuntimeError("Cannot send a request, as the client has been closed.")


@pytest.mark.parametrize(
    "unexpected",
    [
        pytest.param(False, id="fatal-openrouter-error"),
        pytest.param(True, id="unexpected-runtime-error"),
    ],
)
async def test_fatal_failure_never_logs_the_api_key(
    tmp_path: Path, questions: QuestionSet, log_records: list[Any], unexpected: bool
) -> None:
    secret = "sk-or-v1-RUNNERSECRET-1337"
    classifier = _KeyLeakingClassifier(secret, unexpected=unexpected)
    final, _, _ = await _run(tmp_path, questions, classifier, [EmailFactory()])
    assert final.status == "failed"
    assert not any(secret in record for record in log_records)
