### Task 14: Benchmark run executor

**Files:**
- Create: `src/jev_bench/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes:
  - Task 2: `Email`;
  - Task 5: `OpenRouterError`;
  - Task 7: `chunk_by_count`, `plan_requests`, `Sizing`, `Budget`;
  - Task 8: `Classifier`, `EmailOutcome`, `PrepareResult`, `RequestResult`, `Usage`, `failed_result`;
  - Task 11: `RunMeta`, `Prediction`, `ResponseRecord`, `RunStore`;
  - Task 13: `JobProgress`, `JobRegistry`, `cancel_status`, `describe_error`.
- Produces:
  - `jev_bench.runner.OVERSIZE_ERROR = "exceeds model limits; not sent"`;
  - `await execute_run(meta, emails, classifier, store, progress) -> None`: a job body that always
    finalizes `run.json` and re-raises only `CancelledError`;
  - `summarize(meta, predictions) -> RunMeta` (totals recomputed from predictions);
  - `mark_interrupted_runs(store, is_live: Callable[[str], bool]) -> list[str]`.

Flow (spec §7):
1. `prepare()` (cache-resolved emails are recorded immediately).
2. Plan: chunk by `emails_per_request`, then `plan_requests` with the classifier's budget and sizing.
   Oversize emails get `OVERSIZE_ERROR` and are never sent.
3. Save meta with `n_requests`, `n_splits`, setup costs and cache counts.
4. Run requests in a `TaskGroup` under a semaphore of `classifier.concurrency`:
   - a non-fatal `OpenRouterError` becomes per-email errors;
   - a fatal one (401/402/403) aborts the run as `failed`.
5. Finalize: `summarize` from disk, plus `status`, `error`, `finished_at` and `duration_s`.

Cost and tokens of a multi-email request are split evenly across its emails. Latency percentiles are
computed over distinct requests.

- [ ] **Step 1: Write the failing tests**

`tests/test_runner.py`:
```python
"""Tests for jev_bench.runner."""

import asyncio
import contextlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_bench.benchmark_config import ColumnKind, EmbeddingParams, JevParams
from jev_bench.classifiers.base import EmailOutcome, PrepareResult, ProgressCallback, RequestResult, Usage
from jev_bench.emails import Email
from jev_bench.jobs import JobProgress, JobRegistry
from jev_bench.openrouter import OpenRouterError
from jev_bench.questions import QuestionSet
from jev_bench.request_plan import Sizing
from jev_bench.runner import OVERSIZE_ERROR, execute_run, mark_interrupted_runs, summarize
from jev_bench.store.runs import Prediction, RunMeta, RunStore
from jev_bench.tokens import Budget
from tests.factories import EmailFactory

_ANSWER = {"needs_reply": {"yes": 0.9, "no": 0.1}}


class FakeClassifier:
    def __init__(
        self,
        *,
        emails_per_request: int | None = 1,
        budget: Budget = Budget(total=10_000),
        fail: dict[str, OpenRouterError] | None = None,
        resolved: dict[str, EmailOutcome] | None = None,
        sizes: dict[str, int] | None = None,
        gate: asyncio.Event | None = None,
        prepare_error: Exception | None = None,
    ) -> None:
        self.emails_per_request = emails_per_request
        self.concurrency = 4
        self.budget = budget
        self.sizing = Sizing()
        self.calls: list[list[str]] = []
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
        return PrepareResult(usage=Usage(cost=0.01), cached_cost=0.02, resolved=self._resolved, pending=pending)

    async def classify(self, emails: Sequence[Email], on_progress: ProgressCallback | None = None) -> RequestResult:
        self.calls.append([email.id for email in emails])
        if on_progress is not None:
            on_progress(len(emails))
        if self._gate is not None:
            await self._gate.wait()
        for email in emails:
            if email.id in self._fail:
                raise self._fail[email.id]
        outcomes = {email.id: EmailOutcome(answers=_ANSWER) for email in emails}
        usage = Usage(input_tokens=10 * len(emails), output_tokens=4, cost=0.001 * len(emails))
        return RequestResult(outcomes=outcomes, usage=usage, latency_ms=10.0 * len(self.calls), resolved_model="test/resolved", raw={"ok": True})


def _meta(questions: QuestionSet, kind: ColumnKind = "decisions", n_emails: int = 3) -> RunMeta:
    params = EmbeddingParams(email_template="$body", option_template="$description") if kind == "embeddings" else JevParams()
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


async def _run(tmp_path: Path, questions: QuestionSet, classifier: FakeClassifier, emails: list[Email], kind: ColumnKind = "decisions") -> tuple[RunMeta, JobProgress, RunStore]:
    store = RunStore(tmp_path)
    meta = _meta(questions, kind, len(emails))
    store.save(meta)
    progress = JobProgress(len(emails), lambda: 0.0)
    await execute_run(meta, emails, classifier, store, progress)
    return store.get(meta.id), progress, store


async def test_per_email_run_completes_and_summarizes(tmp_path: Path, questions: QuestionSet) -> None:
    emails = [EmailFactory() for _ in range(3)]
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


async def test_all_in_one_is_split_by_the_budget_and_costs_are_shared(tmp_path: Path, questions: QuestionSet) -> None:
    emails = [EmailFactory() for _ in range(3)]
    classifier = FakeClassifier(emails_per_request=None, budget=Budget(total=25))
    final, _, store = await _run(tmp_path, questions, classifier, emails)
    assert [len(call) for call in classifier.calls] == [2, 1]
    assert (final.n_requests, final.n_splits) == (2, 1)
    shared = [p for p in store.predictions(final.id) if p.batch_size == 2]
    assert [p.cost for p in shared] == pytest.approx([0.001, 0.001])
    assert [p.input_tokens for p in shared] == [10.0, 10.0]


async def test_non_fatal_error_is_recorded_per_email(tmp_path: Path, questions: QuestionSet) -> None:
    emails = [EmailFactory() for _ in range(3)]
    classifier = FakeClassifier(fail={emails[1].id: OpenRouterError("HTTP 500: boom", status=500, retryable=True)})
    final, progress, store = await _run(tmp_path, questions, classifier, emails)
    errors = {p.email_id: p.error for p in store.predictions(final.id) if p.error}
    assert final.status == "completed"
    assert errors == {emails[1].id: "HTTP 500: boom"}
    assert (final.n_errors, progress.errors) == (1, 1)


@pytest.mark.parametrize(
    ("fatal", "prepare_error", "message"),
    [
        pytest.param(OpenRouterError("HTTP 401: no auth", status=401), None, "401", id="fatal-http"),
        pytest.param(None, OpenRouterError("expected 2 embeddings, got 1"), "expected 2 embeddings", id="prepare-error"),
        pytest.param(None, RuntimeError("bug"), "bug", id="unexpected-exception"),
    ],
)
async def test_failures_finalize_as_failed(
    tmp_path: Path, questions: QuestionSet, fatal: OpenRouterError | None, prepare_error: Exception | None, message: str
) -> None:
    emails = [EmailFactory() for _ in range(2)]
    fail = {emails[0].id: fatal} if fatal is not None else None
    classifier = FakeClassifier(fail=fail, prepare_error=prepare_error)
    final, _, _ = await _run(tmp_path, questions, classifier, emails)
    assert final.status == "failed"
    assert final.error is not None
    assert message in final.error


async def test_cached_emails_are_recorded_without_requests(tmp_path: Path, questions: QuestionSet) -> None:
    emails = [EmailFactory() for _ in range(3)]
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
    emails = [EmailFactory(subject="small"), EmailFactory(subject="huge")]
    classifier = FakeClassifier(budget=Budget(total=50), sizes={"huge": 100})
    final, _, store = await _run(tmp_path, questions, classifier, emails)
    errors = {p.email_id: p.error for p in store.predictions(final.id)}
    assert classifier.calls == [[emails[0].id]]
    assert errors == {emails[0].id: None, emails[1].id: OVERSIZE_ERROR}


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        pytest.param("cancel", "cancelled", id="cancel"),
        pytest.param("shutdown", "interrupted", id="shutdown"),
    ],
)
async def test_cancellation_is_persisted(tmp_path: Path, questions: QuestionSet, stop: str, expected: str) -> None:
    store = RunStore(tmp_path)
    meta = _meta(questions)
    store.save(meta)
    registry = JobRegistry()
    classifier = FakeClassifier(gate=asyncio.Event())
    emails = [EmailFactory()]
    task = registry.start(meta.id, 1, lambda progress: execute_run(meta, emails, classifier, store, progress))
    for _ in range(5):
        await asyncio.sleep(0)
    if stop == "cancel":
        registry.cancel(meta.id)
    else:
        await registry.shutdown(timeout_s=0.05)
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert store.get(meta.id).status == expected


def test_summarize_percentiles_use_distinct_requests(questions: QuestionSet) -> None:
    predictions = [
        Prediction(email_id="a", request_index=0, batch_size=2, latency_ms=100.0, cost=0.5),
        Prediction(email_id="b", request_index=0, batch_size=2, latency_ms=100.0, cost=0.5),
        Prediction(email_id="c", request_index=1, batch_size=1, latency_ms=300.0, cost=1.0, error="x"),
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
    store.append_predictions(orphan.id, [Prediction(email_id="a", request_index=0, batch_size=1, cost=0.1)])
    marked = mark_interrupted_runs(store, lambda run_id: run_id == live.id)
    assert marked == [orphan.id]
    assert (store.get(orphan.id).status, store.get(orphan.id).n_done) == ("interrupted", 1)
    assert store.get(live.id).status == "running"
    assert store.get(done.id).status == "completed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.runner'`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/runner.py`:
```python
"""Executes one benchmark run: token-budgeted request plan, concurrent classification, persistence.

Constants:
    OVERSIZE_ERROR
Functions:
    execute_run: job body for one run (always finalizes the meta; re-raises only CancelledError).
    summarize: RunMeta totals recomputed from persisted predictions.
    mark_interrupted_runs: startup sweep turning orphaned `running` runs into `interrupted`.
"""

import asyncio
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import numpy as np
from loguru import logger

from jev_bench.classifiers.base import Classifier, EmailOutcome, PrepareResult, RequestResult, failed_result
from jev_bench.emails import Email
from jev_bench.jobs import JobProgress, cancel_status, describe_error
from jev_bench.openrouter import OpenRouterError
from jev_bench.request_plan import RequestPlan, chunk_by_count, plan_requests
from jev_bench.store.runs import Prediction, ResponseRecord, RunMeta, RunStore
from jev_bench.store.status import JobStatus

OVERSIZE_ERROR = "exceeds model limits; not sent"


async def execute_run(
    meta: RunMeta, emails: Sequence[Email], classifier: Classifier, store: RunStore, progress: JobProgress
) -> None:
    started = time.perf_counter()
    try:
        await _execute(meta, emails, classifier, store, progress)
    except asyncio.CancelledError as exc:
        _finish(store, meta.id, cancel_status(exc), None, started)
        raise
    except Exception as exc:
        logger.bind(run=meta.id).opt(exception=exc).warning("run failed")
        _finish(store, meta.id, "failed", describe_error(exc), started)
        return
    _finish(store, meta.id, "completed", None, started)


async def _execute(
    meta: RunMeta, emails: Sequence[Email], classifier: Classifier, store: RunStore, progress: JobProgress
) -> None:
    prepared = await classifier.prepare(emails)
    progress.cost += prepared.usage.cost
    _record_outcomes(store, meta.id, prepared.resolved, progress)
    pending = list(prepared.pending)
    plan = _plan(classifier, pending)
    oversize = {pending[index].id: EmailOutcome(error=OVERSIZE_ERROR) for index in plan.oversize}
    _record_outcomes(store, meta.id, oversize, progress)
    store.save(_planned(meta, prepared, len(pending), plan))
    await _run_requests(meta.id, [[pending[index] for index in part] for part in plan.requests], classifier, store, progress)


def _plan(classifier: Classifier, pending: Sequence[Email]) -> RequestPlan:
    chunks = chunk_by_count(len(pending), classifier.emails_per_request)
    sizes = [classifier.input_tokens(email) for email in pending]
    return plan_requests(chunks, sizes, classifier.budget, classifier.sizing)


def _planned(meta: RunMeta, prepared: PrepareResult, pending: int, plan: RequestPlan) -> RunMeta:
    embeddings = meta.kind == "embeddings"
    return meta.model_copy(
        update={
            "n_requests": len(plan.requests),
            "n_splits": plan.splits,
            "setup_cost": prepared.usage.cost,
            "setup_cached_cost": prepared.cached_cost,
            "cache_hits": len(prepared.resolved) if embeddings else 0,
            "cache_misses": pending if embeddings else 0,
        }
    )


async def _run_requests(
    run_id: str, batches: list[list[Email]], classifier: Classifier, store: RunStore, progress: JobProgress
) -> None:
    semaphore = asyncio.Semaphore(classifier.concurrency)
    async with asyncio.TaskGroup() as group:
        for index, batch in enumerate(batches):
            group.create_task(_request(index, batch, classifier, store, run_id, semaphore, progress))


async def _request(
    index: int,
    batch: list[Email],
    classifier: Classifier,
    store: RunStore,
    run_id: str,
    semaphore: asyncio.Semaphore,
    progress: JobProgress,
) -> None:
    async with semaphore:
        result = await _classify(index, batch, classifier, progress)
    _record_request(store, run_id, index, batch, result, progress)


async def _classify(index: int, batch: list[Email], classifier: Classifier, progress: JobProgress) -> RequestResult:
    def on_progress(count: int) -> None:
        progress.streaming[index] = count

    try:
        return await classifier.classify(batch, on_progress)
    except OpenRouterError as exc:
        if exc.fatal:
            raise
        return failed_result(batch, str(exc))
    finally:
        progress.streaming.pop(index, None)


def _record_outcomes(store: RunStore, run_id: str, outcomes: dict[str, EmailOutcome], progress: JobProgress) -> None:
    predictions = [_standalone(email_id, outcome) for email_id, outcome in outcomes.items()]
    store.append_predictions(run_id, predictions)
    progress.done += len(predictions)
    progress.errors += sum(prediction.error is not None for prediction in predictions)


def _standalone(email_id: str, outcome: EmailOutcome) -> Prediction:
    return Prediction(
        email_id=email_id,
        answers=outcome.answers,
        error=outcome.error,
        notes=outcome.notes,
        similarities=outcome.similarities,
        cached=outcome.cached,
        cached_cost=outcome.cached_cost,
    )


def _record_request(
    store: RunStore, run_id: str, index: int, batch: list[Email], result: RequestResult, progress: JobProgress
) -> None:
    predictions = [_prediction(email, result, index, len(batch)) for email in batch]
    store.append_predictions(run_id, predictions)
    record = ResponseRecord(
        request_index=index,
        email_ids=tuple(email.id for email in batch),
        latency_ms=result.latency_ms,
        error=result.error,
        body=result.raw,
    )
    store.append_response(run_id, record)
    progress.done += len(predictions)
    progress.errors += sum(prediction.error is not None for prediction in predictions)
    progress.cost += result.usage.cost


def _prediction(email: Email, result: RequestResult, index: int, size: int) -> Prediction:
    outcome = result.outcomes.get(email.id) or EmailOutcome(error="no outcome returned for this email")
    share = 1.0 / size
    return Prediction(
        email_id=email.id,
        answers=outcome.answers,
        error=outcome.error,
        notes=outcome.notes,
        request_index=index,
        batch_size=size,
        latency_ms=result.latency_ms,
        cost=result.usage.cost * share,
        input_tokens=result.usage.input_tokens * share,
        output_tokens=result.usage.output_tokens * share,
        cost_estimated=result.usage.cost_estimated,
        resolved_model=result.resolved_model,
        similarities=outcome.similarities,
    )


def _finish(store: RunStore, run_id: str, status: JobStatus, error: str | None, started: float) -> None:
    summary = summarize(store.get(run_id), store.predictions(run_id))
    finished = {
        "status": status,
        "error": error,
        "finished_at": datetime.now(UTC),
        "duration_s": time.perf_counter() - started,
    }
    store.save(summary.model_copy(update=finished))


def summarize(meta: RunMeta, predictions: Sequence[Prediction]) -> RunMeta:
    paid = sum(prediction.cost for prediction in predictions)
    cached = sum(prediction.cached_cost for prediction in predictions)
    latencies = _request_latencies(predictions)
    return meta.model_copy(
        update={
            "n_done": len(predictions),
            "n_errors": sum(prediction.error is not None for prediction in predictions),
            "total_cost": meta.setup_cost + paid,
            "cold_cost": meta.setup_cost + meta.setup_cached_cost + paid + cached,
            "input_tokens": round(sum(prediction.input_tokens for prediction in predictions)),
            "output_tokens": round(sum(prediction.output_tokens for prediction in predictions)),
            "resolved_models": tuple(sorted({p.resolved_model for p in predictions if p.resolved_model})),
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
        }
    )


def _request_latencies(predictions: Sequence[Prediction]) -> list[float]:
    by_request = {
        prediction.request_index: prediction.latency_ms
        for prediction in predictions
        if prediction.request_index is not None and prediction.latency_ms is not None
    }
    return list(by_request.values())


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def mark_interrupted_runs(store: RunStore, is_live: Callable[[str], bool]) -> list[str]:
    orphaned = [meta for meta in store.list_metas() if meta.status == "running" and not is_live(meta.id)]
    for meta in orphaned:
        summary = summarize(meta, store.predictions(meta.id))
        store.save(summary.model_copy(update={"status": "interrupted"}))
    return [meta.id for meta in orphaned]
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_runner.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/runner.py tests/test_runner.py
git commit -m "feat: benchmark run executor with token-budgeted plan and cost apportioning

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
