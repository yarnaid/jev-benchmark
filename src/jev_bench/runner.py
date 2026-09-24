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

from jev_bench.classifiers.base import (
    Classifier,
    EmailOutcome,
    PrepareResult,
    RequestResult,
    failed_result,
)
from jev_bench.emails import Email
from jev_bench.jobs import JobProgress, cancel_status, describe_error
from jev_bench.openrouter import OpenRouterError
from jev_bench.request_plan import RequestPlan, chunk_by_count, plan_requests
from jev_bench.store.runs import Prediction, ResponseRecord, RunMeta, RunStore
from jev_bench.store.status import JobStatus

OVERSIZE_ERROR = "exceeds model limits; not sent"


async def execute_run(
    meta: RunMeta,
    emails: Sequence[Email],
    classifier: Classifier,
    store: RunStore,
    progress: JobProgress,
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
    meta: RunMeta,
    emails: Sequence[Email],
    classifier: Classifier,
    store: RunStore,
    progress: JobProgress,
) -> None:
    prepared = await classifier.prepare(emails)
    progress.cost += prepared.usage.cost
    _record_outcomes(store, meta.id, prepared.resolved, progress)
    pending = list(prepared.pending)
    plan = _plan(classifier, pending)
    oversize = {pending[index].id: EmailOutcome(error=OVERSIZE_ERROR) for index in plan.oversize}
    _record_outcomes(store, meta.id, oversize, progress)
    store.save(_planned(meta, prepared, len(pending), plan))
    await _run_requests(
        meta.id,
        [[pending[index] for index in part] for part in plan.requests],
        classifier,
        store,
        progress,
    )


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
    run_id: str,
    batches: list[list[Email]],
    classifier: Classifier,
    store: RunStore,
    progress: JobProgress,
) -> None:
    semaphore = asyncio.Semaphore(classifier.concurrency)
    async with asyncio.TaskGroup() as group:
        for index, batch in enumerate(batches):
            group.create_task(
                _request(index, batch, classifier, store, run_id, semaphore, progress)
            )


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


async def _classify(
    index: int, batch: list[Email], classifier: Classifier, progress: JobProgress
) -> RequestResult:
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


def _record_outcomes(
    store: RunStore, run_id: str, outcomes: dict[str, EmailOutcome], progress: JobProgress
) -> None:
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
    store: RunStore,
    run_id: str,
    index: int,
    batch: list[Email],
    result: RequestResult,
    progress: JobProgress,
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
    outcome = result.outcomes.get(email.id) or EmailOutcome(
        error="no outcome returned for this email"
    )
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


def _finish(
    store: RunStore, run_id: str, status: JobStatus, error: str | None, started: float
) -> None:
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
            "resolved_models": tuple(
                sorted({p.resolved_model for p in predictions if p.resolved_model})
            ),
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
    orphaned = [
        meta for meta in store.list_metas() if meta.status == "running" and not is_live(meta.id)
    ]
    for meta in orphaned:
        summary = summarize(meta, store.predictions(meta.id))
        store.save(summary.model_copy(update={"status": "interrupted"}))
    return [meta.id for meta in orphaned]
