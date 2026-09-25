"""Executes one analysis: a streamed chat completion whose text is checkpointed into the analysis
record as it arrives (a live preview), then the final result, usage, cost and status.

Constants:
    CHARS_PER_TOKEN: progress total = output-token budget x this (an upper bound on characters).
    CHECKPOINT_CHARS: persist the partial text after this many new characters.
Classes:
    AnalysisDeps: collaborators of an analysis job (`api_key` is a `SecretStr`, never persisted).
Functions:
    execute_analysis: job body (always finalizes the record; re-raises only CancelledError). The
        result is the final reply's text; a truncated or empty reply fails the analysis but keeps
        its text, and a call that fails before any reply keeps the streamed preview.
    mark_interrupted_analyses: startup sweep for analyses left `running` (the preview is kept).
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, NamedTuple

from pydantic import SecretStr

from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import usage_from_body
from jev_bench.failures import failure_text, log_job_failure
from jev_bench.jobs import JobProgress, cancel_status
from jev_bench.openrouter import (
    CHAT_PATH,
    ApiResponse,
    ChatContentError,
    JsonObject,
    OpenRouterClient,
    chat_content,
)
from jev_bench.store.analyses import AnalysisMeta, AnalysisStore
from jev_bench.store.status import JobStatus

__all__ = [
    "CHARS_PER_TOKEN",
    "CHECKPOINT_CHARS",
    "AnalysisDeps",
    "execute_analysis",
    "mark_interrupted_analyses",
]

CHARS_PER_TOKEN = 4
CHECKPOINT_CHARS = 2000


class AnalysisDeps(NamedTuple):
    client: OpenRouterClient
    api_key: SecretStr
    store: AnalysisStore
    pricing: ModelInfo | None
    timeout_s: float


class _Preview:
    def __init__(self, meta: AnalysisMeta, store: AnalysisStore, progress: JobProgress) -> None:
        self._meta = meta
        self._store = store
        self._progress = progress
        self._parts: list[str] = []
        self._saved = 0

    def text(self) -> str:
        return "".join(self._parts)

    def feed(self, text: str) -> None:
        self._parts.append(text)
        self._progress.done += len(text)
        if self._progress.done - self._saved >= CHECKPOINT_CHARS:
            self._saved = self._progress.done
            self._store.save(self._meta.model_copy(update={"result": self.text()}))


async def execute_analysis(
    meta: AnalysisMeta, body: Mapping[str, Any], deps: AnalysisDeps, progress: JobProgress
) -> None:
    started = time.perf_counter()
    preview = _Preview(meta, deps.store, progress)
    finish = _finisher(meta, deps.store, started)
    try:
        response = await _call(body, deps, preview)
    except asyncio.CancelledError as exc:
        finish(cancel_status(exc), {"result": preview.text()})
        raise
    except Exception as exc:
        text = _failure(exc, deps.timeout_s)
        log_job_failure(exc, text, analysis=meta.id)
        finish("failed", {"result": preview.text(), "error": text})
        return
    status, update = _outcome(response, deps.pricing, progress)
    finish(status, update)


async def _call(body: Mapping[str, Any], deps: AnalysisDeps, preview: _Preview) -> ApiResponse:
    async with asyncio.timeout(deps.timeout_s):
        return await deps.client.post_stream(
            CHAT_PATH, body, api_key=deps.api_key.get_secret_value(), on_text=preview.feed
        )


def _failure(exc: Exception, timeout_s: float) -> str:
    if isinstance(exc, TimeoutError):
        return f"no complete reply within {timeout_s:g} s"
    return failure_text(exc)


def _outcome(
    response: ApiResponse, pricing: ModelInfo | None, progress: JobProgress
) -> tuple[JobStatus, dict[str, object]]:
    usage = usage_from_body(response.body.get("usage"), pricing)
    progress.cost = usage.cost
    update: dict[str, object] = {
        "resolved_model": response.body.get("model"),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost": usage.cost,
        "cost_estimated": usage.cost_estimated,
    }
    try:
        return "completed", {**update, "result": chat_content(response.body)}
    except ChatContentError as exc:
        return "failed", {**update, "result": _reply_text(response.body), "error": str(exc)}


def _reply_text(body: JsonObject) -> str:
    try:
        return str(body["choices"][0]["message"]["content"])
    except KeyError, IndexError, TypeError:
        return ""


def _finisher(
    meta: AnalysisMeta, store: AnalysisStore, started: float
) -> Callable[[JobStatus, dict[str, object]], None]:
    def finish(status: JobStatus, update: dict[str, object]) -> None:
        finished = {
            "status": status,
            "finished_at": datetime.now(UTC),
            "duration_s": time.perf_counter() - started,
        }
        store.save(meta.model_copy(update={**update, **finished}))

    return finish


def mark_interrupted_analyses(store: AnalysisStore, is_live: Callable[[str], bool]) -> list[str]:
    orphaned = [
        meta for meta in store.list_metas() if meta.status == "running" and not is_live(meta.id)
    ]
    for meta in orphaned:
        store.save(meta.model_copy(update={"status": "interrupted"}))
    return [meta.id for meta in orphaned]
