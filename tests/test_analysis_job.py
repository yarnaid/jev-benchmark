"""Tests for jev_bench.analysis.job."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest
from pydantic import SecretStr
from tests.factories import ClientFactory, FakeOpenRouter

from jev_bench.analysis import job
from jev_bench.analysis.job import AnalysisDeps, execute_analysis, mark_interrupted_analyses
from jev_bench.catalog import ModelInfo
from jev_bench.jobs import SHUTDOWN, JobProgress
from jev_bench.store.analyses import AnalysisMeta, AnalysisStore

_ID = "20260925-120000-analysis-0001"
_BODY = {"model": "anthropic/claude-sonnet-5", "messages": [{"role": "user", "content": "Hi"}]}
_PRICING = ModelInfo(id="m", name="m", prompt_price=0.000002, completion_price=0.00001)


class _Recording(AnalysisStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.saved: list[tuple[str, str]] = []

    def save(self, meta: AnalysisMeta) -> None:
        self.saved.append((meta.status, meta.result))
        super().save(meta)


def _meta(**update: object) -> AnalysisMeta:
    meta = AnalysisMeta(
        id=_ID,
        created_at=datetime(2026, 9, 25, 12, tzinfo=UTC),
        model="anthropic/claude-sonnet-5",
        run_ids=("r",),
        generation_ids=("g",),
        system_prompt="S",
        user_prompt="U",
        email_refs={},
        n_emails=1,
        n_disputed=0,
        max_output_tokens=500,
    )
    return meta.model_copy(update=update)


async def _run(
    make_client: ClientFactory,
    tmp_path: Path,
    handler: object,
    *,
    timeout_s: float = 1.0,
) -> tuple[AnalysisMeta, _Recording, JobProgress]:
    store = _Recording(tmp_path)
    deps = AnalysisDeps(make_client(handler), SecretStr("sk-test"), store, _PRICING, timeout_s)
    progress = JobProgress(2000, lambda: 0.0)
    await execute_analysis(_meta(), _BODY, deps, progress)
    return store.get(_ID), store, progress


async def test_completed_analysis(make_client: ClientFactory, tmp_path: Path) -> None:
    fake = FakeOpenRouter(analysis_parts=("## Summary\n", "Jev wins.\n"))
    meta, _, progress = await _run(make_client, tmp_path, fake)
    assert (meta.status, meta.result, meta.error) == ("completed", "## Summary\nJev wins.", None)
    assert (meta.input_tokens, meta.output_tokens, meta.cost) == (100, 20, 0.002)
    assert meta.resolved_model == "anthropic/claude-sonnet-5"
    assert (meta.duration_s is None, meta.finished_at is None) == (False, False)
    assert (progress.done, progress.cost) == (len("## Summary\nJev wins.\n"), 0.002)
    sent = fake.requests[-1]
    assert (sent.headers["authorization"], b'"stream":true' in sent.content) == (
        "Bearer sk-test",
        True,
    )


async def test_partial_text_is_checkpointed(
    make_client: ClientFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job, "CHECKPOINT_CHARS", 5)
    fake = FakeOpenRouter(analysis_parts=("123456", "789"))
    _, store, _ = await _run(make_client, tmp_path, fake)
    assert store.saved == [("running", "123456"), ("completed", "123456789")]


@pytest.mark.parametrize(
    ("fake", "error", "result"),
    [
        pytest.param(
            FakeOpenRouter(analysis_parts=("half",), analysis_finish="length"),
            "response truncated (finish_reason=length)",
            "half",
            id="truncated-keeps-partial-text",
        ),
        pytest.param(
            FakeOpenRouter(analysis_parts=("  ",)), "empty response", "  ", id="empty-reply"
        ),
        pytest.param(
            lambda request: httpx2.Response(402, json={"error": {"message": "no credits"}}),
            "HTTP 402: no credits",
            "",
            id="provider-refusal",
        ),
    ],
)
async def test_failed_analysis(
    make_client: ClientFactory, tmp_path: Path, fake: object, error: str, result: str
) -> None:
    meta, _, _ = await _run(make_client, tmp_path, fake)
    assert (meta.status, meta.error, meta.result) == ("failed", error, result)


async def _slow(request: httpx2.Request) -> httpx2.Response:
    await asyncio.sleep(1)
    return httpx2.Response(500)


async def test_timeout_fails_with_a_readable_message(
    make_client: ClientFactory, tmp_path: Path
) -> None:
    meta, _, _ = await _run(make_client, tmp_path, _slow, timeout_s=0.01)
    assert (meta.status, meta.error) == ("failed", "no complete reply within 0.01 s")


@pytest.mark.parametrize(
    ("message", "status"),
    [
        pytest.param(None, "cancelled", id="user-cancel"),
        pytest.param(SHUTDOWN, "interrupted", id="server-shutdown"),
    ],
)
async def test_cancelled_analysis(
    make_client: ClientFactory, tmp_path: Path, message: str | None, status: str
) -> None:
    store = AnalysisStore(tmp_path)
    deps = AnalysisDeps(make_client(_slow), SecretStr("sk-test"), store, None, 1.0)
    task = asyncio.create_task(execute_analysis(_meta(), _BODY, deps, JobProgress(1, lambda: 0.0)))
    await asyncio.sleep(0.005)
    task.cancel(message)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.get(_ID).status == status


def test_mark_interrupted_analyses(tmp_path: Path) -> None:
    store = AnalysisStore(tmp_path)
    live, orphan, done = (f"20260925-12000{i}-analysis-000{i}" for i in (1, 2, 3))
    store.save(_meta(id=live))
    store.save(_meta(id=orphan, result="partial"))
    store.save(_meta(id=done, status="completed"))
    assert mark_interrupted_analyses(store, lambda analysis_id: analysis_id == live) == [orphan]
    statuses = {meta.id: (meta.status, meta.result) for meta in store.list_metas()}
    assert statuses == {
        live: ("running", ""),
        orphan: ("interrupted", "partial"),
        done: ("completed", ""),
    }
