"""Tests for jev_bench.jobs."""

import asyncio
import contextlib

import pytest
from loguru import logger

from jev_bench.jobs import JobProgress, JobRegistry, ProgressView, cancel_status, describe_error


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_start_tracks_progress_and_forgets_when_done() -> None:
    clock = FakeClock()
    registry = JobRegistry(clock)
    gate = asyncio.Event()

    async def body(progress: JobProgress) -> None:
        progress.done = 2
        progress.cost = 0.5
        progress.streaming[0] = 5
        await gate.wait()

    task = registry.start("job-1", 10, body)
    await asyncio.sleep(0)
    clock.now = 3.0
    assert registry.is_running("job-1")
    view = registry.progress("job-1")
    assert view == ProgressView(total=10, done=7, errors=0, cost=0.5, elapsed_s=3.0)
    gate.set()
    await task
    await asyncio.sleep(0)
    assert not registry.is_running("job-1")
    assert registry.progress("job-1") is None


async def test_start_raises_on_duplicate_job_id() -> None:
    registry = JobRegistry()
    gate = asyncio.Event()

    async def body(progress: JobProgress) -> None:
        await gate.wait()

    task = registry.start("job", 1, body)
    await asyncio.sleep(0)
    assert registry.is_running("job")
    with pytest.raises(ValueError, match="job 'job' is already running"):
        registry.start("job", 1, body)
    assert registry.is_running("job")
    assert registry.progress("job") is not None
    gate.set()
    await task
    await asyncio.sleep(0)
    assert not registry.is_running("job")


def test_progress_view_caps_done_at_total() -> None:
    progress = JobProgress(3, FakeClock())
    progress.done = 2
    progress.streaming[1] = 5
    assert progress.view().done == 3


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        pytest.param("cancel", "cancelled", id="user-cancel"),
        pytest.param("shutdown", "interrupted", id="server-shutdown"),
    ],
)
async def test_cancellation_status(stop: str, expected: str) -> None:
    registry = JobRegistry()
    statuses: list[str] = []

    async def body(progress: JobProgress) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            statuses.append(cancel_status(exc))
            raise

    task = registry.start("job", 1, body)
    await asyncio.sleep(0)
    if stop == "cancel":
        assert registry.cancel("job") is True
    else:
        await registry.shutdown(timeout_s=0.05)
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert statuses == [expected]


async def test_cancel_unknown_job_and_empty_shutdown() -> None:
    registry = JobRegistry()
    assert registry.cancel("missing") is False
    await registry.shutdown(timeout_s=0.01)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(ValueError("plain"), "plain", id="plain"),
        pytest.param(
            ExceptionGroup("outer", [ExceptionGroup("inner", [KeyError("leaf")])]),
            "'leaf'",
            id="nested-group",
        ),
        pytest.param(RuntimeError(), "RuntimeError", id="no-message"),
    ],
)
def test_describe_error(error: BaseException, expected: str) -> None:
    assert describe_error(error) == expected


async def test_crash_is_logged_and_forgotten() -> None:
    messages: list[str] = []
    handler = logger.add(
        lambda message: messages.append(str(message)),
        level="ERROR",
        format="{message}",
    )
    registry = JobRegistry()

    async def body(progress: JobProgress) -> None:
        raise RuntimeError("boom")

    try:
        task = registry.start("job", 1, body)
        with pytest.raises(RuntimeError, match="boom"):
            await task
        await asyncio.sleep(0)
    finally:
        logger.remove(handler)
    assert any("background job crashed" in message for message in messages)
    assert not registry.is_running("job")
