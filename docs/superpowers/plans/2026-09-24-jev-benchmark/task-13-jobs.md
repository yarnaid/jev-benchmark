### Task 13: Background job registry

**Files:**
- Create: `src/jev_bench/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `JobStatus` (Task 3).
- Produces:
  - `jev_bench.jobs.SHUTDOWN = "shutdown"`;
  - `ProgressView(total, done, errors, cost, elapsed_s)`;
  - `JobProgress(total, clock)` with mutable `done`, `errors`, `cost`, `streaming: dict[int, int]` and
    `.view() -> ProgressView`. The view reports `done` as `done + Σ streaming`, capped at `total`;
  - `JobBody = Callable[[JobProgress], Coroutine[Any, Any, None]]`;
  - `JobRegistry(clock=time.monotonic)` with `.start(job_id, total, body) -> asyncio.Task[None]`,
    `.progress(job_id) -> ProgressView | None`, `.is_running(job_id) -> bool`,
    `.cancel(job_id) -> bool` and `await .shutdown(timeout_s=5.0)`;
  - `cancel_status(exc: asyncio.CancelledError) -> JobStatus`: `"interrupted"` for `SHUTDOWN`,
    otherwise `"cancelled"`;
  - `describe_error(exc: BaseException) -> str`: the first leaf of an `ExceptionGroup`, or the class
    name when the message is empty.

Job bodies (the runner and the generator) catch `CancelledError`, persist `cancel_status(exc)`, and
re-raise. `shutdown()` cancels with the `SHUTDOWN` message, so a server stop is persisted as
`interrupted` and a user cancel as `cancelled`. A crashed body is logged, never silently dropped.

- [ ] **Step 1: Write the failing tests**

`tests/test_jobs.py`:
```python
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
    assert registry.progress("job-1") == ProgressView(total=10, done=7, errors=0, cost=0.5, elapsed_s=3.0)
    gate.set()
    await task
    await asyncio.sleep(0)
    assert not registry.is_running("job-1")
    assert registry.progress("job-1") is None


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
        pytest.param(ExceptionGroup("outer", [ExceptionGroup("inner", [KeyError("leaf")])]), "'leaf'", id="nested-group"),
        pytest.param(RuntimeError(), "RuntimeError", id="no-message"),
    ],
)
def test_describe_error(error: BaseException, expected: str) -> None:
    assert describe_error(error) == expected


async def test_crash_is_logged_and_forgotten() -> None:
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(str(message)), level="ERROR", format="{message}")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jobs.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.jobs'`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/jobs.py`:
```python
"""In-process registry of background jobs (runs, generations): live progress, cancel, shutdown.

Constants:
    SHUTDOWN: cancellation message used when the server stops (persisted as `interrupted`).
Types:
    JobBody
Classes:
    ProgressView: immutable progress snapshot served to the UI.
    JobProgress: live counters a job mutates.
    JobRegistry: start / progress / is_running / cancel / shutdown.
Functions:
    cancel_status: the status a cancelled job should persist.
    describe_error: human-readable message of an exception (first leaf of an ExceptionGroup).
"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any

from loguru import logger
from pydantic import BaseModel

from jev_bench.store.status import JobStatus

SHUTDOWN = "shutdown"

type Clock = Callable[[], float]


class ProgressView(BaseModel):
    total: int
    done: int
    errors: int
    cost: float
    elapsed_s: float


class JobProgress:
    def __init__(self, total: int, clock: Clock) -> None:
        self.total = total
        self.done = 0
        self.errors = 0
        self.cost = 0.0
        self.streaming: dict[int, int] = {}
        self._clock = clock
        self._started = clock()

    def view(self) -> ProgressView:
        done = min(self.total, self.done + sum(self.streaming.values()))
        elapsed = self._clock() - self._started
        return ProgressView(total=self.total, done=done, errors=self.errors, cost=self.cost, elapsed_s=elapsed)


type JobBody = Callable[[JobProgress], Coroutine[Any, Any, None]]


class JobRegistry:
    def __init__(self, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._progress: dict[str, JobProgress] = {}

    def start(self, job_id: str, total: int, body: JobBody) -> asyncio.Task[None]:
        progress = JobProgress(total, self._clock)
        task = asyncio.create_task(body(progress), name=job_id)
        self._tasks[job_id] = task
        self._progress[job_id] = progress
        task.add_done_callback(lambda finished: self._forget(job_id, finished))
        return task

    def progress(self, job_id: str) -> ProgressView | None:
        progress = self._progress.get(job_id)
        return progress.view() if progress is not None else None

    def is_running(self, job_id: str) -> bool:
        return job_id in self._tasks

    def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        return task.cancel() if task is not None else False

    async def shutdown(self, timeout_s: float = 5.0) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel(SHUTDOWN)
        if tasks:
            await asyncio.wait(tasks, timeout=timeout_s)

    def _forget(self, job_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(job_id, None)
        self._progress.pop(job_id, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.bind(job=job_id).opt(exception=error).error("background job crashed")


def cancel_status(exc: asyncio.CancelledError) -> JobStatus:
    return "interrupted" if exc.args and exc.args[0] == SHUTDOWN else "cancelled"


def describe_error(exc: BaseException) -> str:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return str(exc) or type(exc).__name__
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_jobs.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/jobs.py tests/test_jobs.py
git commit -m "feat: background job registry with progress, cancel and shutdown semantics

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
