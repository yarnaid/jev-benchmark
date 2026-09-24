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
        return ProgressView(
            total=self.total,
            done=done,
            errors=self.errors,
            cost=self.cost,
            elapsed_s=elapsed,
        )


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
