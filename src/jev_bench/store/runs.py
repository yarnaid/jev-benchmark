"""Run persistence: `run.json` meta, append-only predictions and raw responses per run.

Types:
    RunMode, RunParams
Classes:
    RunMeta: persisted description, snapshots and totals of one run.
    Prediction: one email's outcome within a run.
    ResponseRecord: one HTTP request's raw outcome (stored once per request).
    RunStore: save / get / list runs and read or append their predictions and responses.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

from loguru import logger
from pydantic import AwareDatetime, BaseModel, Field

from jev_bench.benchmark_config import (
    ColumnConfig,
    ColumnKind,
    EmbeddingParams,
    JevParams,
    LlmParams,
)
from jev_bench.ids import is_safe_id
from jev_bench.questions import Distribution, QuestionSet
from jev_bench.store.jsonfiles import append_jsonl, read_json, read_jsonl, write_json_atomic
from jev_bench.store.status import JobStatus

__all__ = [
    "META_FILE",
    "PREDICTIONS_FILE",
    "RESPONSES_FILE",
    "Prediction",
    "ResponseRecord",
    "RunMeta",
    "RunMode",
    "RunParams",
    "RunStore",
]

META_FILE = "run.json"
PREDICTIONS_FILE = "predictions.jsonl"
RESPONSES_FILE = "responses.jsonl"

type RunMode = Literal["per_email", "all_in_one", "batched"]
RunParams = Annotated[JevParams | LlmParams | EmbeddingParams, Field(discriminator="kind")]


class RunMeta(BaseModel):
    id: str
    column: str
    column_config: ColumnConfig | None = None
    kind: ColumnKind
    model: str
    resolved_models: tuple[str, ...] = ()
    generation_ids: tuple[str, ...]
    mode: RunMode
    emails_per_request: int | None
    question_set: QuestionSet
    params: RunParams
    concurrency: int
    status: JobStatus = "running"
    created_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    duration_s: float | None = None
    n_emails: int
    n_done: int = 0
    n_errors: int = 0
    n_requests: int = 0
    n_splits: int = 0
    total_cost: float = 0.0
    setup_cost: float = 0.0
    setup_cached_cost: float = 0.0
    cold_cost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    error: str | None = None


class Prediction(BaseModel):
    email_id: str
    answers: dict[str, Distribution] | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    request_index: int | None = None
    batch_size: int = 0
    latency_ms: float | None = None
    cost: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cost_estimated: bool = False
    resolved_model: str | None = None
    similarities: dict[str, dict[str, float]] | None = None
    cached: bool = False
    cached_cost: float = 0.0


class ResponseRecord(BaseModel):
    request_index: int
    email_ids: tuple[str, ...]
    latency_ms: float | None = None
    error: str | None = None
    body: dict[str, Any] | None = None


class RunStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, meta: RunMeta) -> None:
        write_json_atomic(self._dir(meta.id) / META_FILE, meta.model_dump(mode="json"))

    def get(self, run_id: str) -> RunMeta:
        path = self._dir(run_id) / META_FILE
        if not path.exists():
            raise KeyError(run_id)
        return RunMeta.model_validate(read_json(path))

    def list_metas(self) -> list[RunMeta]:
        if not self._root.exists():
            return []
        ids = [
            path.name
            for path in self._root.iterdir()
            if is_safe_id(path.name) and (path / META_FILE).exists()
        ]
        metas = [meta for meta in (self._safe_get(run_id) for run_id in ids) if meta is not None]
        return sorted(metas, key=lambda meta: meta.id, reverse=True)

    def _safe_get(self, run_id: str) -> RunMeta | None:
        try:
            return self.get(run_id)
        except ValueError as exc:
            logger.bind(path=str(self._dir(run_id) / META_FILE)).warning(
                "skipping unreadable meta: {}", exc
            )
            return None

    def append_predictions(self, run_id: str, predictions: Iterable[Prediction]) -> None:
        records = [prediction.model_dump(mode="json") for prediction in predictions]
        append_jsonl(self._dir(run_id) / PREDICTIONS_FILE, records)

    def predictions(self, run_id: str) -> list[Prediction]:
        path = self._existing(run_id) / PREDICTIONS_FILE
        return [Prediction.model_validate(record) for record in read_jsonl(path)]

    def append_response(self, run_id: str, record: ResponseRecord) -> None:
        append_jsonl(self._dir(run_id) / RESPONSES_FILE, [record.model_dump(mode="json")])

    def responses(self, run_id: str) -> list[ResponseRecord]:
        path = self._existing(run_id) / RESPONSES_FILE
        return [ResponseRecord.model_validate(record) for record in read_jsonl(path)]

    def _existing(self, run_id: str) -> Path:
        directory = self._dir(run_id)
        if not directory.exists():
            raise KeyError(run_id)
        return directory

    def _dir(self, run_id: str) -> Path:
        if not is_safe_id(run_id):
            raise KeyError(run_id)
        return self._root / run_id
