"""Analysis persistence: one `<id>.json` document per analysis.

Classes:
    AnalysisMeta: request, prompts actually sent, result Markdown (partial while streaming), usage,
        cost, duration and status of one analysis. `email_refs` maps the e001… refs the analyst saw
        to email ids.
    AnalysisStore: save / get / list analyses (newest first; unreadable files are skipped).
"""

from pathlib import Path

from loguru import logger
from pydantic import AwareDatetime, BaseModel

from jev_bench.ids import is_safe_id
from jev_bench.store.jsonfiles import read_json, write_json_atomic
from jev_bench.store.status import JobStatus

__all__ = [
    "AnalysisMeta",
    "AnalysisStore",
]


class AnalysisMeta(BaseModel):
    id: str
    created_at: AwareDatetime
    status: JobStatus = "running"
    model: str
    resolved_model: str | None = None
    run_ids: tuple[str, ...]
    generation_ids: tuple[str, ...]
    threshold: float | None = None
    system_prompt: str
    user_prompt: str
    email_refs: dict[str, str]
    n_emails: int
    n_disputed: int
    max_output_tokens: int
    temperature: float | None = None
    result: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cost_estimated: bool = False
    duration_s: float | None = None
    error: str | None = None
    finished_at: AwareDatetime | None = None


class AnalysisStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, meta: AnalysisMeta) -> None:
        write_json_atomic(self._path(meta.id), meta.model_dump(mode="json"))

    def get(self, analysis_id: str) -> AnalysisMeta:
        path = self._path(analysis_id)
        if not path.exists():
            raise KeyError(analysis_id)
        return AnalysisMeta.model_validate(read_json(path))

    def list_metas(self) -> list[AnalysisMeta]:
        if not self._root.exists():
            return []
        ids = [path.stem for path in self._root.glob("*.json") if is_safe_id(path.stem)]
        metas = [meta for meta in map(self._safe_get, ids) if meta is not None]
        return sorted(metas, key=lambda meta: meta.id, reverse=True)

    def _safe_get(self, analysis_id: str) -> AnalysisMeta | None:
        try:
            return self.get(analysis_id)
        except ValueError as exc:
            logger.bind(path=str(self._path(analysis_id))).warning(
                "skipping unreadable analysis: {}", exc
            )
            return None

    def _path(self, analysis_id: str) -> Path:
        if not is_safe_id(analysis_id):
            raise KeyError(analysis_id)
        return self._root / f"{analysis_id}.json"
