"""Generation persistence: `generation.json` meta plus append-only `emails.jsonl` per generation.

Classes:
    GenerationMeta: persisted description, config snapshot and totals of one generation.
    GenerationStore: save / get / list generations and read or append their emails.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import AwareDatetime, BaseModel

from jev_bench.emails import Email
from jev_bench.generation.config import GenerationConfig
from jev_bench.ids import is_safe_id, split_email_id
from jev_bench.questions import QuestionSet
from jev_bench.store.jsonfiles import append_jsonl, read_json, read_jsonl, write_json_atomic
from jev_bench.store.status import JobStatus

META_FILE = "generation.json"
EMAILS_FILE = "emails.jsonl"


class GenerationMeta(BaseModel):
    id: str
    name: str
    created_at: AwareDatetime
    status: JobStatus = "running"
    requested: int
    done: int = 0
    errors: int = 0
    seed: int
    models: tuple[str, ...]
    question_set: QuestionSet
    config: GenerationConfig
    total_cost: float = 0.0
    duration_s: float | None = None
    trait_mismatches: int = 0
    error: str | None = None
    finished_at: AwareDatetime | None = None


class GenerationStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, meta: GenerationMeta) -> None:
        write_json_atomic(self._dir(meta.id) / META_FILE, meta.model_dump(mode="json"))

    def get(self, generation_id: str) -> GenerationMeta:
        path = self._dir(generation_id) / META_FILE
        if not path.exists():
            raise KeyError(generation_id)
        return GenerationMeta.model_validate(read_json(path))

    def list_metas(self) -> list[GenerationMeta]:
        if not self._root.exists():
            return []
        ids = [
            path.name
            for path in self._root.iterdir()
            if is_safe_id(path.name) and (path / META_FILE).exists()
        ]
        return sorted(
            (self.get(generation_id) for generation_id in ids),
            key=lambda meta: meta.id,
            reverse=True,
        )

    def append_email(self, generation_id: str, email: Email) -> None:
        append_jsonl(self._dir(generation_id) / EMAILS_FILE, [email.model_dump(mode="json")])

    def emails(self, generation_id: str) -> list[Email]:
        directory = self._dir(generation_id)
        if not directory.exists():
            raise KeyError(generation_id)
        return [Email.model_validate(record) for record in read_jsonl(directory / EMAILS_FILE)]

    def emails_for(self, generation_ids: Iterable[str]) -> list[Email]:
        result: list[Email] = []
        for generation_id in dict.fromkeys(generation_ids):
            result.extend(self.emails(generation_id))
        return result

    def email(self, email_id: str) -> Email:
        generation_id, _ = split_email_id(email_id)
        for email in self.emails(generation_id):
            if email.id == email_id:
                return email
        raise KeyError(email_id)

    def _dir(self, generation_id: str) -> Path:
        if not is_safe_id(generation_id):
            raise KeyError(generation_id)
        return self._root / generation_id
