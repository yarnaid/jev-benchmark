"""Human labels per generation: `{email_id: {question_id: option_id}}`, edited atomically.

Classes:
    LabelStore: read labels per email or generation; apply edits under a lock.
"""

import asyncio
from collections.abc import Iterable, Mapping
from pathlib import Path

from jev_bench.ids import is_safe_id, split_email_id
from jev_bench.store.jsonfiles import read_json, write_json_atomic

__all__ = [
    "LabelStore",
    "Labels",
]

type Labels = dict[str, dict[str, str]]


class LabelStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = asyncio.Lock()

    def for_generations(self, generation_ids: Iterable[str]) -> Labels:
        merged: Labels = {}
        for generation_id in generation_ids:
            merged.update(self._read(generation_id))
        return merged

    def for_email(self, email_id: str) -> dict[str, str]:
        generation_id, _ = split_email_id(email_id)
        return self._read(generation_id).get(email_id, {})

    async def update(self, email_id: str, changes: Mapping[str, str | None]) -> dict[str, str]:
        generation_id, _ = split_email_id(email_id)
        async with self._lock:
            labels = self._read(generation_id)
            current = _apply(labels.get(email_id, {}), changes)
            if current:
                labels[email_id] = current
            else:
                labels.pop(email_id, None)
            write_json_atomic(self._path(generation_id), labels)
            return current

    def _read(self, generation_id: str) -> Labels:
        path = self._path(generation_id)
        return read_json(path) if path.exists() else {}

    def _path(self, generation_id: str) -> Path:
        if not is_safe_id(generation_id):
            raise KeyError(generation_id)
        return self._root / f"{generation_id}.json"


def _apply(current: Mapping[str, str], changes: Mapping[str, str | None]) -> dict[str, str]:
    updated = dict(current)
    for question_id, option in changes.items():
        if option is None:
            updated.pop(question_id, None)
        else:
            updated[question_id] = option
    return updated
