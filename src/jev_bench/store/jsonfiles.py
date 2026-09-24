"""JSON / JSONL persistence primitives shared by every store.

Functions:
    write_json_atomic: write a JSON document via a unique temp file + os.replace.
    read_json: read and parse a JSON document.
    append_jsonl: append records as JSON lines, always starting on a fresh line.
    read_jsonl: parse every line; unparseable (torn or corrupt) lines are skipped with a warning.
"""

import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, records: Iterable[object]) -> None:
    def _record_line(rec: object) -> str:
        return json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"

    lines = "".join(_record_line(record) for record in records)
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "\n" if _ends_without_newline(path) else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + lines)


def _ends_without_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) != b"\n"


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.rstrip("\r") for line in raw_text.split("\n") if line.strip()]
    records, skipped = _decode_lines(lines)
    if skipped:
        logger.bind(path=str(path), skipped=skipped).warning("skipped unparseable JSONL lines")
    return records


def _decode_lines(lines: list[str]) -> tuple[list[Any], int]:
    records: list[Any] = []
    skipped = 0
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    return records, skipped
