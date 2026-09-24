### Task 3: JSON / JSONL persistence primitives

**Files:**
- Create: `src/jev_bench/store/__init__.py`, `src/jev_bench/store/jsonfiles.py`, `src/jev_bench/store/status.py`
- Test: `tests/test_jsonfiles.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `jev_bench.store.jsonfiles.write_json_atomic(path: Path, data: object) -> None`;
  - `read_json(path: Path) -> Any`;
  - `append_jsonl(path: Path, records: Iterable[object]) -> None`;
  - `read_jsonl(path: Path) -> list[Any]` (skips unparseable lines with a warning; missing file → `[]`);
  - `jev_bench.store.status.JobStatus = Literal["running", "completed", "cancelled", "failed",
    "interrupted"]`.

Review Focus #3 (a crash mid-write) is pinned here. A torn line must never make a file unreadable, and the
next append must start on a fresh line so it doesn't merge with the torn bytes.

- [ ] **Step 1: Write the failing tests**

`tests/test_jsonfiles.py`:
```python
"""Tests for jev_bench.store.jsonfiles."""

import uuid
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jev_bench.store.jsonfiles import append_jsonl, read_json, read_jsonl, write_json_atomic


def test_write_json_atomic_round_trip_and_no_temp_left(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "meta.json"
    write_json_atomic(target, {"name": "тест", "n": 1})
    write_json_atomic(target, {"name": "second", "n": 2})
    assert read_json(target) == {"name": "second", "n": 2}
    assert list(target.parent.glob(".*.tmp")) == []


def test_write_json_keeps_unicode_readable(tmp_path: Path) -> None:
    target = tmp_path / "meta.json"
    write_json_atomic(target, {"name": "тест"})
    assert "тест" in target.read_text(encoding="utf-8")


def test_append_and_read_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows" / "items.jsonl"
    append_jsonl(path, [{"a": 1}, {"a": 2}])
    append_jsonl(path, [{"a": 3}])
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_append_nothing_creates_no_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    append_jsonl(path, [])
    assert not path.exists()


def test_read_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "missing.jsonl") == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(b'{"a":1}\n{"a":', [{"a": 1}], id="torn-last-line"),
        pytest.param(b'{"a":1}\n\n   \n{"a":2}\n', [{"a": 1}, {"a": 2}], id="blank-lines"),
        pytest.param(b'{"a":1}\n{"b":"\xd0', [{"a": 1}], id="torn-multibyte"),
        pytest.param(b'{"a":1}\nnot json\n{"a":2}\n', [{"a": 1}, {"a": 2}], id="corrupt-middle-line"),
    ],
)
def test_read_jsonl_skips_unparseable_lines(tmp_path: Path, raw: bytes, expected: list[dict[str, object]]) -> None:
    path = tmp_path / "items.jsonl"
    path.write_bytes(raw)
    assert read_jsonl(path) == expected


def test_append_after_torn_line_starts_a_new_line(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_bytes(b'{"a":1}\n{"a":')
    append_jsonl(path, [{"a": 2}])
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


_JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(), children, max_size=3),
    max_leaves=8,
)


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(records=st.lists(_JSON_VALUES, max_size=5))
def test_jsonl_round_trip_property(tmp_path: Path, records: list[object]) -> None:
    path = tmp_path / f"{uuid.uuid4().hex}.jsonl"
    append_jsonl(path, records)
    assert read_jsonl(path) == records
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jsonfiles.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.store'`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/store/__init__.py`:
```python
"""Persistence layer: JSON/JSONL files under the data directory."""
```

`src/jev_bench/store/status.py`:
```python
"""Lifecycle status shared by runs and generations.

Types:
    JobStatus: running -> completed | cancelled | failed | interrupted.
"""

from typing import Literal

type JobStatus = Literal["running", "completed", "cancelled", "failed", "interrupted"]
```

`src/jev_bench/store/jsonfiles.py`:
```python
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
    lines = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
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
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
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
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_jsonfiles.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/store tests/test_jsonfiles.py
git commit -m "feat(store): crash-tolerant JSON and JSONL primitives

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
