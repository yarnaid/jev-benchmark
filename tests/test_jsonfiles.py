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
        pytest.param(
            b'{"a":1}\nnot json\n{"a":2}\n',
            [{"a": 1}, {"a": 2}],
            id="corrupt-middle-line",
        ),
    ],
)
def test_read_jsonl_skips_unparseable_lines(
    tmp_path: Path,
    raw: bytes,
    expected: list[dict[str, object]],
) -> None:
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
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(), children, max_size=3)
    ),
    max_leaves=8,
)


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(records=st.lists(_JSON_VALUES, max_size=5))
def test_jsonl_round_trip_property(tmp_path: Path, records: list[object]) -> None:
    path = tmp_path / f"{uuid.uuid4().hex}.jsonl"
    append_jsonl(path, records)
    assert read_jsonl(path) == records
