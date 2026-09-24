"""Tests for jev_bench.store.labels."""

import asyncio
from pathlib import Path

import pytest

from jev_bench.store.jsonfiles import read_json
from jev_bench.store.labels import LabelStore

_GEN = "20260924-100000-a-0001"
_E1 = f"{_GEN}.0001"
_E2 = f"{_GEN}.0002"


async def test_update_sets_overwrites_and_clears(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    result1 = await store.update(_E1, {"category": "spam", "needs_reply": "no"})
    assert result1 == {"category": "spam", "needs_reply": "no"}
    result2 = await store.update(_E1, {"category": "work", "needs_reply": None})
    assert result2 == {"category": "work"}
    assert store.for_email(_E1) == {"category": "work"}
    result3 = await store.update(_E1, {"category": None})
    assert result3 == {}
    assert read_json(tmp_path / f"{_GEN}.json") == {}


async def test_concurrent_updates_are_all_kept(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    await asyncio.gather(
        store.update(_E1, {"category": "spam"}),
        store.update(_E2, {"category": "work"}),
    )
    assert store.for_generations([_GEN]) == {
        _E1: {"category": "spam"},
        _E2: {"category": "work"},
    }


def test_reads_without_files_are_empty(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    assert store.for_email(_E1) == {}
    assert store.for_generations([_GEN, "20260924-110000-b-0002"]) == {}


@pytest.mark.parametrize(
    "email_id",
    [
        pytest.param("../../evil.0001", id="traversal"),
        pytest.param("no-index", id="malformed"),
    ],
)
async def test_invalid_email_ids_raise_key_error(tmp_path: Path, email_id: str) -> None:
    with pytest.raises(KeyError):
        await LabelStore(tmp_path).update(email_id, {"category": "spam"})
