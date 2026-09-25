"""Tests for jev_bench.store.analyses."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_bench.store.analyses import AnalysisMeta, AnalysisStore


def _meta(analysis_id: str, **update: object) -> AnalysisMeta:
    meta = AnalysisMeta(
        id=analysis_id,
        created_at=datetime(2026, 9, 25, tzinfo=UTC),
        model="anthropic/claude-sonnet-5",
        run_ids=("20260925-100000-jev-0001",),
        generation_ids=("20260925-090000-gen-0001",),
        system_prompt="S",
        user_prompt="U",
        email_refs={"e001": "20260925-090000-gen-0001.0001"},
        n_emails=1,
        n_disputed=1,
        max_output_tokens=8000,
    )
    return meta.model_copy(update=update)


def test_save_get_and_list_newest_first(tmp_path: Path) -> None:
    store = AnalysisStore(tmp_path)
    store.save(_meta("20260925-100000-analysis-0001"))
    store.save(_meta("20260925-110000-analysis-0002", result="# Done", status="completed"))
    (tmp_path / "not-an-id.json").write_text("{}", encoding="utf-8")
    (tmp_path / "20260925-120000-analysis-0003.txt").write_text("x", encoding="utf-8")
    assert store.get("20260925-110000-analysis-0002").result == "# Done"
    assert [meta.id for meta in store.list_metas()] == [
        "20260925-110000-analysis-0002",
        "20260925-100000-analysis-0001",
    ]


def test_list_on_missing_root_is_empty(tmp_path: Path) -> None:
    assert AnalysisStore(tmp_path / "missing").list_metas() == []


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("not json {{{", id="corrupt-json"),
        pytest.param('{"id": "x"}', id="invalid-meta"),
    ],
)
def test_list_skips_unreadable_meta(tmp_path: Path, content: str) -> None:
    store = AnalysisStore(tmp_path)
    store.save(_meta("20260925-100000-analysis-0001"))
    (tmp_path / "20260925-110000-analysis-0002.json").write_text(content, encoding="utf-8")
    assert [meta.id for meta in store.list_metas()] == ["20260925-100000-analysis-0001"]


@pytest.mark.parametrize(
    "analysis_id",
    [
        pytest.param("20260925-100000-missing-0001", id="missing"),
        pytest.param("../etc/passwd", id="unsafe"),
        pytest.param("", id="empty"),
    ],
)
def test_get_unknown_raises_key_error(tmp_path: Path, analysis_id: str) -> None:
    with pytest.raises(KeyError):
        AnalysisStore(tmp_path).get(analysis_id)
