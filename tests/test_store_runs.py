"""Tests for jev_bench.store.runs."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_bench.benchmark_config import EmbeddingParams, JevParams, LlmParams
from jev_bench.questions import QuestionSet
from jev_bench.store.runs import META_FILE, Prediction, ResponseRecord, RunMeta, RunParams, RunStore


def _meta(run_id: str, questions: QuestionSet, params: RunParams) -> RunMeta:
    return RunMeta(
        id=run_id,
        column="anthropic",
        kind=params.kind,
        model="anthropic/claude-sonnet-5",
        generation_ids=("20260924-100000-a-0001",),
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=params,
        concurrency=8,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=2,
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(JevParams(), id="jev"),
        pytest.param(
            LlmParams(
                system_prompt="$questions",
                system_prompt_all_in_one="$questions",
            ),
            id="chat",
        ),
        pytest.param(
            EmbeddingParams(
                email_template="$body",
                option_template="$description",
            ),
            id="embeddings",
        ),
    ],
)
def test_meta_round_trip_keeps_param_kind(
    tmp_path: Path, questions: QuestionSet, params: RunParams
) -> None:
    store = RunStore(tmp_path)
    store.save(_meta("20260924-100000-anthropic-x-0001", questions, params))
    loaded = store.get("20260924-100000-anthropic-x-0001")
    assert type(loaded.params) is type(params)
    assert loaded.question_set == questions


def test_list_metas_sorted_newest_first(tmp_path: Path, questions: QuestionSet) -> None:
    store = RunStore(tmp_path)
    for run_id in ("20260924-100000-a-0001", "20260924-120000-c-0003", "20260924-110000-b-0002"):
        store.save(_meta(run_id, questions, JevParams()))
    assert [meta.id for meta in store.list_metas()] == [
        "20260924-120000-c-0003",
        "20260924-110000-b-0002",
        "20260924-100000-a-0001",
    ]


@pytest.mark.parametrize(
    "corrupt_content",
    [
        pytest.param("not json {{{", id="corrupt-meta-skipped"),
    ],
)
def test_list_metas_skips_unreadable_meta(
    tmp_path: Path, questions: QuestionSet, corrupt_content: str
) -> None:
    store = RunStore(tmp_path)
    store.save(_meta("20260924-100000-a-0001", questions, JevParams()))
    invalid_json_dir = tmp_path / "20260924-110000-bad-json-0002"
    invalid_json_dir.mkdir()
    (invalid_json_dir / META_FILE).write_text(corrupt_content, encoding="utf-8")
    schema_invalid_dir = tmp_path / "20260924-120000-bad-schema-0003"
    schema_invalid_dir.mkdir()
    (schema_invalid_dir / META_FILE).write_text(json.dumps({"id": "x"}), encoding="utf-8")
    assert [meta.id for meta in store.list_metas()] == ["20260924-100000-a-0001"]


def test_predictions_and_responses_round_trip(tmp_path: Path, questions: QuestionSet) -> None:
    store = RunStore(tmp_path)
    run_id = "20260924-100000-a-0001"
    store.save(_meta(run_id, questions, JevParams()))
    prediction = Prediction(
        email_id="g.0001",
        answers={"needs_reply": {"yes": 0.9, "no": 0.1}},
        request_index=0,
        batch_size=1,
        cost=0.001,
    )
    failure = Prediction(
        email_id="g.0002",
        error="HTTP 500: boom",
        request_index=1,
        batch_size=1,
    )
    store.append_predictions(run_id, [prediction, failure])
    store.append_response(
        run_id,
        ResponseRecord(
            request_index=0,
            email_ids=("g.0001",),
            latency_ms=12.5,
            body={"ok": 1},
        ),
    )
    assert store.predictions(run_id) == [prediction, failure]
    assert store.responses(run_id)[0].latency_ms == 12.5


@pytest.mark.parametrize(
    "run_id",
    [
        pytest.param("20260924-100000-missing-0001", id="missing"),
        pytest.param("../escape", id="traversal"),
    ],
)
def test_unknown_or_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    store = RunStore(tmp_path / "runs")
    with pytest.raises(KeyError):
        store.get(run_id)
    with pytest.raises(KeyError):
        store.predictions(run_id)
    assert not (tmp_path / "escape").exists()
