"""Tests for jev_bench.analysis.summaries."""

import pytest
from tests.analysis_data import RUN_A, build_source, generation_meta, run_meta

from jev_bench.analysis.summaries import generations_text, questions_text, runs_text
from jev_bench.questions import QuestionSet


def test_generations_text_counts_emails(multi_questions: QuestionSet) -> None:
    source = build_source(multi_questions)
    assert (
        generations_text(source.generations, source.emails)
        == "- seed (created 2026-09-25): 3 emails"
    )


@pytest.mark.parametrize(
    ("totals", "expected"),
    [
        pytest.param(
            {"n_done": 3, "total_cost": 0.003, "duration_s": 3.0, "input_tokens": 1200},
            "- R1: jev · jev/model-1 · per_email — completed; 3/3 emails answered, 0 errors;"
            " 3.0 s; $0.003000 ($0.001000 per email); 1,200 input / 0 output tokens",
            id="completed",
        ),
        pytest.param(
            {"n_done": 0, "status": "failed"},
            "- R1: jev · jev/model-1 · per_email — failed; 0/3 emails answered, 0 errors;"
            " — s; $0.000000 (— per email); 0 input / 0 output tokens",
            id="nothing-answered",
        ),
    ],
)
def test_runs_text(multi_questions: QuestionSet, totals: dict[str, object], expected: str) -> None:
    meta = run_meta(RUN_A, "jev", multi_questions, **totals)
    assert runs_text([meta], {RUN_A: "R1"}) == expected


@pytest.mark.parametrize(
    ("question_id", "header"),
    [
        pytest.param("category", "- category [single choice]: What kind of email?", id="choice"),
        pytest.param("urgency", "- urgency [ordered scale, lowest first]: How urgent?", id="score"),
        pytest.param("needs_reply", "- needs_reply [yes/no]: Needs a reply?", id="noul"),
        pytest.param(
            "topics",
            "- topics [multi-label; a label applies when p >= 0.8 * top p]: Which topics?",
            id="multi",
        ),
    ],
)
def test_questions_text(multi_questions: QuestionSet, question_id: str, header: str) -> None:
    lines = questions_text(multi_questions).splitlines()
    assert header in lines
    assert "    - spam: Junk" in lines
    assert generation_meta(multi_questions).name == "seed"
