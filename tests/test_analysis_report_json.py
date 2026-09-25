"""Tests for jev_bench.analysis.report_json."""

import json

from tests.analysis_data import RUN_A, RUN_B, build_source

from jev_bench.analysis.report_json import report_json
from jev_bench.analysis.source import rater_names
from jev_bench.compare import ComparisonReport
from jev_bench.questions import QuestionSet


def _compact(questions: QuestionSet, report: ComparisonReport | None = None) -> dict:
    source = build_source(questions)
    return json.loads(report_json(report or source.report, rater_names(source)))


def test_raters_are_renamed_and_slimmed(multi_questions: QuestionSet) -> None:
    raters = _compact(multi_questions)["raters"]
    assert raters == [
        {"id": "R1", "kind": "run", "n_items": 3},
        {"id": "R2", "kind": "run", "n_items": 2},
        {"id": "ref", "kind": "ref", "n_items": 3},
    ]


def test_questions_drop_options_and_rename_pairs(multi_questions: QuestionSet) -> None:
    category = _compact(multi_questions)["questions"][0]
    assert "options" not in category
    assert [(pair["a"], pair["b"]) for pair in category["pairs"]] == [
        ("R1", "R2"),
        ("R1", "ref"),
        ("R2", "ref"),
    ]
    assert {stats["rater"] for stats in category["raters"]} == {"R1", "R2", "ref"}


def test_floats_are_rounded_and_nulls_dropped(multi_questions: QuestionSet) -> None:
    text = report_json(build_source(multi_questions).report, {})
    assert "null" not in text
    floats = [token for token in text.replace(",", " ").replace(":", " ").split() if "." in token]
    assert all(len(token.strip("[]{}").split(".")[-1]) <= 3 for token in floats)


def test_run_ids_in_warnings_are_renamed(multi_questions: QuestionSet) -> None:
    report = build_source(multi_questions).report
    warned = report.model_copy(update={"warnings": [f"rater {RUN_B} skipped", "reference kept"]})
    names = {RUN_A: "R1", RUN_B: "R2", "reference": "ref"}
    assert json.loads(report_json(warned, names))["warnings"] == [
        "rater R2 skipped",
        "reference kept",
    ]


def test_non_finite_floats_are_dropped(multi_questions: QuestionSet) -> None:
    report = build_source(multi_questions).report
    question = report.questions[0].model_copy(update={"fleiss_kappa": float("nan")})
    broken = report.model_copy(update={"questions": [question]})
    text = report_json(broken, {})
    assert "NaN" not in text
    assert "fleiss_kappa" not in json.loads(text)["questions"][0]
