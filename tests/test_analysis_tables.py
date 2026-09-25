"""Tests for jev_bench.analysis.tables."""

import json

import pytest
from tests.analysis_data import GENERATION, build_source

from jev_bench.analysis.source import email_refs, rater_names
from jev_bench.analysis.tables import disputed_emails, disputed_ids, email_tables
from jev_bench.questions import QuestionSet


def _tables(questions: QuestionSet, *, human: bool = False) -> dict[str, list[str]]:
    source = build_source(questions, human=human)
    refs = {email_id: ref for ref, email_id in email_refs(source.emails).items()}
    text = email_tables(source, rater_names(source), refs)
    blocks = [block.splitlines() for block in text.split("\n\n")]
    return {block[0]: block[1:] for block in blocks}


@pytest.mark.parametrize(
    ("title", "line", "expected"),
    [
        pytest.param("### category (choice)", 0, "| email | ref | R1 | R2 |", id="header"),
        pytest.param(
            "### category (choice)", 2, "| e001 | spam | spam 0.80 | spam 0.80 |", id="choice"
        ),
        pytest.param(
            "### category (choice)", 4, "| e003 | spam | spam 0.80 | — |", id="missing-answer"
        ),
        pytest.param(
            "### urgency (score)",
            3,
            "| e002 | now s=100 | now 0.85 s=90 | low 0.90 s=8 |",
            id="score-with-0-100",
        ),
        pytest.param("### needs_reply (noul)", 3, "| e002 | no | no 0.90 | yes 0.95 |", id="noul"),
        pytest.param(
            "### topics (multi)",
            2,
            "| e001 | billing+meeting | billing+meeting 0.50 | billing+meeting 0.50 |",
            id="multi-applied-labels",
        ),
    ],
)
def test_email_tables(multi_questions: QuestionSet, title: str, line: int, expected: str) -> None:
    assert _tables(multi_questions)[title][line] == expected


def test_human_column_only_when_labelled(multi_questions: QuestionSet) -> None:
    category = _tables(multi_questions, human=True)["### category (choice)"]
    assert category[0] == "| email | ref | human | R1 | R2 |"
    assert category[2].startswith("| e001 | spam | work |")
    assert category[3].startswith("| e002 | spam | — |")


def test_tables_never_name_emails_or_generators(multi_questions: QuestionSet) -> None:
    source = build_source(multi_questions)
    refs = {email_id: ref for ref, email_id in email_refs(source.emails).items()}
    text = email_tables(source, rater_names(source), refs)
    assert GENERATION not in text
    assert "gen/secret-model" not in text


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        pytest.param(12, [f"{GENERATION}.0002"], id="only-emails-the-runs-disagree-on"),
        pytest.param(1, [f"{GENERATION}.0002"], id="limited"),
        pytest.param(0, [], id="none"),
    ],
)
def test_disputed_ids(multi_questions: QuestionSet, limit: int, expected: list[str]) -> None:
    assert disputed_ids(build_source(multi_questions).rows, limit) == expected


def test_disputed_ids_rank_by_disagreement(multi_questions: QuestionSet) -> None:
    row = build_source(multi_questions).rows[0]
    values = {"a": 0.1, "b": 0.5, "c": None, "d": 0.0, "e": 0.3, "f": 0.5}
    rows = [
        row.model_copy(update={"id": key, "disagreement": value}) for key, value in values.items()
    ]
    assert disputed_ids(rows, 4) == ["b", "f", "e", "a"]


def test_disputed_emails_are_full_json_by_ref(multi_questions: QuestionSet) -> None:
    source = build_source(multi_questions)
    refs = {email_id: ref for ref, email_id in email_refs(source.emails).items()}
    text = disputed_emails(source, refs, [f"{GENERATION}.0002"])
    heading, body = text.splitlines()
    assert heading.startswith("#### e002 · disagreement 0.")
    state = json.loads(body)
    assert (state["ref"], state["subject"], state["body"]) == (
        "e002",
        "Subject 2",
        "Body 2 costs $5",
    )
    assert GENERATION not in text
    assert "gen/secret-model" not in text


def test_no_disputed_emails(multi_questions: QuestionSet) -> None:
    assert disputed_emails(build_source(multi_questions), {}, []).startswith("(none")
