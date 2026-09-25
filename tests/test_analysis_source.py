"""Tests for jev_bench.analysis.source."""

from tests.analysis_data import GENERATION, RUN_A, RUN_B, build_source
from tests.factories import EmailFactory

from jev_bench.analysis.source import email_refs, rater_names
from jev_bench.questions import QuestionSet


def test_rater_names_follow_run_order(multi_questions: QuestionSet) -> None:
    names = rater_names(build_source(multi_questions))
    assert names == {RUN_A: "R1", RUN_B: "R2", "reference": "ref", "human": "human"}


def test_email_refs_are_ordered_and_padded() -> None:
    email = EmailFactory()
    emails = [email.model_copy(update={"id": f"{GENERATION}.{i:04d}"}) for i in range(1, 1002)]
    refs = email_refs(emails)
    assert list(refs)[:2] == ["e001", "e002"]
    assert refs["e1001"] == f"{GENERATION}.1001"
    assert email_refs([]) == {}
