"""Shared synthetic rater data for jev_bench.compare tests.

Not itself a fixture module (ruff/pyflakes treats a fixture imported by name and reused as a
test-function parameter as a redefinition), so each compare test file wraps these builders in
its own thin `@pytest.fixture`.

Constants:
    IDS, CATEGORY_A, URGENCY, REPLY: synthetic distributions for the mini question set.
Functions:
    build_answers: a full three-question answer set from a category distribution list.
    rater: build a Rater with the given id, kind and answers.
    build_run_a, build_run_b: two "run" raters over CATEGORY_A (b perturbed at index 2).
    build_reference: a reference rater over hard category/urgency/needs_reply labels.
"""

from tests.factories import EmailFactory

from jev_bench.compare.raters import Rater, RaterKind, reference_rater
from jev_bench.questions import Distribution, QuestionSet

IDS = ["g.0001", "g.0002", "g.0003", "g.0004"]
CATEGORY_A = [
    {"spam": 0.9, "personal": 0.05, "work": 0.05},
    {"spam": 0.1, "personal": 0.8, "work": 0.1},
    {"spam": 0.2, "personal": 0.2, "work": 0.6},
    {"spam": 0.5, "personal": 0.3, "work": 0.2},
]
URGENCY = [
    {"low": 0.1, "today": 0.2, "now": 0.7},
    {"low": 0.6, "today": 0.3, "now": 0.1},
    {"low": 0.2, "today": 0.6, "now": 0.2},
    {"low": 0.1, "today": 0.1, "now": 0.8},
]
REPLY = [{"yes": p, "no": 1 - p} for p in (0.2, 0.9, 0.4, 0.7)]


def build_answers(category: list[Distribution]) -> dict[str, dict[str, Distribution]]:
    return {
        email_id: {"category": category[i], "urgency": URGENCY[i], "needs_reply": REPLY[i]}
        for i, email_id in enumerate(IDS)
    }


def rater(
    rater_id: str,
    kind: RaterKind,
    answers: dict[str, dict[str, Distribution]],
    questions: QuestionSet,
) -> Rater:
    return Rater(
        id=rater_id, label=rater_id.upper(), kind=kind, questions=questions, answers=answers
    )


def build_run_a(questions: QuestionSet) -> Rater:
    return rater("a", "run", build_answers(CATEGORY_A), questions)


def build_run_b(questions: QuestionSet) -> Rater:
    category = [*CATEGORY_A[:2], {"spam": 0.7, "personal": 0.1, "work": 0.2}, CATEGORY_A[3]]
    return rater("b", "run", build_answers(category), questions)


def build_reference(questions: QuestionSet) -> Rater:
    labels = ["spam", "personal", "work", "personal"]
    emails = [
        EmailFactory(
            id=email_id,
            reference_answers={"category": label, "urgency": "now", "needs_reply": "no"},
        )
        for email_id, label in zip(IDS, labels, strict=True)
    ]
    return reference_rater(emails, questions, {"g": questions})
