"""Per-email text for the analyst: one Markdown table per question with every run's answer per
email, and the full text of the most disputed emails. Emails are named by their e001… refs.

Classes:
    TableView: what every table needs (analyst names, refs by email id, whether to show a human
        column).
Functions:
    email_tables: one table per question (ref, human when any email has human labels, R1…).
    disputed_ids: the ids of the `limit` emails with the highest disagreement index, highest first.
    disputed_emails: those emails in full (their `to_state()` JSON plus the ref).
"""

import json
from collections.abc import Mapping, Sequence

from jev_bench.analysis.source import AnalysisSource
from jev_bench.compare import EmailRow
from jev_bench.questions import AnyQuestion, Distribution, HardAnswer, ScoreQuestion

__all__ = [
    "disputed_emails",
    "disputed_ids",
    "email_tables",
]

_MISSING = "—"


def email_tables(source: AnalysisSource, names: Mapping[str, str], refs: Mapping[str, str]) -> str:
    human = any(row.human for row in source.rows)
    return "\n\n".join(
        _table(question, source, names, refs, human) for question in source.questions.questions
    )


def _table(
    question: AnyQuestion,
    source: AnalysisSource,
    names: Mapping[str, str],
    refs: Mapping[str, str],
    human: bool,
) -> str:
    header = ["email", "ref", *(["human"] if human else []), *(names[r.id] for r in source.runs)]
    rows = [_cells(row, question, source, refs[row.id], human) for row in source.rows]
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join([f"### {question.id} ({question.type})", *map(_line, lines)])


def _line(cells: Sequence[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _cells(
    row: EmailRow, question: AnyQuestion, source: AnalysisSource, ref: str, human: bool
) -> list[str]:
    reference = _with_score(_hard(row.reference.get(question.id)), row.reference_scores, question)
    labelled = [_hard(row.human.get(question.id))] if human else []
    runs = [_run_cell(row, question, rater.id, rater.answers) for rater in source.runs]
    return [ref, reference, *labelled, *runs]


def _run_cell(
    row: EmailRow,
    question: AnyQuestion,
    rater_id: str,
    answers: Mapping[str, Mapping[str, Distribution]],
) -> str:
    distribution = answers.get(row.id, {}).get(question.id)
    top = row.top.get(rater_id, {}).get(question.id)
    if distribution is None or top is None:
        return _MISSING
    text = f"{_hard(top)} {max(distribution.values()):.2f}"
    return _with_score(text, row.scores.get(rater_id, {}), question)


def _hard(answer: HardAnswer | None) -> str:
    if answer is None:
        return _MISSING
    return "+".join(answer) if isinstance(answer, list) else answer


def _with_score(text: str, scores: Mapping[str, float], question: AnyQuestion) -> str:
    score = scores.get(question.id)
    return (
        f"{text} s={score:.0f}"
        if isinstance(question, ScoreQuestion) and score is not None
        else text
    )


def disputed_ids(rows: Sequence[EmailRow], limit: int) -> list[str]:
    scored = [row for row in rows if row.disagreement is not None]
    ranked = sorted(scored, key=lambda row: -(row.disagreement or 0.0))
    return [row.id for row in ranked[:limit]]


def disputed_emails(
    source: AnalysisSource, refs: Mapping[str, str], email_ids: Sequence[str]
) -> str:
    if not email_ids:
        return "(none: no email was answered by two or more runs, or none were requested)"
    emails = {email.id: email for email in source.emails}
    rows = {row.id: row for row in source.rows}
    return "\n\n".join(
        _disputed(refs[email_id], emails[email_id].to_state(), rows[email_id].disagreement)
        for email_id in email_ids
    )


def _disputed(ref: str, state: Mapping[str, object], disagreement: float | None) -> str:
    body = json.dumps({"ref": ref, **state}, ensure_ascii=False)
    return f"#### {ref} · disagreement {disagreement or 0.0:.3f}\n{body}"
