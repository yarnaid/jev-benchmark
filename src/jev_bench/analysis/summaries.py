"""Plain-text summaries for the analyst: the generations, the runs (speed, cost, tokens) and the
questions. Generator models are never named.

Functions:
    generations_text: one line per generation (name, date, emails in the comparison).
    runs_text: one line per run under its analyst name.
    questions_text: every question with its type and options.
"""

from collections import Counter
from collections.abc import Mapping, Sequence

from jev_bench.emails import Email
from jev_bench.questions import AnyQuestion, MultiQuestion, QuestionSet
from jev_bench.store.generations import GenerationMeta
from jev_bench.store.runs import RunMeta

__all__ = [
    "generations_text",
    "questions_text",
    "runs_text",
]

_KINDS = {"choice": "single choice", "score": "ordered scale, lowest first", "noul": "yes/no"}


def generations_text(generations: Sequence[GenerationMeta], emails: Sequence[Email]) -> str:
    counts = Counter(email.generation_id for email in emails)
    return "\n".join(
        f"- {meta.name} (created {meta.created_at.date().isoformat()}): {counts[meta.id]} emails"
        for meta in generations
    )


def runs_text(metas: Sequence[RunMeta], names: Mapping[str, str]) -> str:
    return "\n".join(_run_line(meta, names[meta.id]) for meta in metas)


def _run_line(meta: RunMeta, name: str) -> str:
    duration = "—" if meta.duration_s is None else f"{meta.duration_s:.1f}"
    per_email = f"${meta.total_cost / meta.n_done:.6f}" if meta.n_done else "—"
    return (
        f"- {name}: {meta.column} · {meta.model} · {meta.mode} — {meta.status}; "
        f"{meta.n_done}/{meta.n_emails} emails answered, {meta.n_errors} errors; {duration} s; "
        f"${meta.total_cost:.6f} ({per_email} per email); "
        f"{meta.input_tokens:,} input / {meta.output_tokens:,} output tokens"
    )


def questions_text(questions: QuestionSet) -> str:
    return "\n".join(_question_block(question) for question in questions.questions)


def _question_block(question: AnyQuestion) -> str:
    lines = [f"    - {option}: {text}" for option, text in question.options.items()]
    return "\n".join([f"- {question.id} [{_kind(question)}]: {question.instructions}", *lines])


def _kind(question: AnyQuestion) -> str:
    if isinstance(question, MultiQuestion):
        return f"multi-label; a label applies when p >= {question.threshold:g} * top p"
    return _KINDS[question.type]
