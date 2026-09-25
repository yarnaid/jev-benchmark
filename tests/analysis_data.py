"""Shared synthetic analysis source for jev_bench.analysis tests.

Not a fixture module: each test file builds what it needs from these helpers.

Constants:
    GENERATION: the generation id of every email.
    RUN_A, RUN_B: run ids (Jev and a chat run); RUN_B has no answer for the third email.
Functions:
    run_meta: a RunMeta with totals, for the source and the summaries.
    generation_meta: a completed GenerationMeta named "seed".
    build_source: an AnalysisSource over three emails of the multi_questions set, with optional
        human labels on the first email.
"""

from datetime import UTC, datetime

from tests.factories import EmailFactory

from jev_bench.analysis.source import AnalysisSource
from jev_bench.benchmark_config import JevParams
from jev_bench.compare import compare, email_rows, reference_rater, run_rater
from jev_bench.emails import Email
from jev_bench.generation.config import GenerationConfig
from jev_bench.questions import Distribution, HardAnswer, QuestionSet
from jev_bench.store.generations import GenerationMeta
from jev_bench.store.runs import Prediction, RunMeta

GENERATION = "20260925-090000-seed-abcd"
RUN_A = "20260925-100000-jev-0001"
RUN_B = "20260925-100500-anthropic-0002"

_AGREE: dict[str, Distribution] = {
    "category": {"spam": 0.8, "personal": 0.1, "work": 0.1},
    "urgency": {"low": 0.05, "today": 0.1, "now": 0.85},
    "needs_reply": {"yes": 0.1, "no": 0.9},
    "topics": {"billing": 0.5, "meeting": 0.45, "travel": 0.05},
}
_DISAGREE: dict[str, Distribution] = {
    "category": {"spam": 0.05, "personal": 0.9, "work": 0.05},
    "urgency": {"low": 0.9, "today": 0.05, "now": 0.05},
    "needs_reply": {"yes": 0.95, "no": 0.05},
    "topics": {"billing": 0.05, "meeting": 0.05, "travel": 0.9},
}


def run_meta(run_id: str, column: str, questions: QuestionSet, **totals: object) -> RunMeta:
    meta = RunMeta(
        id=run_id,
        column=column,
        kind="decisions",
        model=f"{column}/model-1",
        generation_ids=(GENERATION,),
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=JevParams(),
        concurrency=1,
        status="completed",
        created_at=datetime(2026, 9, 25, 10, tzinfo=UTC),
        n_emails=3,
    )
    return meta.model_copy(update=totals)


def generation_meta(questions: QuestionSet) -> GenerationMeta:
    config = GenerationConfig(models=("gen/a",), system_prompt="$questions", user_prompt="Write.")
    return GenerationMeta(
        id=GENERATION,
        name="seed",
        created_at=datetime(2026, 9, 25, 9, tzinfo=UTC),
        status="completed",
        requested=3,
        done=3,
        seed=1,
        models=("gen/a",),
        question_set=questions,
        config=config,
    )


def _emails() -> list[Email]:
    reference: dict[str, HardAnswer] = {
        "category": "spam",
        "urgency": "now",
        "needs_reply": "no",
        "topics": ["billing", "meeting"],
    }
    return [
        EmailFactory(
            id=f"{GENERATION}.{index:04d}",
            subject=f"Subject {index}",
            body=f"Body {index} costs $5",
            generator_model="gen/secret-model",
            reference_answers=reference,
        )
        for index in (1, 2, 3)
    ]


def _predictions(answers: list[dict[str, Distribution] | None]) -> list[Prediction]:
    return [
        Prediction(email_id=f"{GENERATION}.{index:04d}", answers=chosen)
        if chosen is not None
        else Prediction(email_id=f"{GENERATION}.{index:04d}", error="timeout")
        for index, chosen in enumerate(answers, start=1)
    ]


def build_source(questions: QuestionSet, *, human: bool = False) -> AnalysisSource:
    emails = _emails()
    metas = [
        run_meta(RUN_A, "jev", questions, n_done=3, total_cost=0.003, duration_s=3.0),
        run_meta(RUN_B, "anthropic", questions, n_done=2, n_errors=1, total_cost=0.24),
    ]
    runs = [
        run_rater(metas[0], _predictions([_AGREE, _AGREE, _AGREE])),
        run_rater(metas[1], _predictions([_AGREE, _DISAGREE, None])),
    ]
    reference = reference_rater(emails, questions, {GENERATION: questions})
    labels = {emails[0].id: {"category": "work"}} if human else {}
    report = compare([*runs, reference], questions, runs={m.id: m for m in metas}, resamples=20)
    rows = email_rows(emails, runs, labels, questions)
    return AnalysisSource(
        report=report,
        rows=rows,
        runs=runs,
        metas=metas,
        emails=emails,
        generations=[generation_meta(questions)],
        questions=questions,
    )
