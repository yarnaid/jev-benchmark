"""Loads comparison and explorer inputs from the stores, mapping unknown ids to HTTP 404.

Functions:
    load_runs: run metas for ids.
    load_emails: emails of generations.
    load_email: one email.
    run_raters: raters for run metas.
    generations_of: unique generation ids across runs, in order.
    generation_snapshots: generation_id -> question-set snapshot, for the reference rater.
"""

from collections.abc import Sequence

from fastapi import HTTPException

from jev_bench.compare import Rater, run_rater
from jev_bench.emails import Email
from jev_bench.questions import QuestionSet
from jev_bench.services import Services
from jev_bench.store.runs import RunMeta
from jev_bench.web.views import load_or_404

__all__ = [
    "generation_snapshots",
    "generations_of",
    "load_email",
    "load_emails",
    "load_runs",
    "run_raters",
]


def load_runs(services: Services, run_ids: Sequence[str]) -> list[RunMeta]:
    return [load_or_404(services.runs.get, run_id, "run") for run_id in run_ids]


def load_emails(services: Services, generation_ids: Sequence[str]) -> list[Email]:
    try:
        return services.generations.emails_for(generation_ids)
    except KeyError as exc:
        detail = f"unknown generation {exc.args[0]!r}"
        raise HTTPException(status_code=404, detail=detail) from exc


def load_email(services: Services, email_id: str) -> Email:
    return load_or_404(services.generations.email, email_id, "email")


def run_raters(services: Services, metas: Sequence[RunMeta]) -> list[Rater]:
    return [run_rater(meta, services.runs.predictions(meta.id)) for meta in metas]


def generations_of(metas: Sequence[RunMeta]) -> list[str]:
    return list(
        dict.fromkeys(generation_id for meta in metas for generation_id in meta.generation_ids)
    )


def generation_snapshots(
    services: Services, generation_ids: Sequence[str]
) -> dict[str, QuestionSet]:
    return {
        generation_id: load_or_404(
            services.generations.get, generation_id, "generation"
        ).question_set
        for generation_id in generation_ids
    }
