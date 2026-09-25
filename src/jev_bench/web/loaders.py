"""Loads comparison and explorer inputs from the stores, mapping unknown ids to HTTP 404.

Classes:
    ComparisonInputs: the selected runs, the base question set, their emails, run raters, the
        reference rater and the human labels; `raters()` is every rater of the report.
Functions:
    load_comparison: ComparisonInputs for run ids (HTTP 400 without ids). The base question set is
        the snapshot of the most recent selected run (by created_at, then id), so the order of the
        ids never changes the report.
    load_runs: run metas for ids.
    load_emails: emails of generations.
    load_email: one email.
    run_raters: raters for run metas.
    generations_of: unique generation ids across runs, in order.
    generation_snapshots: generation_id -> question-set snapshot, for the reference rater.
"""

from collections.abc import Sequence
from typing import NamedTuple

from fastapi import HTTPException

from jev_bench.compare import Rater, human_rater, reference_rater, run_rater
from jev_bench.emails import Email
from jev_bench.questions import QuestionSet
from jev_bench.services import Services
from jev_bench.store.labels import Labels
from jev_bench.store.runs import RunMeta
from jev_bench.web.views import load_or_404

__all__ = [
    "ComparisonInputs",
    "generation_snapshots",
    "generations_of",
    "load_comparison",
    "load_email",
    "load_emails",
    "load_runs",
    "run_raters",
]


class ComparisonInputs(NamedTuple):
    metas: list[RunMeta]
    base: QuestionSet
    emails: list[Email]
    runs: list[Rater]
    reference: Rater
    labels: Labels

    def raters(self) -> list[Rater]:
        human = human_rater(self.labels, self.base)
        return [*self.runs, self.reference, *([human] if human is not None else [])]


def load_comparison(services: Services, run_ids: Sequence[str]) -> ComparisonInputs:
    metas = load_runs(services, run_ids)
    if not metas:
        raise HTTPException(status_code=400, detail="no run ids given")
    base = max(metas, key=lambda meta: (meta.created_at, meta.id)).question_set
    generation_ids = generations_of(metas)
    emails = load_emails(services, generation_ids)
    snapshots = generation_snapshots(services, generation_ids)
    reference = reference_rater(emails, base, snapshots)
    labels = services.labels.for_generations(generation_ids)
    return ComparisonInputs(metas, base, emails, run_raters(services, metas), reference, labels)


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
