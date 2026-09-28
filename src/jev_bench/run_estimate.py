"""Preliminary cost of a run before it starts, without an API key or a model call.

Two figures are returned:
- a token estimate: the runner's own request plan priced with the catalog. It is an upper
  bound: Jev's output reserve is pessimistic, and cached embeddings are ignored;
- a history estimate: the mean cost per email of comparable past runs (same column, model and
  mode; completed; a recorded cost above zero; embeddings use their cold cost), times the
  number of emails.
Kev columns are free: their token cost is 0 and they have no history figure.

Classes:
    RunEstimate: email, request and token counts, plus both cost figures.
Functions:
    estimate_run: RunRequest -> RunEstimate (raises RunLaunchError like launching would).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

from jev_bench.classifiers.base import Classifier
from jev_bench.emails import Email
from jev_bench.run_launcher import RunRequest, build_classifier, resolve_run
from jev_bench.runner import plan_run
from jev_bench.store.runs import RunMeta, RunMode

__all__ = [
    "RunEstimate",
    "estimate_run",
]

if TYPE_CHECKING:
    from jev_bench.services import Services


class RunEstimate(BaseModel):
    n_emails: int
    n_requests: int
    input_tokens: int
    output_tokens: int
    token_cost: float | None
    history_cost: float | None
    history_emails: int


async def estimate_run(request: RunRequest, services: Services) -> RunEstimate:
    column, model, mode, emails, info, questions, config = await resolve_run(request, services)
    classifier = build_classifier(column, model, mode, info, questions, config, services, "")
    n_requests, input_tokens, output_tokens = _planned_tokens(classifier, emails)
    per_email, history_emails = _history(services.runs.list_metas(), column.id, model, mode)
    return RunEstimate(
        n_emails=len(emails),
        n_requests=n_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_cost=info.estimate_cost(input_tokens, output_tokens) if info else None,
        history_cost=None if per_email is None else per_email * len(emails),
        history_emails=history_emails,
    )


def _planned_tokens(classifier: Classifier, emails: Sequence[Email]) -> tuple[int, int, int]:
    plan = plan_run(classifier, emails)
    sizes = [classifier.input_tokens(email) for email in emails]
    overhead, output_per_email = classifier.sizing
    inputs = sum(overhead + sum(sizes[index] for index in part) for part in plan.requests)
    outputs = sum(output_per_email * len(part) for part in plan.requests)
    return len(plan.requests), inputs, outputs


def _history(
    metas: Sequence[RunMeta], column: str, model: str, mode: RunMode
) -> tuple[float | None, int]:
    costs = [
        (_recorded_cost(meta), meta.n_done)
        for meta in metas
        if (meta.column, meta.model, meta.mode, meta.status) == (column, model, mode, "completed")
        and meta.n_done > 0
        and _recorded_cost(meta) > 0
    ]
    emails = sum(done for _, done in costs)
    return (sum(cost for cost, _ in costs) / emails if emails else None), emails


def _recorded_cost(meta: RunMeta) -> float:
    return meta.cold_cost if meta.kind == "embeddings" else meta.total_cost
