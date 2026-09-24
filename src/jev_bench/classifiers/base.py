"""Classifier contract shared by the Jev, chat-LLM and embedding columns.

Types:
    ProgressCallback
Classes:
    Usage: tokens and cost of one request (or a sum of requests).
    EmailOutcome: answers (or an error) for one email.
    PrepareResult: one-off setup outcome: paid usage, cache, emails pending.
    RequestResult: outcomes of one request plus its usage, latency and raw body.
    Classifier: protocol implemented by every column kind.
Functions:
    usage_from_body: OpenRouter `usage` (key style agnostic) -> Usage with estimates.
    failed_result: RequestResult in which every email carries the same error.
    outcome_from_parsed: parsed answers + notes -> EmailOutcome (error when nothing parsed).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from jev_bench.catalog import ModelInfo
from jev_bench.emails import Email
from jev_bench.questions import Distribution
from jev_bench.request_plan import Sizing
from jev_bench.tokens import Budget

type ProgressCallback = Callable[[int], None]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cost_estimated: bool = False

    def plus(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost,
            cost_estimated=self.cost_estimated or other.cost_estimated,
        )


class EmailOutcome(BaseModel):
    answers: dict[str, Distribution] | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    similarities: dict[str, dict[str, float]] | None = None
    cached: bool = False
    cached_cost: float = 0.0


class PrepareResult(BaseModel):
    usage: Usage = Field(default_factory=Usage)
    cached_cost: float = 0.0
    resolved: dict[str, EmailOutcome] = Field(default_factory=dict)
    pending: tuple[Email, ...] = ()


class RequestResult(BaseModel):
    outcomes: dict[str, EmailOutcome]
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = None
    resolved_model: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class Classifier(Protocol):
    @property
    def emails_per_request(self) -> int | None: ...

    @property
    def concurrency(self) -> int: ...

    @property
    def budget(self) -> Budget: ...

    @property
    def sizing(self) -> Sizing: ...

    def input_tokens(self, email: Email) -> int: ...

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult: ...

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult: ...


def usage_from_body(usage: Mapping[str, Any] | None, pricing: ModelInfo | None) -> Usage:
    raw = usage or {}
    input_tokens = _count(raw, "input_tokens", "prompt_tokens")
    output_tokens = _count(raw, "output_tokens", "completion_tokens")
    cost = raw.get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=float(cost))
    estimate = pricing.estimate_cost(input_tokens, output_tokens) if pricing else 0.0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=estimate,
        cost_estimated=True,
    )


def _count(raw: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def failed_result(emails: Sequence[Email], message: str) -> RequestResult:
    outcomes = {email.id: EmailOutcome(error=message) for email in emails}
    return RequestResult(outcomes=outcomes, error=message)


def outcome_from_parsed(answers: dict[str, Distribution], notes: Sequence[str]) -> EmailOutcome:
    if answers:
        return EmailOutcome(answers=answers, notes=tuple(notes))
    return EmailOutcome(error="; ".join(notes) or "no answers", notes=tuple(notes))
