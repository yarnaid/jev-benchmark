"""Deterministic generation plan: per-email traits, generator model and `sent_at` from seed.

Classes:
    ResolvedTrait: a trait with concrete value ids, weights and prompt texts.
    PlanItem: what one email should look like.
Functions:
    resolve_traits: config traits -> ResolvedTrait (links question options and text).
    build_plan: seeded plan for `count` emails (stratified traits cycle evenly).
"""

import random
from collections.abc import Sequence
from datetime import datetime, timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict

from jev_bench.generation.config import GenerationConfig, Trait, TraitValue
from jev_bench.questions import QuestionSet

__all__ = [
    "PlanItem",
    "ResolvedTrait",
    "build_plan",
    "resolve_traits",
]


class ResolvedTrait(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    question: str | None
    values: dict[str, TraitValue]
    stratify: bool


class PlanItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    model: str
    sent_at: AwareDatetime
    traits: dict[str, str]


def resolve_traits(config: GenerationConfig, questions: QuestionSet) -> list[ResolvedTrait]:
    return [_resolve(trait, questions) for trait in config.traits]


def _resolve(trait: Trait, questions: QuestionSet) -> ResolvedTrait:
    if trait.question is None:
        return ResolvedTrait(
            name=trait.name, question=None, values=dict(trait.values), stratify=trait.stratify
        )
    try:
        question = questions.get(trait.question)
    except KeyError as exc:
        msg = f"trait {trait.name!r} references unknown question {trait.question!r}"
        raise ValueError(msg) from exc
    values = {option: TraitValue(prompt=text) for option, text in question.options.items()}
    return ResolvedTrait(
        name=trait.name, question=trait.question, values=values, stratify=trait.stratify
    )


def build_plan(
    config: GenerationConfig,
    questions: QuestionSet,
    *,
    count: int,
    seed: int,
    models: Sequence[str],
    now: datetime,
) -> list[PlanItem]:
    rng = random.Random(seed)
    resolved = resolve_traits(config, questions)
    columns = {trait.name: _trait_column(trait, count, rng) for trait in resolved}
    return [
        PlanItem(
            index=index + 1,
            model=models[index % len(models)],
            sent_at=_sent_at(now, config.sent_at_window_days, rng),
            traits={name: column[index] for name, column in columns.items()},
        )
        for index in range(count)
    ]


def _trait_column(trait: ResolvedTrait, count: int, rng: random.Random) -> list[str]:
    keys = list(trait.values)
    if trait.stratify:
        order = rng.sample(keys, len(keys))
        column = [order[index % len(order)] for index in range(count)]
        rng.shuffle(column)
        return column
    return rng.choices(keys, weights=[trait.values[key].weight for key in keys], k=count)


def _sent_at(now: datetime, window_days: int, rng: random.Random) -> datetime:
    minutes = rng.randrange(window_days * 24 * 60)
    return (now - timedelta(minutes=minutes)).replace(second=0, microsecond=0)
