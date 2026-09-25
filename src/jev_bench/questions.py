"""Typed question set: the single source of truth for what every column is asked.

Every answer is a distribution over the question's options (noul: `{yes, no}`). For a multi question
the applied labels are the options with p >= threshold * max(p), so the top option always applies.

Types:
    Distribution: probabilities keyed by option id (summing to 1).
    HardAnswer: a hard label: one option id, or a list of option ids for a multi question.
    AnyQuestion: union of the four question classes; Question: its discriminated form.
Classes:
    ChoiceQuestion, ScoreQuestion, NoulQuestion: the three Jev primitives.
    MultiQuestion: multi-label question; its distribution is read with a relative threshold.
    QuestionSet: named, ordered, id-unique collection of questions.
Functions:
    load_question_set: parse and validate a question-set TOML file.
    render_questions: prompt-ready description of every question and option.
    one_hot: hard label -> distribution.
    hard_distribution: a hard answer -> its distribution, or None when it is not valid for the
        question (one-hot; for a multi question, uniform over one option id or a non-empty list of
        unique option ids).
    with_threshold: the same set with every multi question's threshold replaced (None keeps it).
    compatible: whether two questions' answers share shape and option ids (in order); choice, score
        and multi answers are all distributions, a noul only matches a noul.
"""

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AnyQuestion",
    "ChoiceQuestion",
    "Distribution",
    "HardAnswer",
    "MultiQuestion",
    "NoulQuestion",
    "Question",
    "QuestionSet",
    "ScoreQuestion",
    "compatible",
    "hard_distribution",
    "load_question_set",
    "one_hot",
    "render_questions",
    "with_threshold",
]

type Distribution = dict[str, float]
type HardAnswer = str | list[str]

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")
_KIND_HINTS: dict[str, str] = {
    "choice": "choose exactly one option",
    "score": "ordered scale, lowest level first",
    "noul": "yes/no",
    "multi": "multi-label: one or more options can apply",
}


class _Question(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    instructions: str = Field(min_length=1)
    options: dict[str, str] = Field(min_length=2, max_length=255)

    @field_validator("options")
    @classmethod
    def _valid_options(cls, options: dict[str, str]) -> dict[str, str]:
        bad = [
            key
            for key, text in options.items()
            if not _IDENTIFIER.fullmatch(key) or not text.strip()
        ]
        if bad:
            raise ValueError(f"option ids must be snake_case with a non-empty description: {bad}")
        return options

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(self.options)


class ChoiceQuestion(_Question):
    type: Literal["choice"]


class ScoreQuestion(_Question):
    type: Literal["score"]


class NoulQuestion(_Question):
    type: Literal["noul"]

    @model_validator(mode="after")
    def _yes_then_no(self) -> NoulQuestion:
        if self.option_ids != ("yes", "no"):
            raise ValueError(f"noul question {self.id!r} must define options 'yes' then 'no'")
        return self


class MultiQuestion(_Question):
    type: Literal["multi"]
    threshold: float = Field(default=0.8, gt=0.0, le=1.0)


AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion | MultiQuestion
Question = Annotated[AnyQuestion, Field(discriminator="type")]


class QuestionSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    questions: tuple[Question, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> QuestionSet:
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"duplicate question ids in {self.name!r}")
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(question.id for question in self.questions)

    def get(self, question_id: str) -> AnyQuestion:
        for question in self.questions:
            if question.id == question_id:
                return question
        raise KeyError(question_id)


def load_question_set(path: Path) -> QuestionSet:
    return QuestionSet.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def render_questions(questions: QuestionSet) -> str:
    return "\n".join(_render_question(question) for question in questions.questions)


def _render_question(question: AnyQuestion) -> str:
    header = f"- {question.id} ({_KIND_HINTS[question.type]}): {question.instructions}"
    lines = [f"    - {key}: {text}" for key, text in question.options.items()]
    return "\n".join([header, *lines])


def one_hot(question: AnyQuestion, option_id: str) -> Distribution:
    if option_id not in question.options:
        raise ValueError(f"{option_id!r} is not an option of {question.id!r}")
    return {key: float(key == option_id) for key in question.option_ids}


def hard_distribution(question: AnyQuestion, answer: object) -> Distribution | None:
    if isinstance(question, MultiQuestion) and isinstance(answer, list):
        return _uniform(question, answer)
    if isinstance(answer, str) and answer in question.options:
        return one_hot(question, answer)
    return None


def _uniform(question: MultiQuestion, answer: Sequence[object]) -> Distribution | None:
    labels = [item for item in answer if isinstance(item, str) and item in question.options]
    if not labels or len(labels) != len(answer) or len(set(labels)) != len(labels):
        return None
    share = 1.0 / len(labels)
    return {key: share if key in labels else 0.0 for key in question.option_ids}


def with_threshold(questions: QuestionSet, threshold: float | None) -> QuestionSet:
    if threshold is None:
        return questions
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    replaced = tuple(_with_threshold(question, threshold) for question in questions.questions)
    return questions.model_copy(update={"questions": replaced})


def _with_threshold(question: AnyQuestion, threshold: float) -> AnyQuestion:
    if isinstance(question, MultiQuestion):
        return question.model_copy(update={"threshold": threshold})
    return question


def compatible(left: AnyQuestion, right: AnyQuestion) -> bool:
    return _shape(left) == _shape(right) and left.option_ids == right.option_ids


def _shape(question: AnyQuestion) -> str:
    return "yes/no" if isinstance(question, NoulQuestion) else "distribution"
