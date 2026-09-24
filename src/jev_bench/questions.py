"""Typed question set: the single source of truth for what every column is asked.

Types:
    Distribution: probabilities keyed by option id.
    AnyQuestion: union of the three question classes; Question: its discriminated form.
Classes:
    ChoiceQuestion, ScoreQuestion, NoulQuestion: the three Jev primitives.
    QuestionSet: named, ordered, id-unique collection of questions.
Functions:
    load_question_set: parse and validate a question-set TOML file.
    render_questions: prompt-ready description of every question and option.
    one_hot: hard label -> distribution.
    compatible: whether two questions share type and option ids (in order).
"""

import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type Distribution = dict[str, float]

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")
_KIND_HINTS: dict[str, str] = {
    "choice": "choose exactly one option",
    "score": "ordered scale, lowest level first",
    "noul": "yes/no",
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


AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion
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


def compatible(left: AnyQuestion, right: AnyQuestion) -> bool:
    return left.type == right.type and left.option_ids == right.option_ids
