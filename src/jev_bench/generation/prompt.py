"""Prompt rendering, structured-output schema and response validation for the email generator.

Classes:
    GeneratedEmail: the email part of a generator response.
    GeneratorOutput: validated generator response (email + one hard answer per question).
Functions:
    render_prompts: (system, user) prompts for one plan item.
    generation_schema: strict JSON schema for `{email, answers}` (an array of option ids for a
        multi-label question).
    parse_generator_output: JSON text -> GeneratorOutput (every answer must be a valid hard answer,
        see questions.hard_distribution; text around the outermost JSON object, e.g. a model's
        preamble, is ignored).
    count_mismatches: question-linked traits that the generator's own answers contradict (a
        multi-label answer agrees when it contains the trait value; traits absent from
        `requested` are skipped).
"""

from collections.abc import Mapping, Sequence
from string import Template

from pydantic import BaseModel, ConfigDict, Field

from jev_bench.emails import Party
from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.plan import PlanItem, ResolvedTrait
from jev_bench.json_schema import JsonSchema, strict_object
from jev_bench.questions import (
    AnyQuestion,
    HardAnswer,
    MultiQuestion,
    QuestionSet,
    hard_distribution,
    render_questions,
)

__all__ = [
    "GeneratedEmail",
    "GeneratorOutput",
    "count_mismatches",
    "generation_schema",
    "parse_generator_output",
    "render_prompts",
]

_PARTY: JsonSchema = strict_object({"name": {"type": "string"}, "address": {"type": "string"}})


class GeneratedEmail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sender: Party
    to: tuple[Party, ...] = Field(min_length=1)
    cc: tuple[Party, ...] = ()
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class GeneratorOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: GeneratedEmail
    answers: dict[str, HardAnswer]


def render_prompts(
    config: GenerationConfig,
    traits: Sequence[ResolvedTrait],
    item: PlanItem,
    questions: QuestionSet,
) -> tuple[str, str]:
    mapping = _placeholders(traits, item, questions)
    system = Template(config.system_prompt).substitute(mapping)
    user = Template(config.user_prompt).substitute(mapping)
    return system, user


def _placeholders(
    traits: Sequence[ResolvedTrait], item: PlanItem, questions: QuestionSet
) -> dict[str, str]:
    values = {
        "questions": render_questions(questions),
        "sent_at": item.sent_at.isoformat(),
    }
    for trait in traits:
        value = item.traits[trait.name]
        values[trait.name] = value
        values[f"{trait.name}_prompt"] = trait.values[value].prompt
    return values


def generation_schema(questions: QuestionSet) -> JsonSchema:
    parties: JsonSchema = {"type": "array", "items": _PARTY}
    email_props = {
        "sender": _PARTY,
        "to": parties,
        "cc": parties,
        "subject": {"type": "string"},
        "body": {"type": "string"},
    }
    email = strict_object(email_props)
    answers_props = {question.id: _answer_schema(question) for question in questions.questions}
    answers = strict_object(answers_props)
    return strict_object({"email": email, "answers": answers})


def _answer_schema(question: AnyQuestion) -> JsonSchema:
    option: JsonSchema = {"type": "string", "enum": list(question.option_ids)}
    return {"type": "array", "items": option} if isinstance(question, MultiQuestion) else option


def parse_generator_output(content: str, questions: QuestionSet) -> GeneratorOutput:
    output = GeneratorOutput.model_validate_json(_outermost_object(content))
    invalid = [
        question.id
        for question in questions.questions
        if hard_distribution(question, output.answers.get(question.id)) is None
    ]
    if invalid:
        raise ValueError(f"generator answers missing or invalid for: {invalid}")
    ordered = {question.id: output.answers[question.id] for question in questions.questions}
    return output.model_copy(update={"answers": ordered})


def _outermost_object(content: str) -> str:
    start, end = content.find("{"), content.rfind("}")
    return content[start : end + 1] if 0 <= start < end else content


def count_mismatches(
    requested: Mapping[str, str],
    traits: Sequence[ResolvedTrait],
    answers: Mapping[str, HardAnswer],
) -> int:
    return sum(
        1
        for trait in traits
        if trait.question
        and trait.name in requested
        and not _agrees(answers.get(trait.question), requested[trait.name])
    )


def _agrees(answer: HardAnswer | None, value: str) -> bool:
    return value in answer if isinstance(answer, list) else answer == value
