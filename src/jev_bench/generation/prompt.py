"""Prompt rendering, structured-output schema and response validation for the email generator.

Classes:
    GeneratedEmail: the email part of a generator response.
    GeneratorOutput: validated generator response (email + one answer per question).
Functions:
    render_prompts: (system, user) prompts for one plan item.
    generation_schema: strict JSON schema for `{email, answers}`.
    parse_generator_output: JSON text -> GeneratorOutput (answers checked against the options).
    count_mismatches: question-linked traits that the generator's own answers contradict.
"""

from collections.abc import Mapping, Sequence
from string import Template

from pydantic import BaseModel, ConfigDict, Field

from jev_bench.emails import Party
from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.plan import PlanItem, ResolvedTrait
from jev_bench.json_schema import JsonSchema, strict_object
from jev_bench.questions import QuestionSet, render_questions

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
    answers: dict[str, str]


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
    answers_props = {
        question.id: {"type": "string", "enum": list(question.option_ids)}
        for question in questions.questions
    }
    answers = strict_object(answers_props)
    return strict_object({"email": email, "answers": answers})


def parse_generator_output(content: str, questions: QuestionSet) -> GeneratorOutput:
    output = GeneratorOutput.model_validate_json(content)
    invalid = [
        question.id
        for question in questions.questions
        if output.answers.get(question.id) not in question.options
    ]
    if invalid:
        raise ValueError(f"generator answers missing or invalid for: {invalid}")
    ordered = {question.id: output.answers[question.id] for question in questions.questions}
    return output.model_copy(update={"answers": ordered})


def count_mismatches(
    item: PlanItem, traits: Sequence[ResolvedTrait], answers: Mapping[str, str]
) -> int:
    return sum(
        1
        for trait in traits
        if trait.question and answers.get(trait.question) != item.traits[trait.name]
    )
