"""JSON schemas for chat-model answers: one email, or an all-in-one `results` array.

Functions:
    answers_schema: per-email answer object (one number per option; noul questions give P(yes)).
    all_in_one_schema: `{results: [{ref, <answers>}]}` with `ref` restricted to the refs sent.
    email_refs: positional refs e001, e002, ... (zero-padded, widening past 999).
"""

from collections.abc import Sequence

from jev_bench.json_schema import JsonSchema, strict_object
from jev_bench.questions import AnyQuestion, NoulQuestion, QuestionSet


def answers_schema(questions: QuestionSet) -> JsonSchema:
    properties = {question.id: _question_schema(question) for question in questions.questions}
    return strict_object(properties)


def all_in_one_schema(questions: QuestionSet, refs: Sequence[str]) -> JsonSchema:
    ref: JsonSchema = {"type": "string", "enum": list(refs)}
    answers = {question.id: _question_schema(question) for question in questions.questions}
    items = strict_object({"ref": ref, **answers})
    return strict_object({"results": {"type": "array", "items": items}})


def email_refs(count: int) -> list[str]:
    width = max(3, len(str(count)))
    return [f"e{index:0{width}d}" for index in range(1, count + 1)]


def _question_schema(question: AnyQuestion) -> JsonSchema:
    if isinstance(question, NoulQuestion):
        desc = f"Probability that the answer is yes: {question.instructions}"
        return {"type": "number", "description": desc}
    options: dict[str, JsonSchema] = {option: {"type": "number"} for option in question.option_ids}
    return strict_object(options, description=question.instructions)
