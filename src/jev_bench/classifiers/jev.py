"""Jev column: one Decisions API call per email carrying every question.

Constants:
    DECISIONS_PATH
Classes:
    JevClassifier
Functions:
    questions_payload: QuestionSet -> Decisions `questions` object.
    parse_decisions: Decisions `answers` -> canonical distributions and per-question notes.
"""

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from jev_bench.benchmark_config import JevParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import (
    PrepareResult,
    ProgressCallback,
    RequestResult,
    outcome_from_parsed,
    usage_from_body,
)
from jev_bench.emails import Email
from jev_bench.metrics.distributions import normalize, unit_probability
from jev_bench.openrouter import JsonObject, OpenRouterClient
from jev_bench.questions import (
    AnyQuestion,
    Distribution,
    NoulQuestion,
    QuestionSet,
    ScoreQuestion,
    one_hot,
)
from jev_bench.request_plan import Sizing
from jev_bench.tokens import estimate_tokens, jev_budget

DECISIONS_PATH = "/alpha/decisions"

type Parsed = tuple[Distribution | None, str | None]


def questions_payload(questions: QuestionSet) -> dict[str, JsonObject]:
    return {question.id: _question_payload(question) for question in questions.questions}


def _question_payload(question: AnyQuestion) -> JsonObject:
    criteria: object
    if isinstance(question, NoulQuestion):
        criteria = {"true": question.options["yes"], "false": question.options["no"]}
    elif isinstance(question, ScoreQuestion):
        criteria = list(question.options.values())
    else:
        criteria = dict(question.options)
    return {"type": question.type, "instructions": question.instructions, "criteria": criteria}


def parse_decisions(
    answers: object, questions: QuestionSet
) -> tuple[dict[str, Distribution], list[str]]:
    parsed: dict[str, Distribution] = {}
    notes: list[str] = []
    answers_dict = answers if isinstance(answers, Mapping) else {}
    for question in questions.questions:
        answer = answers_dict.get(question.id)
        if not isinstance(answer, dict) or answer.get("type") != question.type:
            notes.append(f"{question.id}: missing or mistyped answer")
            continue
        distribution, note = _parse_answer(question, answer)
        if note:
            notes.append(f"{question.id}: {note}")
        if distribution is not None:
            parsed[question.id] = distribution
    return parsed, notes


def _parse_answer(question: AnyQuestion, answer: Mapping[str, Any]) -> Parsed:
    if isinstance(question, NoulQuestion):
        return _parse_noul(answer)
    if isinstance(question, ScoreQuestion):
        return _parse_score(question, answer)
    return _parse_choice(question, answer)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _parse_noul(answer: Mapping[str, Any]) -> Parsed:
    probability = unit_probability(answer.get("noul"))
    if probability is None:
        return None, "noul probability missing"
    return {"yes": probability, "no": 1.0 - probability}, None


def _parse_choice(question: AnyQuestion, answer: Mapping[str, Any]) -> Parsed:
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and (dist := normalize(probabilities, question.option_ids)):
        return dist, None
    choice = answer.get("choice")
    if isinstance(choice, str) and choice in question.options:
        return one_hot(question, choice), "probabilities missing, one-hot on choice"
    return None, "no usable choice"


def _parse_score(question: ScoreQuestion, answer: Mapping[str, Any]) -> Parsed:
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict):
        by_level = {
            level: probabilities.get(str(index)) for index, level in enumerate(question.option_ids)
        }
        if dist := normalize(by_level, question.option_ids):
            return dist, None
    score = _finite(answer.get("score"))
    if score is None:
        return None, "no usable score"
    index = min(len(question.option_ids) - 1, max(0, round(score)))
    return (
        one_hot(question, question.option_ids[index]),
        "probabilities missing, one-hot at round(score)",
    )


class JevClassifier:
    def __init__(
        self,
        *,
        client: OpenRouterClient,
        api_key: str,
        model: str,
        model_info: ModelInfo | None,
        questions: QuestionSet,
        params: JevParams,
        tokens: TokenParams,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._info = model_info
        self._questions = questions
        self._payload = questions_payload(questions)
        self._bytes_per_token = tokens.bytes_per_token
        self.emails_per_request: int | None = 1
        self.concurrency = params.concurrency
        self.budget = jev_budget(model_info, tokens)
        payload_json = json.dumps(self._payload, ensure_ascii=False)
        overhead = estimate_tokens(payload_json, tokens.bytes_per_token)
        self.sizing = Sizing(overhead=overhead, output_per_email=tokens.jev_output_reserve)

    def input_tokens(self, email: Email) -> int:
        email_json = json.dumps(email.to_state(), ensure_ascii=False)
        return estimate_tokens(email_json, self._bytes_per_token)

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        return PrepareResult(pending=tuple(emails))

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        (email,) = emails
        body = {"model": self._model, "state": email.to_state(), "questions": self._payload}
        response = await self._client.post_json(DECISIONS_PATH, body, api_key=self._api_key)
        answers, notes = parse_decisions(response.body.get("answers"), self._questions)
        return RequestResult(
            outcomes={email.id: outcome_from_parsed(answers, notes)},
            usage=usage_from_body(response.body.get("usage"), self._info),
            latency_ms=response.latency_ms,
            resolved_model=response.body.get("model"),
            raw=response.body,
        )
