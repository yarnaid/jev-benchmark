"""Chat-model classifier with per-email and streamed all-in-one modes.

Classes:
    LlmClassifier: implements the Classifier protocol for `per_email` and `all_in_one` modes.
"""

import asyncio
import json
import math
from collections.abc import Sequence
from string import Template
from typing import Any

from pydantic import SecretStr

from jev_bench.benchmark_config import ChatMode, LlmParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import (
    EmailOutcome,
    PrepareResult,
    ProgressCallback,
    RequestResult,
    outcome_from_parsed,
    usage_from_body,
)
from jev_bench.classifiers.llm_parse import RefCounter, parse_email_answers, split_results
from jev_bench.classifiers.llm_schema import all_in_one_schema, answers_schema, email_refs
from jev_bench.emails import Email
from jev_bench.openrouter import (
    CHAT_PATH,
    ApiResponse,
    ChatContentError,
    JsonObject,
    OpenRouterClient,
    OpenRouterError,
    TextCallback,
    chat_content,
    json_schema_format,
)
from jev_bench.questions import QuestionSet, render_questions
from jev_bench.request_plan import Sizing
from jev_bench.tokens import chat_budget, estimate_tokens

__all__ = [
    "LlmClassifier",
]

_REF_BYTES = len('{"ref":"e0000",},') + len('"e0000",')


class LlmClassifier:
    def __init__(
        self,
        *,
        client: OpenRouterClient,
        api_key: str,
        model: str,
        model_info: ModelInfo | None,
        questions: QuestionSet,
        params: LlmParams,
        tokens: TokenParams,
        mode: ChatMode,
        cache_system_prompt: bool,
    ) -> None:
        self._client = client
        self._api_key = SecretStr(api_key)
        self._model = model
        self._info = model_info
        self._questions = questions
        self._params = params
        self._mode = mode
        self._cache = cache_system_prompt
        self._bytes_per_token = tokens.bytes_per_token
        template = params.system_prompt if mode == "per_email" else params.system_prompt_all_in_one
        self._system = Template(template).substitute(questions=render_questions(questions))
        self.emails_per_request: int | None = 1 if mode == "per_email" else None
        self.concurrency = params.concurrency
        self.budget = chat_budget(model_info, tokens)
        self.sizing = Sizing(
            overhead=self._overhead(), output_per_email=params.est_output_tokens_per_email
        )

    def _overhead(self) -> int:
        if self._mode == "per_email":
            schema = answers_schema(self._questions)
        else:
            schema = all_in_one_schema(self._questions, [])
        return estimate_tokens(self._system + json.dumps(schema), self._bytes_per_token)

    def input_tokens(self, email: Email) -> int:
        size = len(json.dumps(email.to_state(), ensure_ascii=False).encode("utf-8"))
        extra = _REF_BYTES if self._mode == "all_in_one" else 0
        return math.ceil((size + extra) / self._bytes_per_token)

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        return PrepareResult(pending=tuple(emails))

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        if self._mode == "per_email":
            return await self._classify_one(emails[0])
        return await self._classify_all(emails, on_progress)

    async def _classify_one(self, email: Email) -> RequestResult:
        user = json.dumps(email.to_state(), ensure_ascii=False)
        response_format = json_schema_format("email_triage", answers_schema(self._questions))
        body = self._body(user, response_format, [email])
        response = await self._client.post_json(
            CHAT_PATH, body, api_key=self._api_key.get_secret_value()
        )
        return self._result(response, {email.id: self._single_outcome(response.body)})

    async def _classify_all(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None
    ) -> RequestResult:
        refs = email_refs(len(emails))
        states = [{"ref": ref, **email.to_state()} for ref, email in zip(refs, emails, strict=True)]
        response_format = json_schema_format(
            "email_triage_batch", all_in_one_schema(self._questions, refs)
        )
        body = self._body(json.dumps(states, ensure_ascii=False), response_format, emails)
        on_text = RefCounter(on_progress).feed if on_progress is not None else None
        response = await self._stream(body, on_text)
        return self._result(response, self._batch_outcomes(response.body, refs, emails))

    async def _stream(self, body: JsonObject, on_text: TextCallback | None) -> ApiResponse:
        try:
            async with asyncio.timeout(self._params.all_in_one_timeout_s):
                return await self._client.post_stream(
                    CHAT_PATH, body, api_key=self._api_key.get_secret_value(), on_text=on_text
                )
        except TimeoutError as exc:
            raise OpenRouterError(
                f"all-in-one request exceeded {self._params.all_in_one_timeout_s:g} s"
            ) from exc

    def _body(self, user: str, response_format: JsonObject, emails: Sequence[Email]) -> JsonObject:
        estimated_input = self.sizing.overhead + sum(self.input_tokens(email) for email in emails)
        max_output = self.budget.output or 1
        return {
            "model": self._model,
            "messages": [self._system_message(), {"role": "user", "content": user}],
            "temperature": self._params.temperature,
            "reasoning": {"enabled": self._params.reasoning_enabled},
            "response_format": response_format,
            "provider": {"require_parameters": True},
            "max_tokens": max(1, min(max_output, self.budget.total - estimated_input)),
        }

    def _system_message(self) -> JsonObject:
        if not self._cache:
            return {"role": "system", "content": self._system}
        part = {"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}
        return {"role": "system", "content": [part]}

    def _single_outcome(self, body: JsonObject) -> EmailOutcome:
        try:
            raw = json.loads(chat_content(body))
        except ChatContentError as exc:
            return EmailOutcome(error=str(exc))
        except json.JSONDecodeError:
            return EmailOutcome(error="response is not valid JSON")
        return outcome_from_parsed(*parse_email_answers(raw, self._questions))

    def _batch_outcomes(
        self, body: JsonObject, refs: Sequence[str], emails: Sequence[Email]
    ) -> dict[str, EmailOutcome]:
        try:
            found, notes = split_results(json.loads(chat_content(body)), refs)
        except ValueError as exc:
            error = f"unusable all-in-one response: {exc}"
            return {email.id: EmailOutcome(error=error) for email in emails}
        pairs = zip(refs, emails, strict=True)
        return {email.id: self._ref_outcome(found.get(ref), notes.get(ref)) for ref, email in pairs}

    def _ref_outcome(self, raw: dict[str, Any] | None, note: str | None) -> EmailOutcome:
        if raw is None:
            return EmailOutcome(error="missing from all-in-one response")
        answers, notes = parse_email_answers(raw, self._questions)
        return outcome_from_parsed(answers, [*notes, note] if note else notes)

    def _result(self, response: ApiResponse, outcomes: dict[str, EmailOutcome]) -> RequestResult:
        return RequestResult(
            outcomes=outcomes,
            usage=usage_from_body(response.body.get("usage"), self._info),
            latency_ms=response.latency_ms,
            resolved_model=response.body.get("model"),
            raw=response.body,
        )
