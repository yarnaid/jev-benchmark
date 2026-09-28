"""Kev column: one decide call to Kev's Hugging Face Space per email, carrying every question.

Kev speaks TypeSafe's `/v1/systemone` contract, so the Jev payload and parser are reused: a
multi-label question is sent and parsed as a `choice`. The state is the email's JSON (the Space
parses a leading `{` as JSON). `prepare` checks once that the Space still serves the endpoint and
the model, so a run against a Space that no longer does fails before any email is sent. The HF
token is optional and held as a SecretStr.

Classes:
    KevClassifier
"""

import json
from collections.abc import Sequence

from pydantic import SecretStr

from jev_bench.benchmark_config import KevParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import (
    PrepareResult,
    ProgressCallback,
    RequestResult,
    outcome_from_parsed,
    usage_from_body,
)
from jev_bench.classifiers.jev import parse_decisions, questions_payload
from jev_bench.emails import Email
from jev_bench.kev_space import KevSpaceClient, KevSpaceError, SpaceDecision, model_choices
from jev_bench.questions import QuestionSet
from jev_bench.request_plan import Sizing
from jev_bench.tokens import estimate_tokens, jev_budget

__all__ = [
    "KevClassifier",
]


class KevClassifier:
    def __init__(
        self,
        *,
        client: KevSpaceClient,
        hf_token: str | None,
        model: str,
        model_info: ModelInfo | None,
        questions: QuestionSet,
        params: KevParams,
        tokens: TokenParams,
    ) -> None:
        self._client = client
        self._token = SecretStr(hf_token) if hf_token else None
        self._model = model
        self._info = model_info
        self._questions = questions
        self._params = params
        self._questions_json = json.dumps(questions_payload(questions), ensure_ascii=False)
        self._bytes_per_token = tokens.bytes_per_token
        self.emails_per_request: int | None = 1
        self.concurrency = params.concurrency
        self.budget = jev_budget(model_info, tokens)
        overhead = estimate_tokens(self._questions_json, tokens.bytes_per_token)
        self.sizing = Sizing(overhead=overhead, output_per_email=tokens.jev_output_reserve)

    def input_tokens(self, email: Email) -> int:
        return estimate_tokens(_state(email), self._bytes_per_token)

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        info = await self._client.info(self._params.space_url, token=self._secret())
        if self._model not in (model_choices(info, self._params.api_name) or ()):
            raise KevSpaceError(
                f"Kev Space {self._params.space_url} has no /{self._params.api_name} "
                f"endpoint serving {self._model!r}",
                fatal=True,
            )
        return PrepareResult(pending=tuple(emails))

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        (email,) = emails
        response = await self._client.decide(
            self._params.space_url,
            self._params.api_name,
            self._decision(email),
            token=self._secret(),
            timeout_s=self._params.timeout_s,
        )
        answers, notes = parse_decisions(response.body.get("answers"), self._questions)
        return RequestResult(
            outcomes={email.id: outcome_from_parsed(answers, notes)},
            usage=usage_from_body(response.body.get("usage"), self._info),
            latency_ms=response.latency_ms,
            resolved_model=response.body.get("model"),
            raw=response.body,
        )

    def _decision(self, email: Email) -> SpaceDecision:
        return SpaceDecision(
            state=_state(email),
            questions=self._questions_json,
            model=self._model,
            calibrated=self._params.calibrated,
        )

    def _secret(self) -> str | None:
        return self._token.get_secret_value() if self._token else None


def _state(email: Email) -> str:
    return json.dumps(email.to_state(), ensure_ascii=False)
