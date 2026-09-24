"""Factory Boy factories for domain models shared across tests.

Classes:
    PartyFactory: named mailbox.
    EmailFactory: generated email with deterministic ids and a fixed sent_at.
Functions:
    chat_body: OpenRouter chat completion response.
    sse_body: OpenRouter streaming chat completion response.
Types:
    ClientFactory: type of the make_client fixture.
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from factory.base import Factory
from factory.declarations import LazyFunction, SubFactory
from factory.declarations import Sequence as FactorySequence

from jev_bench.emails import Email, Party
from jev_bench.openrouter import OpenRouterClient


class PartyFactory(Factory[Party]):
    class Meta:
        model = Party

    name = FactorySequence(lambda n: f"Person {n}")
    address = FactorySequence(lambda n: f"person{n}@mail.test")


class EmailFactory(Factory[Email]):
    class Meta:
        model = Email

    id = FactorySequence(lambda n: f"20260924-100000-gen-abcd.{n + 1:04d}")
    sent_at = datetime(2026, 9, 20, 9, 30, tzinfo=UTC)
    sender = SubFactory(PartyFactory)
    to = LazyFunction(lambda: (PartyFactory(),))
    cc = ()
    subject = FactorySequence(lambda n: f"Subject {n}")
    body = "Hello, please confirm the meeting."
    generator_model = "google/gemini-3.8-flash"
    traits = LazyFunction(dict)
    reference_answers = LazyFunction(
        lambda: {"category": "spam", "urgency": "today", "needs_reply": "yes"}
    )


def chat_body(
    content: str, *, finish_reason: str = "stop", model: str = "test/model", cost: float = 0.001
) -> dict[str, Any]:
    return {
        "id": "gen-test",
        "model": model,
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
    }


def sse_body(
    parts: Sequence[str],
    *,
    finish_reason: str = "stop",
    model: str = "test/model",
    cost: float = 0.002,
) -> bytes:
    chunks: list[dict[str, Any]] = [
        {"id": "gen-test", "model": model, "choices": [{"delta": {"content": part}}]}
        for part in parts
    ]
    chunks.append(
        {
            "choices": [{"delta": {"content": ""}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
        }
    )
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"
    )


type ClientFactory = Callable[..., OpenRouterClient]
