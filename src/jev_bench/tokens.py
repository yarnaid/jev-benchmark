"""Tokenizer-free, deliberately pessimistic token estimates and per-kind request budgets.

Classes:
    Budget: request ceilings (total incl. estimated output, estimated output, per-email input).
Functions:
    estimate_tokens: ceil(utf-8 bytes / bytes_per_token).
    chat_budget, jev_budget, embedding_budget: budgets from catalog limits + config fallbacks.
"""

import math
from typing import NamedTuple

from jev_bench.benchmark_config import EmbeddingParams, TokenParams
from jev_bench.catalog import ModelInfo


class Budget(NamedTuple):
    total: int
    output: int | None = None
    item: int | None = None


def estimate_tokens(text: str, bytes_per_token: float) -> int:
    return math.ceil(len(text.encode("utf-8")) / bytes_per_token)


def _context(model: ModelInfo | None, tokens: TokenParams) -> int:
    if model and model.context_length:
        return model.context_length
    return tokens.fallback_context_length


def _max_output(model: ModelInfo | None, tokens: TokenParams) -> int:
    if model and model.max_completion_tokens:
        return model.max_completion_tokens
    return tokens.fallback_max_completion_tokens


def chat_budget(model: ModelInfo | None, tokens: TokenParams) -> Budget:
    return Budget(total=_context(model, tokens), output=_max_output(model, tokens))


def jev_budget(model: ModelInfo | None, tokens: TokenParams) -> Budget:
    return Budget(total=_context(model, tokens))


def embedding_budget(
    model: ModelInfo | None, params: EmbeddingParams, tokens: TokenParams
) -> Budget:
    return Budget(total=params.max_request_tokens, item=_context(model, tokens))
