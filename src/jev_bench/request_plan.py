"""Order-preserving request planning under a token budget (pure).

Classes:
    Sizing: per-request overhead and per-email estimated output.
    RequestPlan: planned requests, oversize emails, and extra request count.
Functions:
    chunk_by_count: contiguous index chunks of at most `size` (None = everything in one chunk).
    plan_requests: split every chunk into the fewest balanced contiguous parts that fit.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

from jev_bench.tokens import Budget

__all__ = [
    "RequestPlan",
    "Sizing",
    "chunk_by_count",
    "plan_requests",
]


class Sizing(NamedTuple):
    overhead: int = 0
    output_per_email: int = 0


class RequestPlan(NamedTuple):
    requests: list[list[int]]
    oversize: list[int]
    splits: int


def chunk_by_count(n: int, size: int | None) -> list[list[int]]:
    step = size or n or 1
    return [list(range(start, min(start + step, n))) for start in range(0, n, step)]


def plan_requests(
    chunks: Sequence[Sequence[int]], sizes: Sequence[int], budget: Budget, sizing: Sizing
) -> RequestPlan:
    requests: list[list[int]] = []
    oversize: list[int] = []
    produced = 0
    for chunk in chunks:
        fitting, too_big = _partition_singles(chunk, sizes, budget, sizing)
        oversize.extend(too_big)
        if fitting:
            produced += 1
            requests.extend(_split(fitting, sizes, budget, sizing))
    return RequestPlan(requests, oversize, len(requests) - produced)


def _partition_singles(
    chunk: Sequence[int], sizes: Sequence[int], budget: Budget, sizing: Sizing
) -> tuple[list[int], list[int]]:
    fitting = [index for index in chunk if _fits([index], sizes, budget, sizing)]
    kept = set(fitting)
    return fitting, [index for index in chunk if index not in kept]


def _fits(part: Sequence[int], sizes: Sequence[int], budget: Budget, sizing: Sizing) -> bool:
    inputs = sum(sizes[index] for index in part)
    outputs = sizing.output_per_email * len(part)
    if budget.item is not None and any(sizes[index] > budget.item for index in part):
        return False
    if budget.output is not None and outputs > budget.output:
        return False
    return sizing.overhead + inputs + outputs <= budget.total


def _split(
    items: list[int], sizes: Sequence[int], budget: Budget, sizing: Sizing
) -> list[list[int]]:
    weights = [sizes[index] + sizing.output_per_email for index in items]
    parts_count = _initial_parts(items, sizes, budget, sizing)
    while True:
        parts = _balanced_cut(items, weights, parts_count)
        if all(_fits(part, sizes, budget, sizing) for part in parts):
            return parts
        parts_count += 1


def _initial_parts(items: list[int], sizes: Sequence[int], budget: Budget, sizing: Sizing) -> int:
    inputs = sum(sizes[index] for index in items)
    outputs = sizing.output_per_email * len(items)
    by_total = math.ceil((inputs + outputs) / max(budget.total - sizing.overhead, 1))
    by_output = math.ceil(outputs / budget.output) if budget.output else 1
    return min(max(1, by_total, by_output), len(items))


def _balanced_cut(items: list[int], weights: list[int], parts_count: int) -> list[list[int]]:
    parts: list[list[int]] = []
    start = 0
    remaining = float(sum(weights))
    for parts_left in range(parts_count, 1, -1):
        end = _cut_point(weights, start, remaining / parts_left, len(items) - (parts_left - 1))
        parts.append(items[start:end])
        remaining -= sum(weights[start:end])
        start = end
    parts.append(items[start:])
    return parts


def _cut_point(weights: list[int], start: int, target: float, max_end: int) -> int:
    end, accumulated = start + 1, weights[start]
    while end < max_end and accumulated + weights[end] / 2 <= target:
        accumulated += weights[end]
        end += 1
    return end
