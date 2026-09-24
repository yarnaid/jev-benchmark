### Task 7: Token estimates, budgets and the request planner

**Files:**
- Create: `src/jev_bench/tokens.py`, `src/jev_bench/request_plan.py`
- Test: `tests/test_tokens.py`, `tests/test_request_plan.py`

**Interfaces:**
- Consumes: `ModelInfo` (Task 6); `TokenParams` and `EmbeddingParams` (Task 6).
- Produces:
  - `jev_bench.tokens`:
    - `Budget(total: int, output: int | None = None, item: int | None = None)` (a `NamedTuple`);
    - `estimate_tokens(text: str, bytes_per_token: float) -> int`;
    - `chat_budget(model: ModelInfo | None, tokens: TokenParams) -> Budget`;
    - `jev_budget(model, tokens) -> Budget`;
    - `embedding_budget(model, params: EmbeddingParams, tokens) -> Budget`.
  - `jev_bench.request_plan`:
    - `Sizing(overhead: int = 0, output_per_email: int = 0)`;
    - `RequestPlan(requests: list[list[int]], oversize: list[int], splits: int)`;
    - `chunk_by_count(n: int, size: int | None) -> list[list[int]]`;
    - `plan_requests(chunks, sizes, budget, sizing) -> RequestPlan`.

A request fits when all of these hold:
- `overhead + Σ sizes + output_per_email × n ≤ budget.total`;
- if `budget.output` is set, `output_per_email × n ≤ budget.output`;
- if `budget.item` is set, every single size ≤ `budget.item`.

The split rule is spec §7. Start at the minimal `k`, cut into `k` contiguous parts balanced by weight, and
increase `k` while any part fails to fit. An email that doesn't fit on its own is reported as `oversize`
and never sent.

- [ ] **Step 1: Write the failing tests**

`tests/test_tokens.py`:
```python
"""Tests for jev_bench.tokens."""

import pytest

from jev_bench.benchmark_config import EmbeddingParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.tokens import Budget, chat_budget, embedding_budget, estimate_tokens, jev_budget

_TOKENS = TokenParams()
_BIG = ModelInfo(id="m", name="m", context_length=1_000_000, max_completion_tokens=128_000)
_BARE = ModelInfo(id="m", name="m")


@pytest.mark.parametrize(
    ("text", "bytes_per_token", "expected"),
    [
        pytest.param("", 3.0, 0, id="empty"),
        pytest.param("ab", 3.0, 1, id="rounds-up"),
        pytest.param("abcdef", 3.0, 2, id="ascii"),
        pytest.param("привет", 3.0, 4, id="cyrillic-two-bytes-per-char"),
        pytest.param("abcd", 4.0, 1, id="custom-ratio"),
    ],
)
def test_estimate_tokens(text: str, bytes_per_token: float, expected: int) -> None:
    assert estimate_tokens(text, bytes_per_token) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param(_BIG, Budget(total=1_000_000, output=128_000), id="catalog-limits"),
        pytest.param(None, Budget(total=32_000, output=4_096), id="no-model"),
        pytest.param(_BARE, Budget(total=32_000, output=4_096), id="model-without-limits"),
    ],
)
def test_chat_budget(model: ModelInfo | None, expected: Budget) -> None:
    assert chat_budget(model, _TOKENS) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param(ModelInfo(id="j", name="j", context_length=32_000), Budget(total=32_000), id="catalog"),
        pytest.param(None, Budget(total=32_000), id="fallback"),
    ],
)
def test_jev_budget(model: ModelInfo | None, expected: Budget) -> None:
    assert jev_budget(model, _TOKENS) == expected


def test_embedding_budget_limits_request_and_item() -> None:
    params = EmbeddingParams(email_template="$body", option_template="$description", max_request_tokens=100_000)
    model = ModelInfo(id="e", name="e", context_length=8192)
    assert embedding_budget(model, params, _TOKENS) == Budget(total=100_000, item=8192)
```

`tests/test_request_plan.py`:
```python
"""Tests for jev_bench.request_plan."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jev_bench.request_plan import RequestPlan, Sizing, chunk_by_count, plan_requests
from jev_bench.tokens import Budget


@pytest.mark.parametrize(
    ("n", "size", "expected"),
    [
        pytest.param(5, 2, [[0, 1], [2, 3], [4]], id="by-two"),
        pytest.param(5, None, [[0, 1, 2, 3, 4]], id="all-in-one"),
        pytest.param(3, 10, [[0, 1, 2]], id="bigger-than-n"),
        pytest.param(0, 3, [], id="empty"),
        pytest.param(0, None, [], id="empty-all"),
    ],
)
def test_chunk_by_count(n: int, size: int | None, expected: list[list[int]]) -> None:
    assert chunk_by_count(n, size) == expected


def _all(n: int) -> list[list[int]]:
    return [list(range(n))]


@pytest.mark.parametrize(
    ("chunks", "sizes", "budget", "sizing", "expected"),
    [
        pytest.param(_all(3), [10, 10, 10], Budget(total=100), Sizing(), RequestPlan([[0, 1, 2]], [], 0), id="fits-as-is"),
        pytest.param(_all(3), [40, 40, 40], Budget(total=70), Sizing(), RequestPlan([[0], [1], [2]], [], 2), id="lumpy-forces-k-plus-one"),
        pytest.param(_all(3), [30, 30, 30], Budget(total=100), Sizing(overhead=50), RequestPlan([[0], [1], [2]], [], 2), id="per-request-overhead"),
        pytest.param(_all(3), [10, 500, 10], Budget(total=100), Sizing(), RequestPlan([[0, 2]], [1], 0), id="oversize-single-dropped"),
        pytest.param([[0], [1], [2]], [10, 500, 10], Budget(total=100), Sizing(), RequestPlan([[0], [2]], [1], 0), id="per-email-oversize"),
        pytest.param(_all(3), [10, 60, 10], Budget(total=1000, item=50), Sizing(), RequestPlan([[0, 2]], [1], 0), id="item-limit"),
        pytest.param(_all(1), [500], Budget(total=100), Sizing(), RequestPlan([], [0], 0), id="nothing-fits"),
    ],
)
def test_plan_requests(
    chunks: list[list[int]], sizes: list[int], budget: Budget, sizing: Sizing, expected: RequestPlan
) -> None:
    assert plan_requests(chunks, sizes, budget, sizing) == expected


def test_user_example_110k_against_100k_splits_into_two_equal_halves() -> None:
    plan = plan_requests(_all(100), [1100] * 100, Budget(total=100_000), Sizing())
    assert [len(part) for part in plan.requests] == [50, 50]
    assert [sum(1100 for _ in part) for part in plan.requests] == [55_000, 55_000]
    assert plan.splits == 1


def test_output_bound_split_respects_output_ceiling_and_order() -> None:
    plan = plan_requests(_all(10), [10] * 10, Budget(total=10_000, output=300), Sizing(output_per_email=100))
    assert len(plan.requests) == 4
    assert all(len(part) * 100 <= 300 for part in plan.requests)
    assert [i for part in plan.requests for i in part] == list(range(10))


@given(
    sizes=st.lists(st.integers(1, 50), min_size=1, max_size=30),
    total=st.integers(20, 200),
    output=st.one_of(st.none(), st.integers(5, 60)),
    output_per_email=st.integers(0, 5),
    overhead=st.integers(0, 10),
)
def test_plan_invariants(sizes: list[int], total: int, output: int | None, output_per_email: int, overhead: int) -> None:
    budget = Budget(total=total, output=output)
    sizing = Sizing(overhead=overhead, output_per_email=output_per_email)
    plan = plan_requests(_all(len(sizes)), sizes, budget, sizing)
    for part in plan.requests:
        assert overhead + sum(sizes[i] for i in part) + output_per_email * len(part) <= total
        assert output is None or output_per_email * len(part) <= output
    flattened = [i for part in plan.requests for i in part]
    assert flattened == sorted(flattened)
    assert sorted(flattened + plan.oversize) == list(range(len(sizes)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tokens.py tests/test_request_plan.py -v`
Expected: `ModuleNotFoundError` for `jev_bench.tokens` / `jev_bench.request_plan`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/tokens.py`:
```python
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
    return model.context_length if model and model.context_length else tokens.fallback_context_length


def _max_output(model: ModelInfo | None, tokens: TokenParams) -> int:
    if model and model.max_completion_tokens:
        return model.max_completion_tokens
    return tokens.fallback_max_completion_tokens


def chat_budget(model: ModelInfo | None, tokens: TokenParams) -> Budget:
    return Budget(total=_context(model, tokens), output=_max_output(model, tokens))


def jev_budget(model: ModelInfo | None, tokens: TokenParams) -> Budget:
    return Budget(total=_context(model, tokens))


def embedding_budget(model: ModelInfo | None, params: EmbeddingParams, tokens: TokenParams) -> Budget:
    return Budget(total=params.max_request_tokens, item=_context(model, tokens))
```

`src/jev_bench/request_plan.py`:
```python
"""Order-preserving request planning under a token budget (pure).

Classes:
    Sizing: per-request overhead and per-email estimated output.
    RequestPlan: planned requests (index lists), emails that can never fit, number of extra requests.
Functions:
    chunk_by_count: contiguous index chunks of at most `size` (None = everything in one chunk).
    plan_requests: split every chunk into the fewest balanced contiguous parts that fit.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

from jev_bench.tokens import Budget


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


def _split(items: list[int], sizes: Sequence[int], budget: Budget, sizing: Sizing) -> list[list[int]]:
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
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_tokens.py tests/test_request_plan.py -v`
Expected: all PASS (hypothesis runs 100 examples; the test must still finish in well under 1 s).

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/tokens.py src/jev_bench/request_plan.py tests/test_tokens.py tests/test_request_plan.py
git commit -m "feat: token budgets and order-preserving equal-split request planner

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
