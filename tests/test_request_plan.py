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
        pytest.param(
            _all(3),
            [10, 10, 10],
            Budget(total=100),
            Sizing(),
            RequestPlan([[0, 1, 2]], [], 0),
            id="fits-as-is",
        ),
        pytest.param(
            _all(3),
            [40, 40, 40],
            Budget(total=70),
            Sizing(),
            RequestPlan([[0], [1], [2]], [], 2),
            id="lumpy-forces-k-plus-one",
        ),
        pytest.param(
            _all(3),
            [30, 30, 30],
            Budget(total=100),
            Sizing(overhead=50),
            RequestPlan([[0], [1], [2]], [], 2),
            id="per-request-overhead",
        ),
        pytest.param(
            _all(3),
            [10, 500, 10],
            Budget(total=100),
            Sizing(),
            RequestPlan([[0, 2]], [1], 0),
            id="oversize-single-dropped",
        ),
        pytest.param(
            [[0], [1], [2]],
            [10, 500, 10],
            Budget(total=100),
            Sizing(),
            RequestPlan([[0], [2]], [1], 0),
            id="per-email-oversize",
        ),
        pytest.param(
            _all(3),
            [10, 60, 10],
            Budget(total=1000, item=50),
            Sizing(),
            RequestPlan([[0, 2]], [1], 0),
            id="item-limit",
        ),
        pytest.param(
            _all(1),
            [500],
            Budget(total=100),
            Sizing(),
            RequestPlan([], [0], 0),
            id="nothing-fits",
        ),
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
    plan = plan_requests(
        _all(10),
        [10] * 10,
        Budget(total=10_000, output=300),
        Sizing(output_per_email=100),
    )
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
def test_plan_invariants(
    sizes: list[int],
    total: int,
    output: int | None,
    output_per_email: int,
    overhead: int,
) -> None:
    budget = Budget(total=total, output=output)
    sizing = Sizing(overhead=overhead, output_per_email=output_per_email)
    plan = plan_requests(_all(len(sizes)), sizes, budget, sizing)
    for part in plan.requests:
        assert overhead + sum(sizes[i] for i in part) + output_per_email * len(part) <= total
        assert output is None or output_per_email * len(part) <= output
    flattened = [i for part in plan.requests for i in part]
    assert flattened == sorted(flattened)
    assert sorted(flattened + plan.oversize) == list(range(len(sizes)))
