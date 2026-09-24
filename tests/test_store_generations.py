"""Tests for jev_bench.store.generations."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.factories import EmailFactory

from jev_bench.generation.config import GenerationConfig
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta, GenerationStore

type StoreCall = Callable[[GenerationStore], object]

_CONFIG = GenerationConfig(models=("m",), system_prompt="$questions", user_prompt="Write.")


def _meta(generation_id: str, questions: QuestionSet) -> GenerationMeta:
    return GenerationMeta(
        id=generation_id,
        name="test",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        requested=2,
        seed=7,
        models=("m",),
        question_set=questions,
        config=_CONFIG,
    )


def test_save_get_and_list(tmp_path: Path, questions: QuestionSet) -> None:
    store = GenerationStore(tmp_path)
    store.save(_meta("20260924-100000-a-0001", questions))
    store.save(_meta("20260924-110000-b-0002", questions))
    (tmp_path / "20260924-120000-no-meta-0003").mkdir()
    (tmp_path / "not-an-id").mkdir()
    assert store.get("20260924-100000-a-0001").seed == 7
    assert [meta.id for meta in store.list_metas()] == [
        "20260924-110000-b-0002",
        "20260924-100000-a-0001",
    ]


def test_list_on_missing_root_is_empty(tmp_path: Path) -> None:
    assert GenerationStore(tmp_path / "missing").list_metas() == []


def test_emails_round_trip_and_lookup(tmp_path: Path, questions: QuestionSet) -> None:
    store = GenerationStore(tmp_path)
    generation_id = "20260924-100000-a-0001"
    store.save(_meta(generation_id, questions))
    first = EmailFactory(id=f"{generation_id}.0001")
    second = EmailFactory(id=f"{generation_id}.0002")
    store.append_email(generation_id, first)
    store.append_email(generation_id, second)
    assert store.emails(generation_id) == [first, second]
    assert store.email(f"{generation_id}.0002") == second
    assert store.emails_for([generation_id, generation_id]) == [first, second]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: s.get("20260924-100000-missing-0001"), id="get-missing"),
        pytest.param(lambda s: s.emails("20260924-100000-missing-0001"), id="emails-missing"),
        pytest.param(lambda s: s.email("20260924-100000-a-0001.0099"), id="email-missing-index"),
        pytest.param(lambda s: s.email("not-an-email-id"), id="email-malformed"),
        pytest.param(lambda s: s.get("../evil"), id="get-traversal"),
        pytest.param(lambda s: s.emails("../../evil"), id="emails-traversal"),
        pytest.param(lambda s: s.append_email("../evil", EmailFactory()), id="append-traversal"),
    ],
)
def test_unknown_or_unsafe_ids_raise_key_error(
    tmp_path: Path, questions: QuestionSet, call: StoreCall
) -> None:
    root = tmp_path / "generations"
    store = GenerationStore(root)
    store.save(_meta("20260924-100000-a-0001", questions))
    with pytest.raises(KeyError):
        call(store)
    assert not (tmp_path / "evil").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["generations"]
