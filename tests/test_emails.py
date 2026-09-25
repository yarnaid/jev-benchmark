"""Tests for jev_bench.emails."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError
from tests.factories import EmailFactory, PartyFactory

from jev_bench.emails import Email, Party, email_id


def test_to_state_contains_exactly_the_model_visible_fields() -> None:
    email = EmailFactory(
        sender=Party(name="Ann Lee", address="ann@corp.test"),
        to=(Party(name="", address="bob@corp.test"),),
        cc=(Party(name="Cy", address="cy@x.test"),),
    )
    assert email.to_state() == {
        "sent_at": "2026-09-20T09:30:00+00:00",
        "from": "Ann Lee <ann@corp.test>",
        "to": ["bob@corp.test"],
        "cc": ["Cy <cy@x.test>"],
        "subject": email.subject,
        "body": email.body,
    }


def test_state_never_leaks_metadata() -> None:
    email = EmailFactory(
        traits={"secret_trait": "t-value"},
        reference_answers={"category": "phishing"},
        generator_model="leaky/model",
    )
    dumped = json.dumps(email.to_state())
    for leak in (email.id, "t-value", "phishing", "leaky/model", "secret_trait"):
        assert leak not in dumped


def test_generation_id() -> None:
    expected_gen_id = "20260924-100000-gen-abcd"
    assert EmailFactory(id="20260924-100000-gen-abcd.0007").generation_id == expected_gen_id


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        pytest.param(1, "g.0001", id="padded"),
        pytest.param(12345, "g.12345", id="wide"),
    ],
)
def test_email_id(index: int, expected: str) -> None:
    assert email_id("g", index) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"sent_at": datetime(2026, 9, 20, 9, 30)}, id="naive-datetime"),
        pytest.param({"to": ()}, id="no-recipients"),
    ],
)
def test_invalid_emails_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EmailFactory(**overrides)


@pytest.mark.parametrize(
    ("party", "expected"),
    [
        pytest.param(Party(name="Ann", address="ann@x.test"), "Ann <ann@x.test>", id="named"),
        pytest.param(Party(name="  ", address="ann@x.test"), "ann@x.test", id="blank-name"),
    ],
)
def test_party_formatted(party: Party, expected: str) -> None:
    assert party.formatted() == expected


def test_party_factory_is_valid() -> None:
    assert "@" in PartyFactory().address


def test_reference_answers_accept_label_lists() -> None:
    answers = {"category": ["spam", "phishing"], "urgency": "today"}
    email = EmailFactory(reference_answers=answers)
    assert Email.model_validate_json(email.model_dump_json()).reference_answers == answers
