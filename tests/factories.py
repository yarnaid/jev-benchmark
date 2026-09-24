"""Factory Boy factories for domain models shared across tests.

Classes:
    PartyFactory: named mailbox.
    EmailFactory: generated email with deterministic ids and a fixed sent_at.
"""

from datetime import UTC, datetime

from factory.base import Factory
from factory.declarations import LazyFunction, Sequence, SubFactory

from jev_bench.emails import Email, Party


class PartyFactory(Factory[Party]):
    class Meta:
        model = Party

    name = Sequence(lambda n: f"Person {n}")
    address = Sequence(lambda n: f"person{n}@mail.test")


class EmailFactory(Factory[Email]):
    class Meta:
        model = Email

    id = Sequence(lambda n: f"20260924-100000-gen-abcd.{n + 1:04d}")
    sent_at = datetime(2026, 9, 20, 9, 30, tzinfo=UTC)
    sender = SubFactory(PartyFactory)
    to = LazyFunction(lambda: (PartyFactory(),))
    cc = ()
    subject = Sequence(lambda n: f"Subject {n}")
    body = "Hello, please confirm the meeting."
    generator_model = "google/gemini-3.8-flash"
    traits = LazyFunction(dict)
    reference_answers = LazyFunction(
        lambda: {"category": "spam", "urgency": "today", "needs_reply": "yes"}
    )
