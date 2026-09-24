"""Factory Boy factories for domain models shared across tests.

Classes:
    PartyFactory: named mailbox.
    EmailFactory: generated email with deterministic ids and a fixed sent_at.
"""

from datetime import UTC, datetime

import factory

from jev_bench.emails import Email, Party


class PartyFactory(factory.Factory):
    class Meta:
        model = Party

    name = factory.Sequence(lambda n: f"Person {n}")
    address = factory.Sequence(lambda n: f"person{n}@mail.test")


class EmailFactory(factory.Factory):
    class Meta:
        model = Email

    id = factory.Sequence(lambda n: f"20260924-100000-gen-abcd.{n + 1:04d}")
    sent_at = datetime(2026, 9, 20, 9, 30, tzinfo=UTC)
    sender = factory.SubFactory(PartyFactory)
    to = factory.LazyFunction(lambda: (PartyFactory(),))
    cc = ()
    subject = factory.Sequence(lambda n: f"Subject {n}")
    body = "Hello, please confirm the meeting."
    generator_model = "google/gemini-3.8-flash"
    traits = factory.LazyFunction(dict)
    reference_answers = factory.LazyFunction(
        lambda: {"category": "spam", "urgency": "today", "needs_reply": "yes"}
    )
