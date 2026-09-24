"""Email records and the exact view benchmarked models receive.

Types:
    EmailState: the JSON object sent as classification input.
Classes:
    Party: a named mailbox.
    Email: one generated email with its generation metadata and reference answers.
Functions:
    email_id: id of the n-th email of a generation.
"""

from typing import TypedDict

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

EmailState = TypedDict(
    "EmailState",
    {"sent_at": str, "from": str, "to": list[str], "cc": list[str], "subject": str, "body": str},
)


class Party(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    address: str = Field(min_length=3)

    def formatted(self) -> str:
        return f"{self.name} <{self.address}>" if self.name.strip() else self.address


class Email(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    sent_at: AwareDatetime
    sender: Party
    to: tuple[Party, ...] = Field(min_length=1)
    cc: tuple[Party, ...] = ()
    subject: str
    body: str
    generator_model: str
    traits: dict[str, str] = Field(default_factory=dict)
    reference_answers: dict[str, str] = Field(default_factory=dict)

    @property
    def generation_id(self) -> str:
        return self.id.rsplit(".", 1)[0]

    def to_state(self) -> EmailState:
        return {
            "sent_at": self.sent_at.isoformat(),
            "from": self.sender.formatted(),
            "to": [party.formatted() for party in self.to],
            "cc": [party.formatted() for party in self.cc],
            "subject": self.subject,
            "body": self.body,
        }


def email_id(generation_id: str, index: int) -> str:
    return f"{generation_id}.{index:04d}"
