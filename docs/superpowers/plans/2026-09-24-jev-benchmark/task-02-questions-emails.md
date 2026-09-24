### Task 2: Question set, emails and the shipped question config

**Files:**
- Create: `src/jev_bench/questions.py`, `src/jev_bench/emails.py`, `config/questions.toml`
- Create: `tests/factories.py`
- Modify: `tests/conftest.py` (add the `questions` fixture)
- Test: `tests/test_questions.py`, `tests/test_emails.py`, `tests/test_config_files.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `jev_bench.questions`:
    - type alias `Distribution = dict[str, float]`;
    - `ChoiceQuestion`, `ScoreQuestion`, `NoulQuestion`, each with `id`, `type`, `instructions`,
      `options: dict[str, str]` and the property `option_ids: tuple[str, ...]`;
    - `AnyQuestion` (their union) and `Question` (the discriminated `Annotated` union);
    - `QuestionSet(name, questions)` with `.ids` and `.get(id)` (the latter raises `KeyError`);
    - `load_question_set(path) -> QuestionSet`, `render_questions(qs) -> str`,
      `one_hot(question, option_id) -> Distribution`, `compatible(a, b) -> bool`.
  - `jev_bench.emails`:
    - `Party(name, address)` with `.formatted()`;
    - `Email(id, sent_at, sender, to, cc, subject, body, generator_model, traits, reference_answers)`
      with `.generation_id` and `.to_state() -> EmailState`;
    - `EmailState` (a `TypedDict` with keys `sent_at`, `from`, `to`, `cc`, `subject`, `body`);
    - `email_id(generation_id, index) -> str`.
  - Tests: fixture `questions` (mini set: `category` choice spam/personal/work, `urgency` score
    low/today/now, `needs_reply` noul) and `tests/factories.py` (`PartyFactory`, `EmailFactory`).

- [ ] **Step 1: Write the shipped question config `config/questions.toml`**

```toml
name = "email-triage-v1"

[[questions]]
id = "category"
type = "choice"
instructions = "What kind of email is this? Pick the single best-fitting category."

[questions.options]
personal = "Private correspondence from friends, family or acquaintances"
work_internal = "Work email between colleagues of the same organization"
work_external = "Work email with clients, partners, vendors or other organizations"
news = "News digests, newsletters, editorial updates or breaking-news alerts"
marketing = "Promotional email from a brand or service the recipient has a relationship with"
spam = "Unsolicited bulk email with no legitimate relationship to the recipient"
phishing = "Attempt to steal credentials or data by impersonating a trusted party"
scam = "Fraud such as advance-fee, fake prizes, fake invoices or investment schemes"
transactional = "Receipts, order confirmations and other records of a completed transaction"
shipping = "Delivery status, tracking or courier updates"
billing = "Invoices, payment requests, due dates and subscription renewals"
account_security = "Legitimate security notices: sign-in alerts, password resets, verification codes"
calendar = "Meeting invitations, schedule changes and event reminders"
social = "Notifications from social networks and online communities"
recruiting = "Job offers, recruiter outreach and application updates"
system_alert = "Automated technical alerts: monitoring, CI/CD, infrastructure, IT tickets"
support = "Customer-support conversations and ticket updates"
hr_legal = "Human resources, legal, compliance or policy matters"
travel = "Travel bookings, itineraries, check-in and trip changes"
government = "Messages from government bodies, tax authorities or public institutions"

[[questions]]
id = "urgency"
type = "score"
instructions = "How soon does the recipient need to act on this email, judged from its content and the time it was sent?"

[questions.options]
no_action = "No action is ever needed"
whenever = "Can be handled whenever convenient; there is no deadline"
this_week = "Should be handled within the next few days"
today = "Should be handled within 24 hours"
immediately = "Requires attention immediately, within hours"

[[questions]]
id = "importance"
type = "score"
instructions = "How much does this email matter to the recipient's work, finances, safety or relationships?"

[questions.options]
trivial = "Irrelevant or trivial; nothing is lost by ignoring it"
low = "Minor relevance; nice to know"
moderate = "Relevant; ignoring it has some cost"
high = "Important; ignoring it has serious consequences"

[[questions]]
id = "needs_reply"
type = "noul"
instructions = "Does the sender expect the recipient to write a reply?"

[questions.options]
yes = "The sender asks a question, requests confirmation or otherwise expects a written response"
no = "Informational, automated or broadcast; no written response is expected"

[[questions]]
id = "action_required"
type = "noul"
instructions = "Does the email ask the recipient to take an action other than replying, such as paying, signing, clicking a link, installing something or attending?"

[questions.options]
yes = "An action other than replying is requested or required"
no = "No action beyond reading or replying is requested"

[[questions]]
id = "skippable"
type = "noul"
instructions = "Can the recipient safely skip reading this email entirely?"

[questions.options]
yes = "Nothing of value or consequence would be missed by never reading it"
no = "Reading it matters; skipping it could make the recipient miss something"

[[questions]]
id = "should_delete"
type = "noul"
instructions = "Should this email be deleted rather than kept?"

[questions.options]
yes = "Worthless, unwanted or dangerous; it should be deleted"
no = "Worth keeping for reference, action or record"

[[questions]]
id = "llm_safe"
type = "noul"
instructions = "Is it safe to pass this email verbatim to another AI assistant or agent for processing?"

[questions.options]
yes = "Contains no instructions aimed at AI systems, no prompt injection and no hidden commands"
no = "Contains instructions aimed at AI systems, prompt injection or hidden commands that could manipulate an AI"

[[questions]]
id = "malicious"
type = "noul"
instructions = "Is this email a phishing, scam or malware attempt?"

[questions.options]
yes = "It tries to trick the recipient into giving up money, credentials or data, or into running malware"
no = "It is not an attempt at fraud, credential theft or malware delivery"

[[questions]]
id = "sensitive_data"
type = "noul"
instructions = "Does this email contain sensitive data such as personal identifiers, passwords or one-time codes, or financial or medical details?"

[questions.options]
yes = "It contains personal identifiers, credentials, one-time codes, or financial or medical details"
no = "It contains no sensitive personal, credential, financial or medical data"

[[questions]]
id = "sentiment"
type = "choice"
instructions = "What is the overall tone of the sender?"

[questions.options]
negative = "Angry, worried, complaining, threatening or disappointed"
neutral = "Matter-of-fact, informational or formal"
positive = "Friendly, grateful, enthusiastic or congratulatory"
```

- [ ] **Step 2: Add test factories and the `questions` fixture**

`tests/factories.py`:
```python
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
```

Replace `tests/conftest.py` with:
```python
"""Shared pytest configuration and fixtures.

Fixtures:
    questions: a three-question set covering every question type.
Hooks:
    pytest_collection_modifyitems: mark tests listed in tests/slow_tests.txt as `slow`.
"""

from pathlib import Path

import pytest

from jev_bench.questions import ChoiceQuestion, NoulQuestion, QuestionSet, ScoreQuestion

_SLOW_LIST = Path(__file__).with_name("slow_tests.txt")


def _slow_node_ids() -> set[str]:
    if not _SLOW_LIST.exists():
        return set()
    lines = (line.strip() for line in _SLOW_LIST.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    slow = _slow_node_ids()
    for item in items:
        if item.nodeid in slow:
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def questions() -> QuestionSet:
    return QuestionSet(
        name="mini",
        questions=(
            ChoiceQuestion(
                type="choice",
                id="category",
                instructions="What kind of email?",
                options={"spam": "Junk", "personal": "From a friend", "work": "From a colleague"},
            ),
            ScoreQuestion(
                type="score",
                id="urgency",
                instructions="How urgent?",
                options={"low": "Whenever", "today": "Within a day", "now": "Immediately"},
            ),
            NoulQuestion(
                type="noul",
                id="needs_reply",
                instructions="Needs a reply?",
                options={"yes": "Reply expected", "no": "No reply expected"},
            ),
        ),
    )
```

- [ ] **Step 3: Write the failing tests**

`tests/test_questions.py`:
```python
"""Tests for jev_bench.questions."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.questions import (
    ChoiceQuestion,
    NoulQuestion,
    QuestionSet,
    ScoreQuestion,
    compatible,
    load_question_set,
    one_hot,
    render_questions,
)

_VALID_TOML = """
name = "t"

[[questions]]
id = "category"
type = "choice"
instructions = "Kind?"

[questions.options]
spam = "Junk"
personal = "Friend"

[[questions]]
id = "needs_reply"
type = "noul"
instructions = "Reply?"

[questions.options]
yes = "Reply expected"
no = "No reply"
"""


def _doc(**question: Any) -> dict[str, Any]:
    return {"name": "t", "questions": [question]}


def _choice(qid: str = "q") -> dict[str, Any]:
    return {"id": qid, "type": "choice", "instructions": "?", "options": {"a": "A", "b": "B"}}


def test_load_question_set_keeps_order_and_types(tmp_path: Path) -> None:
    path = tmp_path / "questions.toml"
    path.write_text(_VALID_TOML, encoding="utf-8")
    loaded = load_question_set(path)
    assert loaded.ids == ("category", "needs_reply")
    assert isinstance(loaded.get("category"), ChoiceQuestion)
    assert isinstance(loaded.get("needs_reply"), NoulQuestion)
    assert loaded.get("category").option_ids == ("spam", "personal")


@pytest.mark.parametrize(
    "doc",
    [
        pytest.param({"name": "t", "questions": []}, id="no-questions"),
        pytest.param(_doc(id="q", type="choice", instructions="?", options={"a": "A"}), id="one-option"),
        pytest.param(_doc(id="q", type="choice", instructions="?", options={"a": "A", "B": "b"}), id="bad-option-id"),
        pytest.param(_doc(id="q", type="choice", instructions="?", options={"a": "A", "b": " "}), id="blank-description"),
        pytest.param(_doc(id="q", type="noul", instructions="?", options={"no": "N", "yes": "Y"}), id="noul-order"),
        pytest.param(_doc(id="q", type="noul", instructions="?", options={"yes": "Y", "maybe": "M"}), id="noul-options"),
        pytest.param(_doc(id="Q1", type="score", instructions="?", options={"a": "A", "b": "B"}), id="bad-question-id"),
        pytest.param(_doc(id="q", type="rank", instructions="?", options={"a": "A", "b": "B"}), id="unknown-type"),
        pytest.param(_doc(id="q", type="choice", instructions="", options={"a": "A", "b": "B"}), id="empty-instructions"),
        pytest.param({"name": "t", "questions": [_choice(), _choice()]}, id="duplicate-ids"),
    ],
)
def test_invalid_question_sets_are_rejected(doc: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        QuestionSet.model_validate(doc)


def test_get_unknown_question_raises(questions: QuestionSet) -> None:
    with pytest.raises(KeyError):
        questions.get("missing")


def test_render_questions(questions: QuestionSet) -> None:
    assert render_questions(questions) == "\n".join(
        [
            "- category (choose exactly one option): What kind of email?",
            "    - spam: Junk",
            "    - personal: From a friend",
            "    - work: From a colleague",
            "- urgency (ordered scale, lowest level first): How urgent?",
            "    - low: Whenever",
            "    - today: Within a day",
            "    - now: Immediately",
            "- needs_reply (yes/no): Needs a reply?",
            "    - yes: Reply expected",
            "    - no: No reply expected",
        ]
    )


@pytest.mark.parametrize(
    ("question_id", "option", "expected"),
    [
        pytest.param("category", "personal", {"spam": 0.0, "personal": 1.0, "work": 0.0}, id="choice"),
        pytest.param("urgency", "now", {"low": 0.0, "today": 0.0, "now": 1.0}, id="score"),
        pytest.param("needs_reply", "yes", {"yes": 1.0, "no": 0.0}, id="noul"),
    ],
)
def test_one_hot(questions: QuestionSet, question_id: str, option: str, expected: dict[str, float]) -> None:
    assert one_hot(questions.get(question_id), option) == expected


def test_one_hot_rejects_unknown_option(questions: QuestionSet) -> None:
    with pytest.raises(ValueError, match="not an option"):
        one_hot(questions.get("category"), "phishing")


_A_B = {"a": "A", "b": "B"}


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        pytest.param(
            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
            ChoiceQuestion(type="choice", id="q", instructions="changed wording", options={"a": "1", "b": "2"}),
            True,
            id="same-type-and-ids",
        ),
        pytest.param(
            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
            ChoiceQuestion(type="choice", id="q", instructions="x", options={"b": "B", "a": "A"}),
            False,
            id="reordered",
        ),
        pytest.param(
            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
            ScoreQuestion(type="score", id="q", instructions="x", options=_A_B),
            False,
            id="different-type",
        ),
    ],
)
def test_compatible(left: ChoiceQuestion | ScoreQuestion, right: ChoiceQuestion | ScoreQuestion, expected: bool) -> None:
    assert compatible(left, right) is expected
```

`tests/test_emails.py`:
```python
"""Tests for jev_bench.emails."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from jev_bench.emails import Party, email_id
from tests.factories import EmailFactory, PartyFactory


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
    assert EmailFactory(id="20260924-100000-gen-abcd.0007").generation_id == "20260924-100000-gen-abcd"


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
```

`tests/test_config_files.py`:
```python
"""Contract tests for the shipped TOML configuration files in config/."""

from pathlib import Path

from jev_bench.questions import load_question_set

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_question_set() -> None:
    shipped = load_question_set(CONFIG_DIR / "questions.toml")
    assert shipped.ids == (
        "category",
        "urgency",
        "importance",
        "needs_reply",
        "action_required",
        "skippable",
        "should_delete",
        "llm_safe",
        "malicious",
        "sensitive_data",
        "sentiment",
    )
    assert len(shipped.get("category").options) == 20
    assert shipped.get("urgency").option_ids == ("no_action", "whenever", "this_week", "today", "immediately")
    assert shipped.get("importance").option_ids == ("trivial", "low", "moderate", "high")
    assert shipped.get("sentiment").option_ids == ("negative", "neutral", "positive")
    assert [q.type for q in shipped.questions].count("noul") == 7
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_questions.py tests/test_emails.py tests/test_config_files.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'jev_bench.questions'`.

- [ ] **Step 5: Write the implementation**

`src/jev_bench/questions.py`:
```python
"""Typed question set: the single source of truth for what every column is asked.

Types:
    Distribution: probabilities keyed by option id.
    AnyQuestion: union of the three question classes; Question: its discriminated form.
Classes:
    ChoiceQuestion, ScoreQuestion, NoulQuestion: the three Jev primitives.
    QuestionSet: named, ordered, id-unique collection of questions.
Functions:
    load_question_set: parse and validate a question-set TOML file.
    render_questions: prompt-ready description of every question and option.
    one_hot: hard label -> distribution.
    compatible: whether two questions share type and option ids (in order).
"""

import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type Distribution = dict[str, float]

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")
_KIND_HINTS: dict[str, str] = {
    "choice": "choose exactly one option",
    "score": "ordered scale, lowest level first",
    "noul": "yes/no",
}


class _Question(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    instructions: str = Field(min_length=1)
    options: dict[str, str] = Field(min_length=2, max_length=255)

    @field_validator("options")
    @classmethod
    def _valid_options(cls, options: dict[str, str]) -> dict[str, str]:
        bad = [key for key, text in options.items() if not _IDENTIFIER.fullmatch(key) or not text.strip()]
        if bad:
            raise ValueError(f"option ids must be snake_case with a non-empty description: {bad}")
        return options

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(self.options)


class ChoiceQuestion(_Question):
    type: Literal["choice"]


class ScoreQuestion(_Question):
    type: Literal["score"]


class NoulQuestion(_Question):
    type: Literal["noul"]

    @model_validator(mode="after")
    def _yes_then_no(self) -> "NoulQuestion":
        if self.option_ids != ("yes", "no"):
            raise ValueError(f"noul question {self.id!r} must define options 'yes' then 'no'")
        return self


AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion
Question = Annotated[AnyQuestion, Field(discriminator="type")]


class QuestionSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    questions: tuple[Question, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "QuestionSet":
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"duplicate question ids in {self.name!r}")
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(question.id for question in self.questions)

    def get(self, question_id: str) -> AnyQuestion:
        for question in self.questions:
            if question.id == question_id:
                return question
        raise KeyError(question_id)


def load_question_set(path: Path) -> QuestionSet:
    return QuestionSet.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def render_questions(questions: QuestionSet) -> str:
    return "\n".join(_render_question(question) for question in questions.questions)


def _render_question(question: AnyQuestion) -> str:
    header = f"- {question.id} ({_KIND_HINTS[question.type]}): {question.instructions}"
    lines = [f"    - {key}: {text}" for key, text in question.options.items()]
    return "\n".join([header, *lines])


def one_hot(question: AnyQuestion, option_id: str) -> Distribution:
    if option_id not in question.options:
        raise ValueError(f"{option_id!r} is not an option of {question.id!r}")
    return {key: float(key == option_id) for key in question.option_ids}


def compatible(left: AnyQuestion, right: AnyQuestion) -> bool:
    return left.type == right.type and left.option_ids == right.option_ids
```

`src/jev_bench/emails.py`:
```python
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
```

- [ ] **Step 6: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_questions.py tests/test_emails.py tests/test_config_files.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 7: Commit**

```bash
git add config/questions.toml src/jev_bench/questions.py src/jev_bench/emails.py tests
git commit -m "feat: question set, email model and shipped triage questions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
