### Task 11: Generation config model and the generation, run and label stores

**Files:**
- Create: `src/jev_bench/generation/__init__.py`, `src/jev_bench/generation/config.py`
- Create: `src/jev_bench/store/generations.py`, `src/jev_bench/store/runs.py`, `src/jev_bench/store/labels.py`
- Test: `tests/test_generation_config.py`, `tests/test_store_generations.py`, `tests/test_store_runs.py`,
  `tests/test_store_labels.py`

**Interfaces:**
- Consumes:
  - Task 1: `is_safe_id`, `split_email_id`;
  - Task 2: `Email`, `QuestionSet`, `Distribution`;
  - Task 3: `write_json_atomic`, `read_json`, `append_jsonl`, `read_jsonl`, `JobStatus`;
  - Task 6: `check_template`, `ColumnKind`, `JevParams`, `LlmParams`, `EmbeddingParams`.
- Produces:
  - `jev_bench.generation.config`:
    - `TraitValue(weight, prompt)`;
    - `Trait(name, question, values, stratify)`;
    - `GenerationConfig(models, temperature, concurrency, sent_at_window_days, system_prompt,
      user_prompt, traits)`;
    - `load_generation_config(path)`.
  - `jev_bench.store.generations`:
    - `GenerationMeta(id, name, created_at, status, requested, done, errors, seed, models,
      question_set, config, total_cost, duration_s, trait_mismatches, error, finished_at)`;
    - `GenerationStore(root)` with `save(meta)`, `get(id)`, `list_metas()`, `append_email(id, email)`,
      `emails(id)`, `emails_for(ids)` and `email(email_id)`. Unknown or unsafe ids raise `KeyError`.
  - `jev_bench.store.runs`:
    - `RunMode = Literal["per_email", "all_in_one", "batched"]`;
    - `RunParams` (discriminated on `kind`);
    - `RunMeta` (fields per spec §4, plus `error`);
    - `Prediction`;
    - `ResponseRecord(request_index, email_ids, latency_ms, error, body)`;
    - `RunStore(root)` with `save`, `get`, `list_metas`, `append_predictions`, `predictions`,
      `append_response` and `responses`.
  - `jev_bench.store.labels.LabelStore(root)` with `for_generations(ids)`, `for_email(email_id)` and
    `await update(email_id, changes: Mapping[str, str | None]) -> dict[str, str]`.

Method names avoid `list` because a method named `list` would shadow the builtin inside the class body.

Review Focus #2 (path traversal) and #5 (concurrent label edits) are pinned here.

- [ ] **Step 1: Write the failing tests**

`tests/test_generation_config.py`:
```python
"""Tests for jev_bench.generation.config."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.generation.config import GenerationConfig, load_generation_config


def _doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "models": ["google/gemini-3.8-flash"],
        "system_prompt": "Write emails.\n$questions",
        "user_prompt": "Category $category: $category_prompt. Length: $length_prompt. Sent $sent_at.",
        "traits": [
            {"name": "category", "question": "category", "stratify": True},
            {"name": "length", "values": {"short": {"weight": 3, "prompt": "Short."}, "long": {"prompt": "Long."}}},
        ],
    }
    doc.update(overrides)
    return doc


def test_valid_config() -> None:
    config = GenerationConfig.model_validate(_doc())
    assert config.temperature == 1.0
    assert config.concurrency == 8
    assert config.sent_at_window_days == 30
    assert config.traits[0].question == "category"
    assert config.traits[1].values["long"].weight == 1.0


@pytest.mark.parametrize(
    "doc",
    [
        pytest.param(_doc(models=[]), id="no-models"),
        pytest.param(_doc(user_prompt="$missing"), id="unknown-placeholder"),
        pytest.param(_doc(traits=[{"name": "x", "question": "q", "values": {"a": {"prompt": "A"}}}]), id="question-and-values"),
        pytest.param(_doc(traits=[{"name": "x"}]), id="neither-source"),
        pytest.param(_doc(traits=[{"name": "x", "question": "q"}, {"name": "x", "question": "q"}], user_prompt="$x"), id="duplicate-trait"),
        pytest.param(_doc(traits=[{"name": "x", "values": {"a": {"weight": 0, "prompt": "A"}}}], user_prompt="$x"), id="zero-weight"),
        pytest.param(_doc(traits=[{"name": "Bad-Name", "question": "q"}], user_prompt="x"), id="bad-trait-name"),
    ],
)
def test_invalid_configs(doc: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(doc)


def test_load_generation_config(tmp_path: Path) -> None:
    path = tmp_path / "generation.toml"
    path.write_text(
        'models = ["m"]\nsystem_prompt = "$questions"\nuser_prompt = "$tone_prompt"\n'
        '[[traits]]\nname = "tone"\n[traits.values.calm]\nprompt = "Calm."\n',
        encoding="utf-8",
    )
    assert load_generation_config(path).traits[0].values["calm"].prompt == "Calm."
```

`tests/test_store_generations.py`:
```python
"""Tests for jev_bench.store.generations."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_bench.generation.config import GenerationConfig
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta, GenerationStore
from tests.factories import EmailFactory

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
    assert [meta.id for meta in store.list_metas()] == ["20260924-110000-b-0002", "20260924-100000-a-0001"]


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
def test_unknown_or_unsafe_ids_raise_key_error(tmp_path: Path, questions: QuestionSet, call: StoreCall) -> None:
    root = tmp_path / "generations"
    store = GenerationStore(root)
    store.save(_meta("20260924-100000-a-0001", questions))
    with pytest.raises(KeyError):
        call(store)
    assert not (tmp_path / "evil").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["generations"]
```

`tests/test_store_runs.py`:
```python
"""Tests for jev_bench.store.runs."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_bench.benchmark_config import EmbeddingParams, JevParams, LlmParams
from jev_bench.questions import QuestionSet
from jev_bench.store.runs import Prediction, ResponseRecord, RunMeta, RunParams, RunStore


def _meta(run_id: str, questions: QuestionSet, params: RunParams) -> RunMeta:
    return RunMeta(
        id=run_id,
        column="anthropic",
        kind=params.kind,
        model="anthropic/claude-sonnet-5",
        generation_ids=("20260924-100000-a-0001",),
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=params,
        concurrency=8,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=2,
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(JevParams(), id="jev"),
        pytest.param(LlmParams(system_prompt="$questions", system_prompt_all_in_one="$questions"), id="chat"),
        pytest.param(EmbeddingParams(email_template="$body", option_template="$description"), id="embeddings"),
    ],
)
def test_meta_round_trip_keeps_param_kind(tmp_path: Path, questions: QuestionSet, params: RunParams) -> None:
    store = RunStore(tmp_path)
    store.save(_meta("20260924-100000-anthropic-x-0001", questions, params))
    loaded = store.get("20260924-100000-anthropic-x-0001")
    assert type(loaded.params) is type(params)
    assert loaded.question_set == questions


def test_list_metas_sorted_newest_first(tmp_path: Path, questions: QuestionSet) -> None:
    store = RunStore(tmp_path)
    for run_id in ("20260924-100000-a-0001", "20260924-120000-c-0003", "20260924-110000-b-0002"):
        store.save(_meta(run_id, questions, JevParams()))
    assert [meta.id for meta in store.list_metas()] == [
        "20260924-120000-c-0003",
        "20260924-110000-b-0002",
        "20260924-100000-a-0001",
    ]


def test_predictions_and_responses_round_trip(tmp_path: Path, questions: QuestionSet) -> None:
    store = RunStore(tmp_path)
    run_id = "20260924-100000-a-0001"
    store.save(_meta(run_id, questions, JevParams()))
    prediction = Prediction(email_id="g.0001", answers={"needs_reply": {"yes": 0.9, "no": 0.1}}, request_index=0, batch_size=1, cost=0.001)
    failure = Prediction(email_id="g.0002", error="HTTP 500: boom", request_index=1, batch_size=1)
    store.append_predictions(run_id, [prediction, failure])
    store.append_response(run_id, ResponseRecord(request_index=0, email_ids=("g.0001",), latency_ms=12.5, body={"ok": 1}))
    assert store.predictions(run_id) == [prediction, failure]
    assert store.responses(run_id)[0].latency_ms == 12.5


@pytest.mark.parametrize(
    "run_id",
    [
        pytest.param("20260924-100000-missing-0001", id="missing"),
        pytest.param("../escape", id="traversal"),
    ],
)
def test_unknown_or_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    store = RunStore(tmp_path / "runs")
    with pytest.raises(KeyError):
        store.get(run_id)
    with pytest.raises(KeyError):
        store.predictions(run_id)
    assert not (tmp_path / "escape").exists()
```

`tests/test_store_labels.py`:
```python
"""Tests for jev_bench.store.labels."""

import asyncio
from pathlib import Path

import pytest

from jev_bench.store.jsonfiles import read_json
from jev_bench.store.labels import LabelStore

_GEN = "20260924-100000-a-0001"
_E1 = f"{_GEN}.0001"
_E2 = f"{_GEN}.0002"


async def test_update_sets_overwrites_and_clears(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    assert await store.update(_E1, {"category": "spam", "needs_reply": "no"}) == {"category": "spam", "needs_reply": "no"}
    assert await store.update(_E1, {"category": "work", "needs_reply": None}) == {"category": "work"}
    assert store.for_email(_E1) == {"category": "work"}
    assert await store.update(_E1, {"category": None}) == {}
    assert read_json(tmp_path / f"{_GEN}.json") == {}


async def test_concurrent_updates_are_all_kept(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    await asyncio.gather(store.update(_E1, {"category": "spam"}), store.update(_E2, {"category": "work"}))
    assert store.for_generations([_GEN]) == {_E1: {"category": "spam"}, _E2: {"category": "work"}}


def test_reads_without_files_are_empty(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    assert store.for_email(_E1) == {}
    assert store.for_generations([_GEN, "20260924-110000-b-0002"]) == {}


@pytest.mark.parametrize(
    "email_id",
    [
        pytest.param("../../evil.0001", id="traversal"),
        pytest.param("no-index", id="malformed"),
    ],
)
async def test_invalid_email_ids_raise_key_error(tmp_path: Path, email_id: str) -> None:
    with pytest.raises(KeyError):
        await LabelStore(tmp_path).update(email_id, {"category": "spam"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generation_config.py tests/test_store_generations.py tests/test_store_runs.py tests/test_store_labels.py -v`
Expected: `ModuleNotFoundError` for `jev_bench.generation` / `jev_bench.store.generations` / `runs` / `labels`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/generation/__init__.py`:
```python
"""Synthetic email generation: config, deterministic plan, prompts, generator job and launcher."""
```

`src/jev_bench/generation/config.py`:
```python
"""Generation configuration (`config/generation.toml`): generator mix, prompt templates, dataset traits.

Classes:
    TraitValue: one weighted trait value with its prompt text.
    Trait: a dataset knob, either linked to a question's options or with explicit values.
    GenerationConfig: the whole file (templates validated against the declared traits).
Functions:
    load_generation_config: parse and validate the TOML file.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jev_bench.templates import check_template


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TraitValue(_Frozen):
    weight: float = Field(default=1.0, gt=0)
    prompt: str = Field(min_length=1)


class Trait(_Frozen):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    question: str | None = None
    values: dict[str, TraitValue] = Field(default_factory=dict)
    stratify: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "Trait":
        if (self.question is None) == (not self.values):
            raise ValueError(f"trait {self.name!r} needs either `question` or `values`, not both or neither")
        return self


class GenerationConfig(_Frozen):
    models: tuple[str, ...] = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0)
    concurrency: int = Field(default=8, ge=1)
    sent_at_window_days: int = Field(default=30, ge=1)
    system_prompt: str
    user_prompt: str
    traits: tuple[Trait, ...] = ()

    @model_validator(mode="after")
    def _traits_and_templates(self) -> "GenerationConfig":
        names = [trait.name for trait in self.traits]
        if len(names) != len(set(names)):
            raise ValueError("duplicate trait names")
        allowed = {"questions", "sent_at", *names, *(f"{name}_prompt" for name in names)}
        check_template(self.system_prompt, allowed)
        check_template(self.user_prompt, allowed)
        return self


def load_generation_config(path: Path) -> GenerationConfig:
    return GenerationConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
```

`src/jev_bench/store/generations.py`:
```python
"""Generation persistence: `generation.json` meta plus append-only `emails.jsonl` per generation.

Classes:
    GenerationMeta: persisted description, config snapshot and totals of one generation.
    GenerationStore: save / get / list generations and read or append their emails.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import AwareDatetime, BaseModel

from jev_bench.emails import Email
from jev_bench.generation.config import GenerationConfig
from jev_bench.ids import is_safe_id, split_email_id
from jev_bench.questions import QuestionSet
from jev_bench.store.jsonfiles import append_jsonl, read_json, read_jsonl, write_json_atomic
from jev_bench.store.status import JobStatus

META_FILE = "generation.json"
EMAILS_FILE = "emails.jsonl"


class GenerationMeta(BaseModel):
    id: str
    name: str
    created_at: AwareDatetime
    status: JobStatus = "running"
    requested: int
    done: int = 0
    errors: int = 0
    seed: int
    models: tuple[str, ...]
    question_set: QuestionSet
    config: GenerationConfig
    total_cost: float = 0.0
    duration_s: float | None = None
    trait_mismatches: int = 0
    error: str | None = None
    finished_at: AwareDatetime | None = None


class GenerationStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, meta: GenerationMeta) -> None:
        write_json_atomic(self._dir(meta.id) / META_FILE, meta.model_dump(mode="json"))

    def get(self, generation_id: str) -> GenerationMeta:
        path = self._dir(generation_id) / META_FILE
        if not path.exists():
            raise KeyError(generation_id)
        return GenerationMeta.model_validate(read_json(path))

    def list_metas(self) -> list[GenerationMeta]:
        if not self._root.exists():
            return []
        ids = [path.name for path in self._root.iterdir() if is_safe_id(path.name) and (path / META_FILE).exists()]
        return sorted((self.get(generation_id) for generation_id in ids), key=lambda meta: meta.id, reverse=True)

    def append_email(self, generation_id: str, email: Email) -> None:
        append_jsonl(self._dir(generation_id) / EMAILS_FILE, [email.model_dump(mode="json")])

    def emails(self, generation_id: str) -> list[Email]:
        directory = self._dir(generation_id)
        if not directory.exists():
            raise KeyError(generation_id)
        return [Email.model_validate(record) for record in read_jsonl(directory / EMAILS_FILE)]

    def emails_for(self, generation_ids: Iterable[str]) -> list[Email]:
        return [email for generation_id in dict.fromkeys(generation_ids) for email in self.emails(generation_id)]

    def email(self, email_id: str) -> Email:
        generation_id, _ = split_email_id(email_id)
        for email in self.emails(generation_id):
            if email.id == email_id:
                return email
        raise KeyError(email_id)

    def _dir(self, generation_id: str) -> Path:
        if not is_safe_id(generation_id):
            raise KeyError(generation_id)
        return self._root / generation_id
```

`src/jev_bench/store/runs.py`:
```python
"""Run persistence: `run.json` meta, append-only predictions and raw responses per run.

Types:
    RunMode, RunParams
Classes:
    RunMeta: persisted description, snapshots and totals of one run.
    Prediction: one email's outcome within a run.
    ResponseRecord: one HTTP request's raw outcome (stored once per request).
    RunStore: save / get / list runs and read or append their predictions and responses.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, Field

from jev_bench.benchmark_config import ColumnConfig, ColumnKind, EmbeddingParams, JevParams, LlmParams
from jev_bench.ids import is_safe_id
from jev_bench.questions import Distribution, QuestionSet
from jev_bench.store.jsonfiles import append_jsonl, read_json, read_jsonl, write_json_atomic
from jev_bench.store.status import JobStatus

META_FILE = "run.json"
PREDICTIONS_FILE = "predictions.jsonl"
RESPONSES_FILE = "responses.jsonl"

type RunMode = Literal["per_email", "all_in_one", "batched"]
RunParams = Annotated[JevParams | LlmParams | EmbeddingParams, Field(discriminator="kind")]


class RunMeta(BaseModel):
    id: str
    column: str
    column_config: ColumnConfig | None = None
    kind: ColumnKind
    model: str
    resolved_models: tuple[str, ...] = ()
    generation_ids: tuple[str, ...]
    mode: RunMode
    emails_per_request: int | None
    question_set: QuestionSet
    params: RunParams
    concurrency: int
    status: JobStatus = "running"
    created_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    duration_s: float | None = None
    n_emails: int
    n_done: int = 0
    n_errors: int = 0
    n_requests: int = 0
    n_splits: int = 0
    total_cost: float = 0.0
    setup_cost: float = 0.0
    setup_cached_cost: float = 0.0
    cold_cost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    error: str | None = None


class Prediction(BaseModel):
    email_id: str
    answers: dict[str, Distribution] | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    request_index: int | None = None
    batch_size: int = 0
    latency_ms: float | None = None
    cost: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cost_estimated: bool = False
    resolved_model: str | None = None
    similarities: dict[str, dict[str, float]] | None = None
    cached: bool = False
    cached_cost: float = 0.0


class ResponseRecord(BaseModel):
    request_index: int
    email_ids: tuple[str, ...]
    latency_ms: float | None = None
    error: str | None = None
    body: dict[str, Any] | None = None


class RunStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, meta: RunMeta) -> None:
        write_json_atomic(self._dir(meta.id) / META_FILE, meta.model_dump(mode="json"))

    def get(self, run_id: str) -> RunMeta:
        path = self._dir(run_id) / META_FILE
        if not path.exists():
            raise KeyError(run_id)
        return RunMeta.model_validate(read_json(path))

    def list_metas(self) -> list[RunMeta]:
        if not self._root.exists():
            return []
        ids = [path.name for path in self._root.iterdir() if is_safe_id(path.name) and (path / META_FILE).exists()]
        return sorted((self.get(run_id) for run_id in ids), key=lambda meta: meta.id, reverse=True)

    def append_predictions(self, run_id: str, predictions: Iterable[Prediction]) -> None:
        records = [prediction.model_dump(mode="json") for prediction in predictions]
        append_jsonl(self._dir(run_id) / PREDICTIONS_FILE, records)

    def predictions(self, run_id: str) -> list[Prediction]:
        return [Prediction.model_validate(record) for record in read_jsonl(self._existing(run_id) / PREDICTIONS_FILE)]

    def append_response(self, run_id: str, record: ResponseRecord) -> None:
        append_jsonl(self._dir(run_id) / RESPONSES_FILE, [record.model_dump(mode="json")])

    def responses(self, run_id: str) -> list[ResponseRecord]:
        return [ResponseRecord.model_validate(record) for record in read_jsonl(self._existing(run_id) / RESPONSES_FILE)]

    def _existing(self, run_id: str) -> Path:
        directory = self._dir(run_id)
        if not directory.exists():
            raise KeyError(run_id)
        return directory

    def _dir(self, run_id: str) -> Path:
        if not is_safe_id(run_id):
            raise KeyError(run_id)
        return self._root / run_id
```

`src/jev_bench/store/labels.py`:
```python
"""Human labels per generation: `{email_id: {question_id: option_id}}`, edited atomically.

Classes:
    LabelStore: read labels per email or generation; apply edits under a lock.
"""

import asyncio
from collections.abc import Iterable, Mapping
from pathlib import Path

from jev_bench.ids import is_safe_id, split_email_id
from jev_bench.store.jsonfiles import read_json, write_json_atomic

type Labels = dict[str, dict[str, str]]


class LabelStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = asyncio.Lock()

    def for_generations(self, generation_ids: Iterable[str]) -> Labels:
        merged: Labels = {}
        for generation_id in generation_ids:
            merged.update(self._read(generation_id))
        return merged

    def for_email(self, email_id: str) -> dict[str, str]:
        generation_id, _ = split_email_id(email_id)
        return self._read(generation_id).get(email_id, {})

    async def update(self, email_id: str, changes: Mapping[str, str | None]) -> dict[str, str]:
        generation_id, _ = split_email_id(email_id)
        async with self._lock:
            labels = self._read(generation_id)
            current = _apply(labels.get(email_id, {}), changes)
            if current:
                labels[email_id] = current
            else:
                labels.pop(email_id, None)
            write_json_atomic(self._path(generation_id), labels)
            return current

    def _read(self, generation_id: str) -> Labels:
        path = self._path(generation_id)
        return read_json(path) if path.exists() else {}

    def _path(self, generation_id: str) -> Path:
        if not is_safe_id(generation_id):
            raise KeyError(generation_id)
        return self._root / f"{generation_id}.json"


def _apply(current: Mapping[str, str], changes: Mapping[str, str | None]) -> dict[str, str]:
    updated = dict(current)
    for question_id, option in changes.items():
        if option is None:
            updated.pop(question_id, None)
        else:
            updated[question_id] = option
    return updated
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_generation_config.py tests/test_store_generations.py tests/test_store_runs.py tests/test_store_labels.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/generation src/jev_bench/store tests/test_generation_config.py tests/test_store_generations.py tests/test_store_runs.py tests/test_store_labels.py
git commit -m "feat(store): generation config model and generation, run and label stores

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
