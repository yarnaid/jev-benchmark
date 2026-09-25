"""Generation configuration: generator mix, prompt templates, dataset traits.

Classes:
    TraitValue: one weighted trait value with its prompt text.
    Trait: a dataset knob, either linked to a question's options or with explicit values.
    GenerationConfig: the whole file (templates validated against the declared traits;
        `max_attempts` bounds tries per email when the generator's output is unusable).
Functions:
    load_generation_config: parse and validate the TOML file.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jev_bench.templates import check_template

__all__ = [
    "GenerationConfig",
    "Trait",
    "TraitValue",
    "load_generation_config",
]


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
    def _exactly_one_source(self) -> Trait:
        if (self.question is None) == (not self.values):
            msg = f"trait {self.name!r} needs either `question` or `values`, not both or neither"
            raise ValueError(msg)
        return self


class GenerationConfig(_Frozen):
    models: tuple[str, ...] = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0)
    concurrency: int = Field(default=8, ge=1)
    max_attempts: int = Field(default=3, ge=1, le=10)
    sent_at_window_days: int = Field(default=30, ge=1)
    system_prompt: str
    user_prompt: str
    traits: tuple[Trait, ...] = ()

    @model_validator(mode="after")
    def _traits_and_templates(self) -> GenerationConfig:
        names = [trait.name for trait in self.traits]
        if len(names) != len(set(names)):
            raise ValueError("duplicate trait names")
        allowed = {"questions", "sent_at", *names, *(f"{name}_prompt" for name in names)}
        check_template(self.system_prompt, allowed)
        check_template(self.user_prompt, allowed)
        return self


def load_generation_config(path: Path) -> GenerationConfig:
    return GenerationConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
