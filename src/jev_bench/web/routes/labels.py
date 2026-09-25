"""Human labelling route: set or clear answers for one email, validated against the question set.

Classes:
    LabelUpdate: `{answers: {question_id: option_id | [option_id, ...] | null}}`; a list is for a
        multi-label question, and `null` or `[]` clears the answer.
Functions:
    put_label: PUT /labels/{email_id}
"""

from collections.abc import Mapping

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from jev_bench.questions import AnyQuestion, HardAnswer, QuestionSet, hard_distribution
from jev_bench.web.deps import ServicesDep
from jev_bench.web.loaders import load_email

__all__ = [
    "LabelUpdate",
    "put_label",
    "router",
]

router = APIRouter(tags=["labels"])


class LabelUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: dict[str, HardAnswer | None]


@router.put("/labels/{email_id}")
async def put_label(
    email_id: str, update: LabelUpdate, services: ServicesDep
) -> dict[str, HardAnswer]:
    load_email(services, email_id)
    changes = _validated(update.answers, services.question_set())
    return await services.labels.update(email_id, changes)


def _validated(
    answers: Mapping[str, HardAnswer | None], questions: QuestionSet
) -> dict[str, HardAnswer | None]:
    changes = {
        question_id: None if answer == [] else answer for question_id, answer in answers.items()
    }
    for question_id, answer in changes.items():
        question = _question(questions, question_id)
        if answer is not None and hard_distribution(question, answer) is None:
            detail = f"{answer!r} is not an option of {question_id!r}"
            raise HTTPException(status_code=400, detail=detail)
    return changes


def _question(questions: QuestionSet, question_id: str) -> AnyQuestion:
    try:
        return questions.get(question_id)
    except KeyError as exc:
        detail = f"unknown question {question_id!r}"
        raise HTTPException(status_code=400, detail=detail) from exc
