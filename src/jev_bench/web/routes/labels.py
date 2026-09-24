"""Human labelling route: set or clear answers for one email, validated against the question set.

Classes:
    LabelUpdate: `{answers: {question_id: option_id | null}}`.
Functions:
    put_label: PUT /labels/{email_id}
"""

from collections.abc import Mapping

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from jev_bench.questions import QuestionSet
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

    answers: dict[str, str | None]


@router.put("/labels/{email_id}")
async def put_label(email_id: str, update: LabelUpdate, services: ServicesDep) -> dict[str, str]:
    load_email(services, email_id)
    _validate(update.answers, services.question_set())
    return await services.labels.update(email_id, update.answers)


def _validate(answers: Mapping[str, str | None], questions: QuestionSet) -> None:
    for question_id, option in answers.items():
        try:
            question = questions.get(question_id)
        except KeyError as exc:
            detail = f"unknown question {question_id!r}"
            raise HTTPException(status_code=400, detail=detail) from exc
        if option is not None and option not in question.options:
            detail = f"{option!r} is not an option of {question_id!r}"
            raise HTTPException(status_code=400, detail=detail)
