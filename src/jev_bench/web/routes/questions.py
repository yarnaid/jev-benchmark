"""Question set route: the current `config/questions.toml`, for the Help page.

Functions:
    questions: GET /questions (a broken question file is a 500 that names the file).
"""

import tomllib

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from jev_bench.questions import QuestionSet
from jev_bench.web.deps import ServicesDep

__all__ = [
    "questions",
    "router",
]

router = APIRouter(tags=["questions"])


@router.get("/questions")
def questions(services: ServicesDep) -> QuestionSet:
    try:
        return services.question_set()
    except (ValidationError, tomllib.TOMLDecodeError) as exc:
        detail = f"config/questions.toml is invalid: {exc}"
        raise HTTPException(status_code=500, detail=detail) from exc
