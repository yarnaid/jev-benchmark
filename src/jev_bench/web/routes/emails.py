"""Explorer routes: email rows for generations (with per-run top answers) and single-email detail.

Classes:
    EmailList: current question set + rows.
    EmailDetail: email, human labels, predictions per run, run labels, current question set.
Functions:
    list_emails: GET /emails?generations=…&runs=…
    email_detail: GET /emails/{email_id}?runs=…
"""

from fastapi import APIRouter
from pydantic import BaseModel

from jev_bench.compare import EmailRow, email_rows, run_label
from jev_bench.emails import Email
from jev_bench.questions import QuestionSet
from jev_bench.services import Services
from jev_bench.store.runs import Prediction
from jev_bench.web.deps import ServicesDep, split_ids
from jev_bench.web.loaders import load_email, load_emails, load_runs, run_raters

router = APIRouter(tags=["emails"])


class EmailList(BaseModel):
    questions: QuestionSet
    rows: list[EmailRow]


class EmailDetail(BaseModel):
    email: Email
    human: dict[str, str]
    predictions: dict[str, Prediction]
    run_labels: dict[str, str]
    questions: QuestionSet


@router.get("/emails")
def list_emails(services: ServicesDep, generations: str, runs: str | None = None) -> EmailList:
    generation_ids = split_ids(generations)
    questions = services.question_set()
    emails = load_emails(services, generation_ids)
    raters = run_raters(services, load_runs(services, split_ids(runs)))
    labels = services.labels.for_generations(generation_ids)
    return EmailList(questions=questions, rows=email_rows(emails, raters, labels, questions))


@router.get("/emails/{email_id}")
def email_detail(email_id: str, services: ServicesDep, runs: str | None = None) -> EmailDetail:
    email = load_email(services, email_id)
    metas = load_runs(services, split_ids(runs))
    predictions = {
        meta.id: found
        for meta in metas
        if (found := _prediction_for(services, meta.id, email_id)) is not None
    }
    return EmailDetail(
        email=email,
        human=services.labels.for_email(email_id),
        predictions=predictions,
        run_labels={meta.id: run_label(meta) for meta in metas},
        questions=services.question_set(),
    )


def _prediction_for(services: Services, run_id: str, email_id: str) -> Prediction | None:
    return next(
        (
            prediction
            for prediction in services.runs.predictions(run_id)
            if prediction.email_id == email_id
        ),
        None,
    )
