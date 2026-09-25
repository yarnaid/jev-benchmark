"""Placeholder values of the analysis prompts, and prompt rendering.

Classes:
    PromptInputs: placeholder -> text, the e001… refs the text uses (ref -> email id), and how many
        disputed emails it includes.
Functions:
    prompt_inputs: every placeholder's text for a source.
    render_prompts: (system, user) templates -> the prompts actually sent. Substitution is a single
        pass, so a `$` inside the data (an email body, say) is never expanded.
"""

from collections.abc import Mapping
from string import Template
from typing import NamedTuple

from jev_bench.analysis.report_json import report_json
from jev_bench.analysis.source import AnalysisSource, email_refs, rater_names
from jev_bench.analysis.summaries import generations_text, questions_text, runs_text
from jev_bench.analysis.tables import disputed_emails, disputed_ids, email_tables

__all__ = [
    "PromptInputs",
    "prompt_inputs",
    "render_prompts",
]


class PromptInputs(NamedTuple):
    values: dict[str, str]
    email_refs: dict[str, str]
    n_disputed: int


def prompt_inputs(source: AnalysisSource, *, max_disputed: int) -> PromptInputs:
    names = rater_names(source)
    refs = email_refs(source.emails)
    by_id = {email_id: ref for ref, email_id in refs.items()}
    disputed = disputed_ids(source.rows, max_disputed)
    values = {
        "generations": generations_text(source.generations, source.emails),
        "runs": runs_text(source.metas, names),
        "questions": questions_text(source.questions),
        "report": report_json(source.report, names),
        "emails": email_tables(source, names, by_id),
        "disputed": disputed_emails(source, by_id, disputed),
    }
    return PromptInputs(values, refs, len(disputed))


def render_prompts(system: str, user: str, values: Mapping[str, str]) -> tuple[str, str]:
    return Template(system).substitute(values), Template(user).substitute(values)
