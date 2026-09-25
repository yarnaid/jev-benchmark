# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmark for **Jev** (`typesafe/jev-1.13`, TypeSafe's decision model on OpenRouter) against an Anthropic
chat model, an OpenAI chat model and an embedding-similarity baseline on synthetic email triage. Everything
goes through OpenRouter.

Design: `docs/superpowers/specs/2026-09-24-jev-benchmark-design.md` (§14 records revisions made while
planning). Implementation plan: `docs/superpowers/plans/2026-09-24-jev-benchmark.md`.
Question model v2 (multi-label category, 0–100 scores, new questions): spec
`docs/superpowers/specs/2026-09-25-question-model-v2-design.md`, plan
`docs/superpowers/plans/2026-09-25-question-model-v2.md`.

**All repository content is in English** (code, docs, UI copy, prompts, commits).

## Commands

```bash
uv sync                                   # install (Python >= 3.14)
uv run jev-bench serve                    # UI + API on http://127.0.0.1:8000
uv run jev-bench generate --count 200     # new generation (needs OPENROUTER_API_KEY in env/.env)
uv run jev-bench run anthropic -g <generation-id> [--mode all_in_one] [--model anthropic/claude-sonnet-5]

uv run pytest                             # default suite (no network; integration + slow excluded)
uv run pytest tests/test_compare_report.py -k fleiss -v    # one module / one test
uv run pytest --cov --cov-fail-under=95   # coverage gate
uv run pytest -m integration tests/test_integration_openrouter.py   # real, PAID OpenRouter calls
uv run ruff check --fix && uv run ruff format && uv run pyright     # after every change
node --check src/jev_bench/web/static/js/*.js   # static JS syntax gate (Node, no npm)
node --test tests/js/                           # JS unit tests (Node's runner, no npm)
```

`-m integration` is opt-in and makes real, paid OpenRouter calls; it is never run automatically and needs
`OPENROUTER_API_KEY`.

## Architecture (read these together)

- **The question set is the single source of truth** (`config/questions.toml` → `questions.py`).
  - Types mirror Jev's primitives (`choice` / `score` / `noul`) plus `multi` (multi-label).
  - Every answer from every source is normalized to a distribution `{option_id: p}` summing to 1; `noul`
    becomes `{yes, no}`.
  - A multi question is answered like a `choice`; Jev gets it as a native Decisions `choice`.
  - Its **applied labels** are the options with `p >= threshold × max(p)`, so the top option always applies:
    - the rule is implemented once in Python, in `metrics.multilabel.relative_labels`;
    - the UI mirror is `answers.appliedLimit`;
    - the threshold is the question's `threshold` (default 0.8), overridable with `?threshold=` on
      `/api/compare` and `/api/emails` (the UI slider).
  - Hard answers (generator reference, human labels) are an option id, or a non-empty list of unique option
    ids for multi (`questions.HardAnswer`, `hard_distribution`). A label list becomes a uniform distribution
    over its labels.
  - `compatible()` is shape-based: choice, score and multi questions with the same option ids are
    comparable, so v1 runs stay comparable on `category` and `sentiment`.
  - Score questions also get a 0–100 score (`metrics.distributions.score_0_100`, the only implementation).
  - Jev payloads (`classifiers/jev.py`), LLM JSON schemas (`classifiers/llm_schema.py`), embedding option
    texts, generator schemas, metrics and the UI are all derived from it.
- **Columns** (`config/benchmark.toml` → `benchmark_config.py`) are `decisions` / `chat` / `embeddings`.
  - Each kind has a classifier implementing the `Classifier` protocol (`classifiers/base.py`): `prepare()`
    once, then `classify(batch)` per request.
  - `run_launcher.py` validates a request and builds the classifier. Chat parameters missing from the
    model's catalog `supported_parameters` (`temperature` on Claude Sonnet 5 and GPT-5.6 Terra) are not
    sent, because `provider.require_parameters` would otherwise 404; the run snapshot records them as
    `null`.
  - `runner.py` plans requests under the token budget (`tokens.py` + `request_plan.py`: minimal equal
    contiguous split), runs them under a semaphore, and appends predictions and responses.
- **Generation** (`generation/`): a seeded trait plan (`plan.py`), then prompts and strict schema
  (`prompt.py`), then the job (`generator.py`). The generator's own answers become `reference_answers`.
  Unusable output (empty, not JSON, schema-invalid) is retried with the same model up to `max_attempts`
  (`config/generation.toml`); text around the JSON object is ignored; provider errors are not retried there.
- **Persistence**: JSON/JSONL under `data/` (`store/`). `data/embeddings/` is a gitignored per-model vector
  cache keyed by sha256 of the exact input text.
  - Each `EmbeddingCache` loads its vectors once per process (`EmbeddingCache._load`), so a CLI run and a
    running web server don't see each other's newly cached vectors until restart; results are unaffected,
    but the embedding cost may be paid twice.
  - Runs and generations left `running` become `interrupted` when the web server starts
    (`Services.sweep_interrupted`; the CLI does not sweep). This sweep can transiently mark a CLI job that
    is still actually running as `interrupted`; the CLI job's own `_finish` rewrites the final status when
    it completes, so the persisted status ends up correct either way.
- **Comparison** (`compare/`: `raters.py`, `pairs.py`, `report.py`, `rows.py`; pure numpy via `metrics/`):
  - raters are runs, the generator reference and optional human labels; a rater whose question-set
    snapshot is incompatible with the base is skipped with a warning;
  - pairwise agreement / κ (quadratic for score) / JSD / Pearson / Brier with bootstrap CIs, plus Fleiss'
    κ over runs;
  - a per-email disagreement index;
  - multi-label questions use `metrics/multilabel.py` via `compare/multi.py`:
    - over the applied label sets: exact-set match, Jaccard, micro-F1, macro κ, macro Fleiss' κ, label counts;
    - over the distributions, as for choice: JSD, entropy, confidence and Brier (against a uniform target);
  - `GET /api/compare` takes its base question set from the **newest** selected run's snapshot, so the URL
    order of `runs=` never changes the report.
- **Web**: `web/app.py` (lifespan builds `Services`) exposes JSON routes under `/api` (`web/routes/`) and a
  build-free UI in `web/static/` (Bootstrap 5.3 + Chart.js from jsDelivr with SRI).
- **CLI**: `cli.py` (typer, lazy imports) delegates to `cli_jobs.py`, which uses the same launchers and
  `Services` as the web app.

## Invariants you must not break

- **Models see only** `Email.to_state()`: `{sent_at, from, to, cc, subject, body}`. They never see email
  ids (these contain generation name slugs), traits, reference answers or the generator model.
  All-in-one prompts address emails as `e001…`.
- **Generator answers never feed the benchmark.** For example, τ for embeddings is fixed in config, not
  tuned on references.
- **API keys:**
  - a key entered in the browser (`X-OpenRouter-Key`) overrides the server key (`OPENROUTER_API_KEY`);
    without one the server key is used;
  - the browser sends the header only on job-starting POSTs (`/api/runs`, `/api/generations`,
    `/api/analyses`);
  - keys are passed per call (never set on the shared client) and are never persisted or logged. A test
    greps `data/` and the logs for a sentinel key.
- **UI:**
  - never use `innerHTML` / `outerHTML` / `insertAdjacentHTML` / `document.write`: build DOM with `h()`
    from `js/dom.js`. Email bodies are hostile by design (prompt injections);
  - no inline scripts. `tests/test_web_static.py` enforces both, along with SRI and pinned CDN versions.
- **Config** TOMLs are re-read on every run/generation start, and runs snapshot what they used. Edit
  `config/*.toml` instead of code to change prompts, questions, traits or default models.

## Key safety

- A non-blank browser-supplied `X-OpenRouter-Key` overrides the server key (`OPENROUTER_API_KEY`); a
  blank or missing header falls back to the server key (`Services.api_key`). The header is sent only on
  `POST /api/runs`, `POST /api/generations` and `POST /api/analyses`.
- Keys are never persisted (run/generation/response files) and never logged.
- `classifiers/*` and `GeneratorDeps` hold the key as `SecretStr`, never a plain `str` field.
- Every entry point (`cli.py`, `web/app.py`) calls `log_setup.configure_logging()` first, which runs with
  `diagnose=False` so a traceback can't leak a key from a local variable.
- A provider refusal (401/402/403) or a 404 (unknown model / no provider for the requested parameters)
  aborts the job and logs the message without a traceback.
- `cli.py` pins Typer's `pretty_exceptions_show_locals=False`, so an uncaught exception's pretty
  traceback can't print a local variable holding a key.

## Conventions

- `httpx2` (Pydantic's maintained httpx continuation, same API), never `httpx`; no OpenAI SDK.
- Every Python file starts with a docstring listing its classes and functions; no inline comments; full type
  annotations; pydantic models for known shapes; `loguru` for logs, `rich` for CLI output.
- Tests:
  - one file per module;
  - tables via `pytest.param(..., id=...)`;
  - OpenRouter is always mocked (`httpx2.MockTransport`). Use `tests.factories.FakeOpenRouter`, the
    `make_client` / `make_services` / `make_app` fixtures and the mini configs in `tests/factories.py`;
  - each test < 50 ms (global `timeout = 1`).
- Pre-commit checks: `ruff check --fix && ruff format && pyright` (0 errors), `pytest` (default suite),
  `node --check src/jev_bench/web/static/js/*.js` and `node --test tests/js/` (Node's own tooling, no npm
  dependency).
- Default suite target < 5 s. Keep `jev-bench --help` < 500 ms: no heavy imports at the top of `cli.py`.
