# Jev Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking. Each task lives in its own file under `2026-09-24-jev-benchmark/`; an implementer needs only
> this index (Global Constraints + Review Focus) and their task file.

**Goal:** A local web app plus CLI that generates synthetic email datasets and benchmarks Jev against an
Anthropic chat model, an OpenAI chat model and an embedding-similarity baseline via OpenRouter. It reports
speed, cost and inter-rater agreement, and supports optional human labels.

**Architecture:**

- A typed Python core; every Python file has one responsibility:
  - models: questions, emails, configs;
  - an OpenRouter client, a catalog and a token-budget planner;
  - three classifiers;
  - JSON/JSONL stores, a job registry, a runner and a generator;
  - pure numpy metrics and a comparison engine.
- FastAPI serves a JSON API and a build-free vanilla-JS UI (Bootstrap 5.3 + Chart.js).
- The CLI and the web app share one `Services` container and the same launchers.

**Tech Stack:** Python ≥ 3.14, uv, FastAPI, uvicorn, httpx2, pydantic v2, pydantic-settings, typer, loguru,
rich, numpy; pytest + pytest-asyncio + pytest-cov + pytest-timeout + hypothesis + factory-boy; ruff,
pyright; Bootstrap 5.3.8, Bootstrap Icons 1.13.1, Chart.js 4.5.1.

**Spec:** `docs/superpowers/specs/2026-09-24-jev-benchmark-design.md` (read §14 "Revisions made while
planning" first).

## Global Constraints

- **Language:** all repository content in English (code, docs, UI copy, prompts, commits).
- **Python:** `requires-python = ">=3.14"`. Dependencies only via `uv add` / `uv add --dev`; never `pip`.
- **HTTP client:** `httpx2` (import `httpx2`), never `httpx`. No OpenAI SDK.
- **Every Python file:**
  - starts with a module docstring: purpose, plus the list of classes and functions;
  - has no inline comments;
  - has full type annotations on arguments, returns and non-trivial variables.
- **Types:**
  - pydantic models (or `NamedTuple` / `TypedDict`) for known shapes; no `dataclasses`, no raw dicts for
    known shapes;
  - `pathlib` only, never `os.path`;
  - `loguru` for logging, `rich` for CLI output, never `print`.
- **Functions:** short (≤ ~10 lines where practical), one abstraction level each.
- **Tests:**
  - one file per module (`tests/test_<module>.py`), testing only that module's public API;
  - parametrize with `pytest.param(..., id="...")` rows;
  - no real network: every OpenRouter call goes through `httpx2.MockTransport`;
  - each test < 50 ms (global `timeout = 1`); timeout budgets 10–100 ms.
- **After every edit:** `uv run ruff check --fix && uv run ruff format`, then `uv run pyright` (0 errors)
  before committing.
- **Commits:** one commit per task, message style `feat(scope): ...` / `test: ...` / `docs: ...`, ending
  with the trailer:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`
- **Default models:**
  - Jev: `typesafe/jev-1.13`
  - Anthropic: `anthropic/claude-sonnet-5`
  - OpenAI: `openai/gpt-5.6-terra`
  - Embeddings: `openai/text-embedding-3-large`
  - Generator mix: `google/gemini-3.8-flash`, `deepseek/deepseek-v4.1-flash`, `z-ai/glm-5.3`
- **OpenRouter base URL:** `https://openrouter.ai/api`. Paths: `/alpha/decisions`,
  `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`.
- **What models see:** exactly `{sent_at, from, to, cc, subject, body}` (`Email.to_state()`). Never
  email ids, traits, reference answers or the generator model.
- **API keys:** never persisted (run/generation/responses files) and never logged. The server key wins;
  the `X-OpenRouter-Key` header is used only when the server has none.
- **CDN assets** (exact URLs and SRI):
  - `https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css` —
    `sha384-sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB`
  - `https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js` —
    `sha384-FKyoEForCGlyvwx9Hj09JcYn3nv7wiPVlz7YYwJrWVcXK/BmnVDxM+D2scQbITxI`
  - `https://cdn.jsdelivr.net/npm/bootstrap-icons@1.13.1/font/bootstrap-icons.min.css` —
    `sha384-CK2SzKma4jA5H/MXDUU7i1TqZlCFaD4T01vtyDFvPlD97JQyS+IsSh1nI2EFbpyk`
  - `https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js` —
    `sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ`
- **Frontend:**
  - no `innerHTML`, `outerHTML`, `insertAdjacentHTML` or `document.write` anywhere; all DOM is built with
    `h()` from `js/dom.js`;
  - no inline `<script>`;
  - every `localStorage` access is wrapped in `try/catch`.
- **Coverage gate** (final task): `uv run pytest --cov --cov-fail-under=95`.

## Review Focus

These are the five input classes and failure modes most likely to bite a real user that happy-path tests
would miss. Each one is pinned by tests in the tasks named at the end of its line.

1. **Hostile email content.** A body with `<script>`, `<img onerror>` or prompt-injection text must reach
   models verbatim and render as literal text in the UI. Pinned by an API test that returns the body
   byte-for-byte (**Task 21**). A static test greps every JS file for `innerHTML` / `outerHTML` /
   `insertAdjacentHTML` / `document.write` and rejects inline scripts and handlers (**Task 22**).
2. **Path traversal through ids.** A generation, run or email id like `../../etc` passed to a store, route
   or loader → `KeyError` / 404, and no file outside `data/` is touched (**Task 11**, **Task 20**,
   **Task 21**).
3. **A crash mid-write.** A torn or corrupt JSONL line is skipped, and the next append starts on a fresh
   line (**Task 3**). After a restart, `running` runs and generations become `interrupted` with totals
   recomputed from disk (**Task 14**, **Task 16**, **Task 19**).
4. **A malformed provider response.** Examples: HTTP 200 with an `error` object; non-JSON content; a
   ```` ```json ```` fence; NaN, negative, string or zero-mass probabilities; a truncated stream
   (`finish_reason=length`); an embedding-count mismatch. Each becomes a per-email error or a per-question
   note, and the run continues. Only 401/402/403 abort a run (**Task 5**, **Task 9**, **Task 10**,
   **Task 12**, **Task 14**).
5. **Concurrent work sharing state.** Two embedding classifiers on one cache instance both finish, and the
   cache file stays parseable (**Task 12**). Two label edits on the same generation are both kept
   (**Task 11**). Browser keys never end up in `data/` or the logs, even with concurrent jobs
   (**Task 20**).

## File map

```
pyproject.toml, .gitignore, .env.example, CLAUDE.md
config/questions.toml, config/generation.toml, config/benchmark.toml
src/jev_bench/
  __init__.py                 package marker (docstring only)
  settings.py                 env settings (pydantic-settings)
  ids.py                      slugify, timestamped ids, safe-id checks
  questions.py                question set models, loader, prompt rendering, one_hot, compatible
  emails.py                   Party, Email, EmailState, email_id
  templates.py                string.Template placeholder validation
  benchmark_config.py         benchmark.toml models + loader
  openrouter.py               OpenRouterClient, errors, SSE accumulator, chat_content, json_schema_format
  catalog.py                  ModelInfo, Catalog (TTL cache), select_models
  tokens.py                   estimate_tokens, Budget, chat/jev/embedding budgets
  request_plan.py             chunk_by_count, plan_requests (balanced equal split)
  json_schema.py              strict_object
  metrics/distributions.py    normalize, unit_probability, softmax, argmax, entropy, JSD, expected level
  metrics/agreement.py        agreement, Cohen/Fleiss kappa, Pearson, Brier
  metrics/bootstrap.py        resample_index, percentile_ci
  classifiers/base.py         Usage, EmailOutcome, PrepareResult, RequestResult, Classifier protocol
  classifiers/jev.py          Decisions payload + parsing, JevClassifier
  classifiers/llm_schema.py   answers / all-in-one JSON schemas, email refs
  classifiers/llm_parse.py    per-email and all-in-one answer parsing, RefCounter
  classifiers/llm.py          LlmClassifier (per_email / all_in_one)
  classifiers/embeddings.py   EmbeddingClassifier (cache-first, cosine → softmax)
  store/jsonfiles.py          atomic JSON, JSONL append/read (tolerant of torn lines)
  store/status.py             JobStatus
  store/embeddings.py         EmbeddingCache, EmbeddingCaches, vector codec
  store/generations.py        GenerationMeta, GenerationStore
  store/runs.py               RunMeta, Prediction, ResponseRecord, RunStore
  store/labels.py             LabelStore
  generation/config.py        GenerationConfig, Trait, load_generation_config
  generation/plan.py          ResolvedTrait, PlanItem, resolve_traits, build_plan
  generation/prompt.py        render_prompts, generation_schema, parse_generator_output, count_mismatches
  generation/generator.py     execute_generation, mark_interrupted_generations
  generation/launcher.py      GenerationRequest, launch_generation
  jobs.py                     JobProgress, ProgressView, JobRegistry, cancel_status, describe_error
  runner.py                   execute_run, summarize, mark_interrupted_runs
  run_launcher.py             RunRequest, RunLaunchError, launch_run, build_classifier
  compare.py                  raters, compare(), email_rows(), disagreement_index, run_label
  services.py                 Services container
  web/app.py                  create_app, create_default_app
  web/security.py             CSP middleware
  web/deps.py                 get_services, require_api_key, split_ids
  web/loaders.py              store lookups mapped to HTTP 404
  web/routes/{status,catalog,generations,runs,compare,emails,labels}.py
  web/static/{index,generations,explorer,runs}.html, css/app.css,
             js/{dom,format,storage,key,api,layout,widgets,charts,report,distribution,
                 benchmark,generations,explorer,runs}.js
  cli.py                      typer app (lazy imports)
  cli_jobs.py                 async CLI bodies (launch + rich progress)
tests/ conftest.py, factories.py, test_*.py (one per module), slow_tests.txt
```

## Tasks (execute in order)

| # | File | Deliverable |
|---|---|---|
| 1 | `task-01-scaffold-settings.md` | uv project, tooling config, `settings.py`, `ids.py`, test infra |
| 2 | `task-02-questions-emails.md` | `questions.py`, `emails.py`, `config/questions.toml`, factories |
| 3 | `task-03-jsonfiles.md` | `store/jsonfiles.py`, `store/status.py` |
| 4 | `task-04-metrics.md` | `metrics/distributions.py`, `metrics/agreement.py`, `metrics/bootstrap.py` |
| 5 | `task-05-openrouter.md` | `openrouter.py` |
| 6 | `task-06-benchmark-config-catalog.md` | `templates.py`, `benchmark_config.py`, `config/benchmark.toml`, `catalog.py` |
| 7 | `task-07-tokens-plan.md` | `tokens.py`, `request_plan.py` |
| 8 | `task-08-classifier-base-jev.md` | `classifiers/base.py`, `classifiers/jev.py` |
| 9 | `task-09-llm-schema.md` | `json_schema.py`, `classifiers/llm_schema.py`, `classifiers/llm_parse.py` |
| 10 | `task-10-llm-classifier.md` | `classifiers/llm.py` |
| 11 | `task-11-stores.md` | `generation/config.py`, `store/generations.py`, `store/runs.py`, `store/labels.py` |
| 12 | `task-12-embeddings.md` | `store/embeddings.py`, `classifiers/embeddings.py` |
| 13 | `task-13-jobs.md` | `jobs.py` |
| 14 | `task-14-runner.md` | `runner.py` |
| 15 | `task-15-generation-core.md` | `generation/plan.py`, `generation/prompt.py`, `config/generation.toml` |
| 16 | `task-16-generator-services.md` | `generation/generator.py`, `generation/launcher.py`, `services.py`, shared test fixtures |
| 17 | `task-17-run-launcher.md` | `run_launcher.py`, `FakeOpenRouter` |
| 18 | `task-18-compare.md` | `compare.py` |
| 19 | `task-19-web-core.md` | `web/app.py`, `web/security.py`, `web/deps.py`, status + catalog routes |
| 20 | `task-20-web-jobs.md` | generations + runs routes (key-never-persisted test) |
| 21 | `task-21-web-data.md` | `web/loaders.py`, compare + emails + labels routes |
| 22 | `task-22-ui-shell.md` | `dom.js`, `format.js`, `storage.js`, `key.js`, `api.js`, `layout.js`, `widgets.js`, `app.css`, static contract test |
| 23 | `task-23-ui-benchmark.md` | Benchmark page (`index.html`, `benchmark.js`, `report.js`, `charts.js`) |
| 24 | `task-24-ui-pages.md` | Generations, Explorer and Runs pages (`distribution.js`) |
| 25 | `task-25-cli-docs-verify.md` | `cli.py`, `cli_jobs.py`, `CLAUDE.md`, integration smoke, final gates, browser smoke |
