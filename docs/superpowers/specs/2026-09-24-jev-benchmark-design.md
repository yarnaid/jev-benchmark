# Jev Benchmark — Design

- **Date:** 2026-09-24
- **Status:** draft, awaiting review

## 1. Purpose

Benchmark **Jev** (`typesafe/jev-1.13`, TypeSafe's decision model on OpenRouter) against frontier chat
models (one Anthropic, one OpenAI) on email triage. Every model gets the same emails and the same typed
questions and returns a probability for every answer option. There is no ground truth: the benchmark
measures **speed, cost and how closely the models agree** with each other, with the generator's reference
answers and, when present, with optional human labels.

### Success criteria

1. `jev-bench generate` (CLI) and the Generations page (UI) each produce a separate, persisted generation of
   synthetic emails with reference answers.
2. The Benchmark page runs any subset of generations through each of three columns (Jev / Anthropic /
   OpenAI). Each column's model is picked from a list filled live from the OpenRouter catalog. The page
   shows live progress, wall-clock duration and total USD cost.
3. Every run is persisted and can be reopened and compared with any other run later.
4. Comparison shows per-model answer distributions (e.g. "how many emails each model called `spam`"),
   pairwise agreement metrics with confidence intervals, and group agreement.
5. The Explorer shows every email with the reference answers and each model's distributions side by side,
   and allows manual labelling.
6. Everything works with **zero human labels**: human-based metrics appear only once labels exist.
7. Correct with no network: the test suite mocks every OpenRouter call; coverage ≥ 95 %.

### Non-goals (foundation)

Auth / multi-user; resuming interrupted runs; deleting runs or generations from the UI; editing TOML config
from the UI; logprob-based probabilities; Jev's System One API.

## 2. External contracts (verified 2026-09-24)

### Jev: Decisions API

`POST https://openrouter.ai/api/alpha/decisions`, `Authorization: Bearer $OPENROUTER_API_KEY`.

```json
{
  "model": "typesafe/jev-1.13",
  "state": {"from": "...", "to": ["..."], "cc": [], "subject": "...", "body": "..."},
  "questions": {
    "needs_reply": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "category":    {"type": "choice", "instructions": "...", "criteria": {"spam": "...", "personal": "..."}},
    "urgency":     {"type": "score", "instructions": "...", "criteria": ["lowest level", "...", "highest level"]}
  }
}
```

The response contains `answers.<qid>`, which takes one of three shapes:

- `{"type": "noul", "noul": p_true}`
- `{"type": "choice", "choice": "...", "confidence": c, "probabilities": {option: p}}`
- `{"type": "score", "score": s, "confidence": c, "legend": {"0": "..."}, "probabilities": {"0": p}}`

It also contains `usage: {input_tokens, output_tokens, cost}` and a resolved `model` id. For `choice` and
`score`, `probabilities` is documented as optional. Any number of questions fit in one call. The context
limit is 32k tokens. The model list comes from `GET /api/v1/models?output_modalities=decisions`
(`typesafe/jev-1.13`, `~typesafe/jev-latest`).

### Chat models: Chat Completions

`POST https://openrouter.ai/api/v1/chat/completions`. The request uses
`response_format: {type: "json_schema", json_schema: {name, strict: true, schema}}`, `temperature: 0`,
`reasoning: {enabled: false}` and `provider: {require_parameters: true}`. `usage.cost` (USD) and the token
counts arrive in every response. Neither Claude nor GPT-5.x exposes `logprobs` on OpenRouter, so LLM
probabilities are **verbalized**: the model writes them as numbers in JSON (see §9, caveats).

### Catalog

`GET https://openrouter.ai/api/v1/models` is public and needs no key. Per model it returns `id`, `name`,
`pricing.prompt` and `pricing.completion` (USD per token), `context_length` and `supported_parameters`.

## 3. Configuration: structured TOML in `config/`

All three files are validated by pydantic at load time. A run or generation stores a snapshot of the config
it used, so later edits never change the meaning of old results.

### `config/questions.toml`: the single source of truth for questions

Types mirror Jev primitives. Option ids are snake_case and every option carries a description. For
`score`, the document order of `options` is the scale order, lowest first. `noul` options are exactly
`yes` and `no`. `instructions` must be self-contained, because Jev never sees question ids.

| id | type | question (gist) |
|---|---|---|
| `category` | choice ×20 | personal, work_internal, work_external, news, marketing, spam, phishing, scam, transactional, shipping, billing, account_security, calendar, social, recruiting, system_alert, support, hr_legal, travel, government |
| `urgency` | score ×5 | no_action → whenever → this_week → today → immediately |
| `importance` | score ×4 | trivial → low → moderate → high |
| `needs_reply` | noul | does the sender expect a reply? |
| `action_required` | noul | does it ask for an action other than replying (pay, sign, click, attend)? |
| `skippable` | noul | can the recipient safely not read it? |
| `should_delete` | noul | should it be deleted? |
| `llm_safe` | noul | is it safe to pass verbatim to another LLM or agent (no prompt injection or instructions aimed at AI)? |
| `malicious` | noul | phishing, scam or malware attempt? (also checks each model's own consistency with `category`) |
| `sensitive_data` | noul | does it contain PII, credentials or financial details? |
| `sentiment` | choice ×3 | negative, neutral, positive |

### `config/generation.toml`

- `models`: generator mix used round-robin. Default: `google/gemini-3.8-flash`,
  `deepseek/deepseek-v4.1-flash`, `z-ai/glm-5.3`. None of these are benchmarked, which avoids
  self-preference bias.
- `temperature` (default 1.0), `concurrency` (default 8), `sent_at_window_days` (default 30).
- `system_prompt` and `user_prompt`: `string.Template` strings (`$name` placeholders, so JSON braces
  need no escaping). The available placeholders are `$questions` (rendered question list), `$sent_at`,
  `$<trait>` and `$<trait>_prompt` for every trait.
- `[[traits]]`: the knobs that make the dataset diverse. Each trait has a `name` and either
  - `question = "<qid>"`: values are that question's option ids, and `$<trait>_prompt` is the option
    description; or
  - `[traits.values.<id>]` with `weight` and `prompt` text.

  `stratify = true` cycles values evenly instead of sampling them by weight. Defaults: `category`
  (question-linked, stratified), `urgency` (question-linked), `length` (short / medium / long),
  `prompt_injection` (none 90 % / injected 10 %, with an instruction to embed a covert instruction aimed
  at an AI assistant).

### `config/benchmark.toml`

- `[llm]`: `system_prompt` template (`$questions`), `temperature = 0`, `reasoning = {enabled = false}`,
  `cache_system_prompt = true` (adds an Anthropic `cache_control` breakpoint on the system prompt; OpenAI
  caches automatically), `concurrency = 8`.
- `[[columns]]`: `id`, `title`, `kind` (`decisions` | `chat`), a catalog filter (`modality = "decisions"`
  or `prefix = "anthropic/"`) and `default_model`. The defaults are:
  - `jev` → `typesafe/jev-1.13`
  - `anthropic` → `anthropic/claude-sonnet-5`
  - `openai` → `openai/gpt-5.6-terra`

### Environment (`.env`, via pydantic-settings)

`OPENROUTER_API_KEY` (required only to generate or run), `JEV_BENCH_DATA_DIR` (default `./data`),
`JEV_BENCH_CONFIG_DIR` (default `./config`), `JEV_BENCH_REQUEST_TIMEOUT_S` (default 60),
`JEV_BENCH_MAX_RETRIES` (default 3).

## 4. Data model and storage (JSON/JSONL files under `data/`)

```
data/
  generations/<gen_id>/generation.json   # GenerationMeta
  generations/<gen_id>/emails.jsonl      # one Email per line, appended as generated
  runs/<run_id>/run.json                 # RunMeta
  runs/<run_id>/predictions.jsonl        # one Prediction per email, appended as answered
  labels/<gen_id>.json                   # {email_id: {question_id: option_id}}, human labels
```

- **Ids:** `gen_id` is `YYYYMMDD-HHMMSS-<name-slug>-<4hex>` and `run_id` is
  `YYYYMMDD-HHMMSS-<column>-<model-slug>-<4hex>`. The UTC timestamp keeps them in chronological order, and
  the random suffix prevents collisions.
- **Email:** `id` (`<gen_id>.<index:04d>`), `sent_at` (ISO 8601, sampled by the plan), `sender {name,
  address}`, `to [{name, address}]`, `cc [...]`, `subject`, `body`, `generator_model`, `traits {name:
  value}`, `reference_answers {question_id: option_id}`.
- **What a benchmarked model sees:** only `{from, to, cc, subject, body}`, built by a single function
  `Email.to_state()`. `sent_at`, `traits`, `reference_answers` and `generator_model` are never sent.
- **GenerationMeta:** `id`, `name`, `created_at`, `status`, `requested`, `done`, `errors`, `seed`, snapshots
  of the question set and generation config, `total_cost`, `duration_s`, `trait_mismatches` (how often the
  generator's own answer differs from a question-linked trait it was asked for).
- **RunMeta:** `id`, `column`, `kind`, `model` (requested), `resolved_models` (as returned by OpenRouter;
  matters for `~…-latest` aliases), `generation_ids`, snapshots of the question set and LLM params,
  `concurrency`, `status`, `created_at`, `finished_at`, `duration_s` (wall clock), `n_emails`, `n_done`,
  `n_errors`, `total_cost`, `input_tokens`, `output_tokens`, `latency_p50_ms`, `latency_p95_ms`.
- **Prediction:** `email_id`, `answers {question_id: {option_id: p}} | null`, `error | null`,
  `latency_ms`, `cost`, `input_tokens`, `output_tokens`, `resolved_model`, `raw` (the full response body).
- **Status lifecycle** (runs and generations): `running` → `completed` | `cancelled` | `failed`. On server
  start, any `running` without a live task becomes `interrupted`, and its totals are recomputed from the
  JSONL. The output is the local filesystem itself, so these markers are authoritative.
- **Writes:** JSONL lines are appended as each result arrives, so a crash loses at most in-flight
  requests. `*.json` files are written atomically (temp file + `os.replace`). Labels are guarded by an
  `asyncio.Lock`.
- `data/` is **tracked in git**: results, generations and labels are meant to be kept and shared.

## 5. Canonical answers

Every answer from any source is normalized to a **distribution** `{option_id: p}` over the question's
option ids, with p ∈ [0, 1] and Σ = 1:

| source | noul | choice | score |
|---|---|---|---|
| Jev | `{yes: noul, no: 1 − noul}` | `probabilities`; missing → one-hot on `choice` | `probabilities` indexed `"0"…` mapped to level ids in order; missing → one-hot at `round(score)` |
| LLM | schema field is P(yes) → `{yes: p, no: 1 − p}` | object with one number per option → clip < 0 to 0, renormalize; Σ = 0 → per-question parse error | same as choice |
| reference / human | one-hot | one-hot | one-hot |

When a fallback was used, the prediction records it in a `notes` list.

## 6. Components (`src/jev_bench/`, one responsibility per module)

| module | responsibility |
|---|---|
| `settings.py` | env settings (pydantic-settings) |
| `questions.py` | question-set models (discriminated union), TOML loader, question rendering for prompts |
| `emails.py` | `Email` model and `to_state()` |
| `openrouter.py` | shared `httpx.AsyncClient` wrapper: auth, timeouts, retry with jittered backoff on 429/5xx/524/529/transport errors, typed `OpenRouterError(status, fatal)`; 401/402/403 are fatal |
| `catalog.py` | fetch and filter the OpenRouter catalog per column (`structured_outputs` required, `:batch` excluded), in-memory TTL cache (1 h) |
| `classifiers/base.py` | `Classifier` protocol, `Classification` result model |
| `classifiers/jev.py` | Decisions request builder and response → distributions |
| `classifiers/llm.py` | chat request builder (prompt + JSON schema) and response → distributions |
| `generation/config.py` | `generation.toml` models and loader |
| `generation/plan.py` | deterministic trait plan from a seed (stratified or weighted), `sent_at` sampling |
| `generation/prompt.py` | template rendering, generator JSON schema (`email` + `answers` with per-question enums) |
| `generation/generator.py` | executes one generation: concurrency, append emails, finalize meta |
| `store/jsonfiles.py` | atomic JSON write, JSONL append and read |
| `store/generations.py`, `store/runs.py`, `store/labels.py` | persistence per entity |
| `jobs.py` | registry of background asyncio tasks (runs and generations): live progress, cancel, graceful shutdown |
| `runner.py` | executes one benchmark run: semaphore concurrency, per-email classification, append predictions, finalize totals |
| `metrics/distributions.py` | argmax, entropy, Jensen–Shannon divergence, expected score |
| `metrics/agreement.py` | percent agreement, Cohen's κ (nominal and quadratic-weighted), Fleiss' κ, Pearson r, Brier score |
| `metrics/bootstrap.py` | vectorized percentile bootstrap CI (numpy, fixed seed) |
| `compare.py` | build raters (runs, reference, human) and the comparison report per question and per email |
| `web/app.py` | FastAPI factory, lifespan (client, jobs, interrupted sweep), static mount |
| `web/routes/*.py` | `catalog`, `generations`, `runs`, `compare`, `emails`, `labels` routers |
| `web/static/` | vanilla HTML + ES modules + CSS |
| `cli.py` | typer app: `serve`, `generate`, `run` (lazy imports: `--help` < 500 ms) |

Dependencies point one way: `web` / `cli` → `runner` / `generation` / `compare` → `classifiers` /
`store` / `metrics` → `questions` / `emails` / `settings`. `metrics` is pure numpy with no I/O.

## 7. Flows

### Generation

`POST /api/generations {name, count, seed?, models?}` or `jev-bench generate --count 200 [--name]
[--seed] [--model …]`. `count` must be between 1 and 2000, and the seed defaults to a random value that
is recorded in the meta.

1. Create `<gen_id>` and write `generation.json` (`running`).
2. Build the trait plan.
3. For each plan item, with bounded concurrency: render the prompts, call the generator (round-robin
   model) with the JSON schema `{email, answers}`, validate, and append the `Email`.
4. Finalize the meta.

A failed item is counted in `errors` and logged, and the generation continues. Fatal HTTP errors abort it
(`failed`).

### Benchmark run

`POST /api/runs {column, model, generation_ids}` or `jev-bench run --column … --generations …`:

1. Load the emails as the union of the chosen generations.
2. Write `run.json` (`running`).
3. Make one classification call per email **with all questions** (Jev's recommended usage; LLMs get the
   same granularity, so cost and latency compare like-for-like), under a semaphore (`concurrency`).
4. Append each `Prediction`: a per-email failure fills `error` and the run continues; a fatal error ends
   the run as `failed`.
5. Finalize the totals.

Duration is wall clock (`perf_counter`) from the first request to the last result. Cancel →
`cancelled`.

### Live progress

`GET /api/runs/{id}` and `GET /api/generations/{id}` merge the persisted meta with in-memory job progress
(`done`, `errors`, `cost_so_far`, `elapsed_s`). The UI polls every second while `running`.

## 8. Comparison and metrics

A **rater** is any source of distributions over `(email_id, question_id)`:

- a run (one rater per run);
- `reference`, the generator's answers, always present for generated emails;
- `human`, included only when at least one compared email has a label.

Metrics are computed per question over the emails both raters answered (pairwise-complete), for questions
whose option ids match across the raters' question-set snapshots (mismatches are skipped with a warning).

- **Per rater:** `n`, argmax counts per option (for example, "emails called `spam`"; for noul, the count of
  P(yes) ≥ 0.5), mean distribution, mean entropy, mean max-probability (confidence). For score questions
  also the mean expected level.
- **Pairwise:**
  - top-1 agreement;
  - Cohen's κ (quadratic-weighted for score questions);
  - mean JSD (base 2, in [0, 1]);
  - Pearson r of P(yes) (noul) or of the expected level (score); `null` when variance is 0;
  - 95 % bootstrap CIs (1000 resamples) for agreement and κ;
  - against a hard-label rater (reference or human), also the multi-class Brier score.
- **Group:** Fleiss' κ over the compared model runs, on emails that all of them answered. It is `null`
  when fewer than two runs are compared.
- **Per email:** disagreement index = mean pairwise JSD across model raters, averaged over questions. It
  sorts the Explorer so the most contested emails come first.
- Argmax ties resolve to the first option in question order, which is deterministic.

## 9. Caveats (made explicit, not solved)

- Verbalized LLM probabilities are not calibrated the way Jev claims to be. Argmax-based metrics (agreement,
  κ) are the primary comparison. JSD and Brier also reflect each model's calibration style.
- `sent_at` is generated and stored but never sent to the models (explicit requirement). Relative dates in
  the body ("by tomorrow") remain interpretable.
- Reference answers come from the generator LLM: they are a weak reference, not ground truth.
- Prompt caching lowers LLM cost on repeated system prompts. It is on by default because it reflects real
  deployment, and it is recorded in the run's param snapshot.

## 10. Web UI (vanilla HTML + ES modules, no build step)

Bootstrap 5.3 (built-in dark mode), Bootstrap Icons, Chart.js 4, and the Inter and JetBrains Mono fonts
(Google Fonts), all from a CDN with pinned versions. A shared `layout.js` injects the navbar. All
email-derived text is rendered with `textContent` or escaping, never `innerHTML`: emails are untrusted and
some contain deliberate injections.

| page | content |
|---|---|
| **Benchmark** `/` | generation multi-select. Three column cards (Jev / Anthropic / OpenAI), each with: model `<select>` from the catalog (showing price per 1M in/out), Run and Cancel, progress bar, live elapsed time, cost, tokens, errors, p50/p95 latency. Below: the comparison of the latest completed run per column whose `generation_ids` set equals the selected set (overridable with `?runs=a,b,c`): a summary table, then per question an argmax-count bar chart (raters side by side), pairwise metric tables with CIs, Fleiss' κ and per-rater entropy. |
| **Generations** `/generations.html` | list (name, created, count, models, cost, status, mismatches), a "New generation" form (name, count, seed, models) with live progress |
| **Explorer** `/explorer.html` | select generations and optional runs; filter by generator model, reference answer, trait or text search; sort by disagreement. A detail panel shows headers, `sent_at`, body, traits, reference answers, each rater's distribution per question (mini bars) and a manual labelling form. |
| **Runs** `/runs.html` | history table with checkboxes → "Compare" (Benchmark `?runs=`) and "Explore" (Explorer `?runs=`) |

## 11. HTTP API (`/api`)

| method + path | purpose |
|---|---|
| `GET /catalog` | columns with their model lists and defaults |
| `GET /generations` · `POST /generations` · `GET /generations/{id}` · `POST /generations/{id}/cancel` | generations |
| `GET /runs?generations=` · `POST /runs` · `GET /runs/{id}` · `POST /runs/{id}/cancel` | runs |
| `GET /compare?runs=a,b,c` | comparison report (reference always included; human if any labels) |
| `GET /emails?generations=…&runs=…` | email rows, plus per-rater argmax and a disagreement index when runs are given |
| `GET /emails/{email_id}?runs=…` | full email, reference, predictions, human label |
| `PUT /labels/{email_id}` | set or clear human answers `{question_id: option_id \| null}` |

A missing `OPENROUTER_API_KEY` returns 400 on `POST /generations` and `POST /runs`. Catalog, browsing and
comparison work without a key.

## 12. Testing

- pytest with one test file per module. OpenRouter is mocked with `httpx.MockTransport`, and routes are
  tested through `httpx.ASGITransport`. No real network in the default suite.
- Hypothesis properties for the metrics and normalization: JSD symmetric and in [0, 1] with JSD(p, p) = 0;
  κ(a, a) = 1; agreement in [0, 1]; normalized distributions sum to 1.
- Known-value tests: Cohen's κ against a textbook confusion matrix, Fleiss' κ against the Wikipedia worked
  example (0.210).
- An opt-in `@pytest.mark.integration` smoke test calls the real OpenRouter for one email per column (needs
  a key).
- Gates: `ruff check`, `ruff format`, `pyright`, `pytest --cov --cov-fail-under=95`. Default suite
  < 5 s. The UI is smoke-tested manually in a browser.

## 13. Stack

Python ≥ 3.14 with uv. FastAPI + uvicorn, httpx (direct, no OpenAI SDK: the Decisions API isn't in it,
it's cheaper to import, and it can be mocked with `MockTransport`), pydantic v2 + pydantic-settings, typer,
loguru + rich, numpy. Dev: pytest, pytest-cov, hypothesis, ruff, pyright.
