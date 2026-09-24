# Jev Benchmark — Design

- **Date:** 2026-09-24
- **Status:** draft, awaiting review

## 1. Purpose

Benchmark **Jev** (`typesafe/jev-1.13`, TypeSafe's decision model on OpenRouter) on email triage against
three alternatives: a frontier Anthropic chat model, a frontier OpenAI chat model, and a zero-shot
**embedding-similarity** baseline. Every column gets the same emails and the same typed questions and
returns a probability for every answer option. There is no ground truth: the benchmark measures **speed,
cost and how closely the columns agree** with each other, with the generator's reference answers and, when
present, with optional human labels.

### Success criteria

1. `jev-bench generate` (CLI) and the Generations page (UI) each produce a separate, persisted generation of
   synthetic emails with reference answers.
2. The Benchmark page runs any subset of generations through four columns (Jev / Anthropic / OpenAI /
   Embeddings).
   - Each column's model is picked from a list filled live from the OpenRouter catalog.
   - The chat columns offer two modes: **per email** (one prompt per email) and **all in one** (every
     selected email in a single prompt).
   - Before any request is sent, its token size is checked against the model's limits. A multi-email
     request that doesn't fit is split into the minimal number of equal parts that do fit.
   - Embeddings are computed once per (embedding model, text) and reused across runs.
   - The page shows live progress, wall-clock duration and total USD cost.
3. Every run is persisted and can be reopened and compared with any other run later.
4. Comparison shows per-column answer distributions (e.g. "how many emails each model called `spam`"),
   pairwise agreement metrics with confidence intervals, and group agreement.
5. The Explorer shows every email with the reference answers and each run's distributions side by side, and
   allows manual labelling.
6. Everything works with **zero human labels**: human-based metrics appear only once labels exist.
7. Correct with no network: the test suite mocks every OpenRouter call; coverage ≥ 95 %.

### Non-goals (foundation)

Auth / multi-user; resuming interrupted runs; deleting runs or generations from the UI; editing TOML config
from the UI; logprob-based probabilities; Jev's System One API; the OpenRouter Batch API (explicitly
excluded).

## 2. External contracts (verified 2026-09-24)

All calls go to OpenRouter with `Authorization: Bearer $OPENROUTER_API_KEY`. Every response carries
`usage.cost` in USD. If it is ever missing, cost is estimated as tokens × catalog pricing and the prediction
is flagged `cost_estimated`.

### Jev: Decisions API

`POST https://openrouter.ai/api/alpha/decisions`

```json
{
  "model": "typesafe/jev-1.13",
  "state": {"sent_at": "2026-09-12T09:41:00+04:00", "from": "...", "to": ["..."], "cc": [], "subject": "...", "body": "..."},
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
`score`, `probabilities` is documented as optional. Any number of questions fit in one call, but only one
`state` (one email) per call. The context limit is 32k tokens.

### Chat models: Chat Completions

`POST https://openrouter.ai/api/v1/chat/completions`. The request uses
`response_format: {type: "json_schema", json_schema: {name, strict: true, schema}}`, `temperature: 0`,
`reasoning: {enabled: false}` and `provider: {require_parameters: true}`. Neither Claude nor GPT-5.x
exposes `logprobs` on OpenRouter, so LLM probabilities are **verbalized**: the model writes them as numbers
in JSON (see §9, caveats). *All in one* requests are streamed (`stream: true`, SSE) to survive long
generations and to report progress. The final chunk carries `usage`. The default models allow at most
128k output tokens (`top_provider.max_completion_tokens`).

### Embeddings

`POST https://openrouter.ai/api/v1/embeddings` with `{model, input: [str, ...]}` returns
`{data: [{index, embedding}], model, usage}`. One request embeds many texts.

### Catalog

`GET https://openrouter.ai/api/v1/models` (text models) and `?output_modalities=decisions` or
`?output_modalities=embeddings` are public and need no key. Per model they return `id`, `name`,
`pricing.prompt` and `pricing.completion` (USD per token), `context_length`,
`top_provider.max_completion_tokens` and `supported_parameters`. The limits used by the token budget
(§7) come from here. Observed values:

| model | context | max output |
|---|---|---|
| `typesafe/jev-1.13` | 32 000 (total) | n/a |
| `anthropic/claude-sonnet-5` | 1 000 000 | 128 000 |
| `openai/gpt-5.6-terra` | 1 050 000 | 128 000 |
| `openai/text-embedding-3-large` | 8 192 per input text | n/a |

There is no public token-count endpoint, so request sizes are **estimated** (§7, token budget).

## 3. Configuration: structured TOML in `config/`

All three files are validated by pydantic at load time. A run or generation stores a snapshot of the config
it used, so later edits never change the meaning of old results.

### `config/questions.toml`: the single source of truth for questions

Types mirror Jev primitives. Option ids are snake_case.

- **Every option has a non-empty description,** including the `yes` and `no` of `noul` questions. The
  embedding column embeds these descriptions.
- For `score`, the document order of `options` is the scale order, lowest first.
- `noul` options are exactly `yes` and `no`.
- `instructions` must be self-contained, because Jev never sees question ids.

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

- `[jev]`: `concurrency = 8`.
- `[llm]`, shared by both chat columns:
  - `system_prompt` (per-email mode) and `system_prompt_all_in_one`: templates using `$questions`;
  - `temperature = 0`, `reasoning = {enabled = false}`;
  - `cache_system_prompt = true`: adds an Anthropic `cache_control` breakpoint on the system prompt;
    OpenAI caches automatically;
  - `concurrency = 8` (per-email mode);
  - `all_in_one_timeout_s = 1800`;
  - `est_output_tokens_per_email = 400`: the output-side estimate for the token budget.
- `[embeddings]`:
  - `email_template` (default
    `"Sent: $sent_at\nFrom: $from\nTo: $to\nCc: $cc\nSubject: $subject\n\n$body"`);
  - `option_template` (default `"$instructions $description"`);
  - `temperature = 0.05` (softmax τ);
  - `emails_per_request = 32`, `max_request_tokens = 100000`, `concurrency = 4`.
- `[tokens]`: `bytes_per_token = 3.0` (conservative estimator, §7), `jev_output_reserve = 1000`, and
  fallback `context_length` and `max_completion_tokens` for models the catalog doesn't describe.
- `[[columns]]`: `id`, `title`, `kind` (`decisions` | `chat` | `embeddings`), a catalog filter
  (`modality = "decisions"`, `modality = "embeddings"` or `prefix = "anthropic/"`) and
  `default_model`. The defaults are:
  - `jev` → `typesafe/jev-1.13`
  - `anthropic` → `anthropic/claude-sonnet-5`
  - `openai` → `openai/gpt-5.6-terra`
  - `embeddings` → `openai/text-embedding-3-large` (strong alternative in the catalog:
    `qwen/qwen3-embedding-8b`, 13× cheaper)

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
  runs/<run_id>/predictions.jsonl        # one Prediction per email
  runs/<run_id>/responses.jsonl          # one line per HTTP request: email_ids, status, latency, body
  labels/<gen_id>.json                   # {email_id: {question_id: option_id}}, human labels
  embeddings/<model-slug>/vectors.jsonl  # embedding cache (gitignored), see below
```

- **Ids:** `gen_id` is `YYYYMMDD-HHMMSS-<name-slug>-<4hex>` and `run_id` is
  `YYYYMMDD-HHMMSS-<column>-<model-slug>-<4hex>`. The UTC timestamp keeps them in chronological order, and
  the random suffix prevents collisions.
- **Email:** `id` (`<gen_id>.<index:04d>`), `sent_at` (ISO 8601, sampled by the plan), `sender {name,
  address}`, `to [{name, address}]`, `cc [...]`, `subject`, `body`, `generator_model`, `traits {name:
  value}`, `reference_answers {question_id: option_id}`.
- **What a benchmarked model sees:** only `{sent_at, from, to, cc, subject, body}`, built by a single
  function `Email.to_state()`. `traits`, `reference_answers`, `generator_model` and the email `id` are
  never sent. Ids contain generation name slugs that could leak hints, so the all-in-one mode addresses
  emails by positional refs (`e001…`).
- **GenerationMeta:** `id`, `name`, `created_at`, `status`, `requested`, `done`, `errors`, `seed`, snapshots
  of the question set and generation config, `total_cost`, `duration_s`, `trait_mismatches` (how often the
  generator's own answer differs from a question-linked trait it was asked for).
- **RunMeta:**
  - identity: `id`, `column`, `kind`, `model` (requested), `resolved_models` (as returned by OpenRouter;
    matters for `~…-latest` aliases), `generation_ids`;
  - request shape: `mode` (`per_email` | `all_in_one` for chat, `per_email` for Jev, `batched` for
    embeddings), `emails_per_request`, snapshots of the question set and column params, `concurrency`;
  - lifecycle: `status`, `created_at`, `finished_at`, `duration_s` (wall clock);
  - totals: `n_emails`, `n_done`, `n_errors`, `n_requests`, `n_splits` (extra requests created by the
    token budget), `total_cost` (actually paid in this run, including `setup_cost`), `setup_cost`
    (embedding the option texts), `input_tokens`, `output_tokens`, `latency_p50_ms` and `latency_p95_ms`
    (over requests);
  - embeddings only: `cache_hits`, `cache_misses`, and `cold_cost` (what the run would have cost with an
    empty cache: paid cost plus the cost originally recorded for every cached vector).
- **Prediction:** `email_id`, `answers {question_id: {option_id: p}} | null`, `error | null`, `notes`,
  `request_index`, `batch_size`, `latency_ms` (of its request), `cost`, `input_tokens` and
  `output_tokens`, `resolved_model`, `similarities {question_id: {option_id: cosine}} | null` (embeddings
  only), `cached` (embeddings only: the email vector came from the cache, so this run paid 0 for it).
  When `batch_size > 1`, cost and tokens are the request's totals split evenly across its emails.
- **responses.jsonl** stores each raw response once per request instead of once per email. Embedding vectors
  are stripped from it; they live only in the cache.
- **Embedding cache** (`data/embeddings/<model-slug>/vectors.jsonl`, append-only):
  - one line per text: `key` (sha256 of the exact rendered input text), `kind` (`option` | `email`),
    `ref` (`<qid>:<option>` or the email id), `resolved_model`, `dim`, `vector` (base64 float32
    little-endian), `cost` and `input_tokens` (the text's share of its original request), `created_at`;
  - the key covers the rendered text, so editing a template or a description invalidates exactly the
    affected entries;
  - loaded into memory at run start; appends are guarded by a per-model `asyncio.Lock`; a duplicate key
    (two concurrent runs) is harmless, and the last line wins;
  - it is derived, large (~16 KB per 3072-dim vector) and regenerable, so it is **gitignored**.
- **Status lifecycle** (runs and generations): `running` → `completed` | `cancelled` | `failed`. On server
  start, any `running` without a live task becomes `interrupted`, and its totals are recomputed from the
  JSONL. The output is the local filesystem itself, so these markers are authoritative.
- **Writes:** JSONL lines are appended as each result arrives, so a crash loses at most in-flight
  requests. `*.json` files are written atomically (temp file + `os.replace`). Labels are guarded by an
  `asyncio.Lock`.
- `data/` is **tracked in git** (except `data/embeddings/`): results, generations and labels are meant to
  be kept and shared.

## 5. Canonical answers

Every answer from any source is normalized to a **distribution** `{option_id: p}` over the question's
option ids, with p ∈ [0, 1] and Σ = 1:

| source | noul | choice | score |
|---|---|---|---|
| Jev | `{yes: noul, no: 1 − noul}` | `probabilities`; missing → one-hot on `choice` | `probabilities` indexed `"0"…` mapped to level ids in order; missing → one-hot at `round(score)` |
| LLM (both modes) | schema field is P(yes) → `{yes: p, no: 1 − p}` | object with one number per option → clip < 0 to 0, renormalize; Σ = 0 → per-question parse error | same as choice |
| Embeddings | `softmax(cos / τ)` over `yes`, `no` | `softmax(cos / τ)` over options | `softmax(cos / τ)` over levels |
| reference / human | one-hot | one-hot | one-hot |

When a fallback was used, the prediction records it in `notes`.

In all-in-one mode the response schema is `{results: [{ref, <per-email answers>}]}`, with `ref` restricted
to an enum of the refs that were sent. The schema's size does not depend on how many emails are in the
prompt. Parsing rules:

- A ref missing from the response → that email gets an error.
- A duplicated ref → the first occurrence wins, with a note.
- A truncated or unparseable response (`finish_reason = "length"`, invalid JSON) or a timeout → every
  email in the request gets that error.

## 6. Components (`src/jev_bench/`, one responsibility per module)

| module | responsibility |
|---|---|
| `settings.py` | env settings (pydantic-settings) |
| `questions.py` | question-set models (discriminated union), TOML loader, question rendering for prompts |
| `emails.py` | `Email` model and `to_state()` |
| `openrouter.py` | shared `httpx.AsyncClient` wrapper: auth, timeouts, JSON and SSE-stream POST, retry with jittered backoff on 429/5xx/524/529/transport errors, typed `OpenRouterError(status, fatal)`; 401/402/403 are fatal |
| `catalog.py` | fetch and filter the OpenRouter catalog per column (chat: `structured_outputs` required, `:batch` excluded), in-memory TTL cache (1 h), `ModelInfo` with pricing, limits and a cost estimate |
| `classifiers/base.py` | `Classifier` protocol: `emails_per_request`, `prepare()` (one-off setup), `classify(emails) -> RequestResult` (per-email answers or errors, plus the request's usage, latency and raw body) |
| `classifiers/jev.py` | Decisions request builder and response → distributions |
| `classifiers/llm_schema.py` | JSON schema builders (single email; `results[]` array for all-in-one) |
| `classifiers/llm.py` | chat classifier for both modes: prompt rendering, request, parsing and normalization |
| `classifiers/embeddings.py` | option vectors in `prepare()` and email vectors per request, both cache-first (only misses are sent); cosine → softmax(τ) |
| `tokens.py` | tokenizer-free, conservative token estimate for a text; per-model budgets from catalog limits and config fallbacks |
| `request_plan.py` | pure function: chunks of emails + per-email token sizes + budget → requests (minimal equal split, oversize singles flagged) |
| `generation/config.py` | `generation.toml` models and loader |
| `generation/plan.py` | deterministic trait plan from a seed (stratified or weighted), `sent_at` sampling |
| `generation/prompt.py` | template rendering, generator JSON schema (`email` + `answers` with per-question enums) |
| `generation/generator.py` | executes one generation: concurrency, append emails, finalize meta |
| `store/jsonfiles.py` | atomic JSON write, JSONL append and read |
| `store/generations.py`, `store/runs.py`, `store/labels.py` | persistence per entity |
| `store/embeddings.py` | embedding cache per model: load, lookup by text hash, locked append |
| `jobs.py` | registry of background asyncio tasks (runs and generations): live progress, cancel, graceful shutdown |
| `runner.py` | executes one benchmark run: `prepare()`, build the request plan, run requests under a semaphore, split costs across emails, append results, finalize totals |
| `metrics/distributions.py` | argmax, entropy, Jensen–Shannon divergence, expected score, softmax |
| `metrics/agreement.py` | percent agreement, Cohen's κ (nominal and quadratic-weighted), Fleiss' κ, Pearson r, Brier score |
| `metrics/bootstrap.py` | vectorized percentile bootstrap CI (numpy, fixed seed) |
| `compare.py` | build raters (runs, reference, human) and the comparison report per question and per email |
| `web/app.py` | FastAPI factory, lifespan (client, jobs, interrupted sweep), static mount |
| `web/routes/*.py` | `catalog`, `generations`, `runs`, `compare`, `emails`, `labels` routers |
| `web/static/` | vanilla HTML + ES modules + CSS |
| `cli.py` | typer app: `serve`, `generate`, `run` (lazy imports: `--help` < 500 ms) |

Dependencies point one way: `web` / `cli` → `runner` / `generation` / `compare` → `classifiers` /
`store` / `metrics` → `questions` / `emails` / `catalog` / `openrouter` / `settings`. `metrics` is pure
numpy with no I/O.

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

`POST /api/runs {column, model, generation_ids, mode?}` or `jev-bench run --column … --generations …
[--mode per_email|all_in_one]`:

1. Load the emails as the union of the chosen generations.
2. Write `run.json` (`running`).
3. Call `prepare()`. For embeddings this resolves the option vectors from the cache and embeds only the
   misses; the cost paid is recorded as `setup_cost`.
4. Chunk the emails by `emails_per_request`:
   - Jev: 1 (one call per email with all questions, Jev's recommended usage);
   - chat per-email: 1;
   - chat all-in-one: every email;
   - embeddings: `[embeddings].emails_per_request`, counting cache misses only. Cached emails need no
     request.
5. Apply the **token budget** (below) to every chunk, which produces the final request list.
6. Run the requests under a semaphore (`concurrency`).
7. Append a `Prediction` for each email of a finished request. A per-email failure fills `error` and the
   run continues; a fatal error ends the run as `failed`.
8. Finalize the totals.

Duration is wall clock (`perf_counter`) from the first request to the last result. Cancel →
`cancelled`.

### Token budget (checked before every request)

**Estimate.** `tokens(text) = ceil(utf8_bytes(text) / bytes_per_token)`, with `bytes_per_token = 3.0`.
There is no tokenizer; it is deliberately pessimistic:

- English runs at ~4 characters per token, so the estimate overshoots by about a third;
- non-Latin scripts are multi-byte, so they are overestimated rather than underestimated.

A request's size is the estimate of everything sent: the rendered system prompt plus the serialized
user payload.

**Budget per kind:**

| kind | input must be ≤ | output must be ≤ |
|---|---|---|
| chat | `context_length − max_completion_tokens` (`max_tokens` is sent as `max_completion_tokens`) | `max_completion_tokens`, estimated as `n_emails × est_output_tokens_per_email` |
| Jev | `context_length − jev_output_reserve` (32 000 total, questions included) | n/a |
| embeddings | each text ≤ `context_length`; request total ≤ `max_request_tokens` | n/a |

**Split rule** (`request_plan.py`, pure and table-tested):

1. If a chunk fits, send it as is.
2. Otherwise start with `k = ceil(needed / budget)` using the tighter of the input and output ratios.
   Example: a 100 000 budget and 110 000 needed gives `k = 2`, two requests of ~55 000.
3. Cut the chunk into `k` **contiguous parts balanced by token size** (greedy against the running target
   `remaining / parts_left`). Email order is preserved.
4. If uneven email sizes still leave a part over budget, increase `k` by 1 and repeat. This terminates at
   one email per part.
5. A single email that doesn't fit on its own gets a per-email error ("exceeds model limits") and is
   **never sent or truncated**: truncating would feed different input to different columns.

Splits are counted in `n_splits` and shown on the column card.

### Live progress

`GET /api/runs/{id}` and `GET /api/generations/{id}` merge the persisted meta with in-memory job progress
(`done`, `errors`, `cost_so_far`, `elapsed_s`). The UI polls every second while `running`. In all-in-one
mode, `done` is estimated while streaming by counting completed `ref` entries in the received text, and it
becomes exact once parsing finishes.

## 8. Comparison and metrics

A **rater** is any source of distributions over `(email_id, question_id)`:

- a run (one rater per run, whatever its column or mode);
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
- **Group:** Fleiss' κ over the compared runs, on emails that all of them answered. It is `null` when fewer
  than two runs are compared.
- **Per email:** disagreement index = mean pairwise JSD across the compared runs, averaged over questions.
  It sorts the Explorer so the most contested emails come first.
- Argmax ties resolve to the first option in question order, which is deterministic.

Comparing a chat model's *per email* and *all in one* runs is an ordinary pairwise comparison between two
raters. It directly measures how much packing emails into one prompt changes the answers.

## 9. Caveats (made explicit, not solved)

- **Verbalized LLM probabilities** are not calibrated the way Jev claims to be. Argmax-based metrics
  (agreement, κ) are the primary comparison. JSD and Brier also reflect each model's calibration style.
- **Embeddings measure topical similarity, not entailment.** They are expected to be weak on yes/no
  questions and negations; that is the point of the baseline. τ only reshapes the distributions (argmax
  metrics do not depend on it), and it is fixed in config. Tuning τ on reference answers is forbidden,
  because generator answers must never feed the benchmark. Some option texts may be close to every email
  ("hubness"); per-option centering is a possible later knob.
- **All in one** lets emails influence each other and adds position effects. When the token budget splits
  it, "all in one" really means "the fewest equal prompts that fit". With the default models the binding
  limit is output (128k ≈ 300 emails at ~400 tokens each), and `n_splits` makes this visible. One failure
  (truncation, timeout) fails every email in that request. Per-email latency is meaningless in this mode;
  only the run duration and the request latency are reported.
- **Token estimates are heuristic.** Overestimating only causes an earlier split. An underestimate, e.g.
  for text much denser than 3 bytes per token, would surface as a provider 400 or `finish_reason =
  "length"` recorded as request errors. There is no automatic re-split after a failed request.
- **The embedding cache makes repeat runs nearly free and instant.** Cost comparisons therefore show both
  the paid `total_cost` and `cold_cost`, plus the cache-hit rate. Wall-clock duration is not
  reconstructable for cached work, so it is reported as measured.
- `sent_at` is sent to every column as part of the email (explicit requirement). Models have no separate
  notion of "now", so urgency is judged relative to `sent_at` and the body.
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
| **Benchmark** `/` | See below. |
| **Generations** `/generations.html` | list (name, created, count, models, cost, status, mismatches), a "New generation" form (name, count, seed, models) with live progress |
| **Explorer** `/explorer.html` | select generations and optional runs; filter by generator model, reference answer, trait or text search; sort by disagreement. A detail panel shows headers, `sent_at`, body, traits, reference answers, each run's distribution per question (mini bars; embeddings also show cosines) and a manual labelling form. |
| **Runs** `/runs.html` | history table (column, model, mode, generations, status, duration, cost, errors) with checkboxes → "Compare" (Benchmark `?runs=`) and "Explore" (Explorer `?runs=`) |

The Benchmark page has three parts:

- A generation multi-select.
- Four column cards (Jev / Anthropic / OpenAI / Embeddings; 4-up on wide screens, 2×2 on medium). Each
  card has a model `<select>` from the catalog (showing price per 1M in/out), Run and Cancel, progress bar,
  live elapsed time, cost, tokens, errors, number of requests (with `n_splits`) and p50/p95 request
  latency.
  - The chat cards add a mode toggle (*per email* / *all in one*).
  - The embeddings card also shows τ, the batch size, the cache-hit rate and `cold_cost` next to the paid
    cost.
- The comparison of the latest completed run per column whose `generation_ids` set equals the selected set
  (overridable with `?runs=a,b,c`): a summary table, then per question an argmax-count bar chart (raters
  side by side), pairwise metric tables with CIs, Fleiss' κ and per-rater entropy.

## 11. HTTP API (`/api`)

| method + path | purpose |
|---|---|
| `GET /catalog` | columns with their model lists (pricing, limits) and defaults |
| `GET /generations` · `POST /generations` · `GET /generations/{id}` · `POST /generations/{id}/cancel` | generations |
| `GET /runs?generations=` · `POST /runs` · `GET /runs/{id}` · `POST /runs/{id}/cancel` | runs (`POST` body: `column`, `model`, `generation_ids`, `mode`) |
| `GET /compare?runs=a,b,c` | comparison report (reference always included; human if any labels) |
| `GET /emails?generations=…&runs=…` | email rows, plus per-rater argmax and a disagreement index when runs are given |
| `GET /emails/{email_id}?runs=…` | full email, reference, predictions, human label |
| `PUT /labels/{email_id}` | set or clear human answers `{question_id: option_id \| null}` |

A missing `OPENROUTER_API_KEY` returns 400 on `POST /generations` and `POST /runs`. Catalog, browsing and
comparison work without a key. `mode` is only accepted for chat columns.

## 12. Testing

- pytest with one test file per module. OpenRouter is mocked with `httpx.MockTransport`, including JSON
  bodies, SSE streams and embedding vectors with known cosines. Routes are tested through
  `httpx.ASGITransport`. No real network in the default suite.
- All-in-one parsing is table-driven: complete, missing ref, duplicate ref, truncated (`finish_reason =
  "length"`) and invalid JSON.
- The request planner is table-driven:
  - fits as is;
  - 110k needed against a 100k budget → 2 × ~55k;
  - lumpy sizes forcing `k + 1`;
  - output-bound versus input-bound splits;
  - an oversize single email flagged and never sent;
  - order preserved.
- The embedding cache is covered for:
  - hit, miss and partial hit (only misses are sent);
  - key change on a template edit;
  - `cold_cost` accounting;
  - a float32 base64 round-trip.
- Hypothesis properties for the metrics and normalization: JSD symmetric and in [0, 1] with JSD(p, p) = 0;
  κ(a, a) = 1; agreement in [0, 1]; normalized distributions and softmax sum to 1; argmax of softmax is
  invariant to τ.
- Known-value tests: Cohen's κ against a textbook confusion matrix, Fleiss' κ against the Wikipedia worked
  example (0.210).
- An opt-in `@pytest.mark.integration` smoke test calls the real OpenRouter for one email per column and
  mode (needs a key).
- Gates: `ruff check`, `ruff format`, `pyright`, `pytest --cov --cov-fail-under=95`. Default suite
  < 5 s. The UI is smoke-tested manually in a browser.

## 13. Stack

Python ≥ 3.14 with uv. FastAPI + uvicorn, httpx (direct, no OpenAI SDK: the Decisions API isn't in it,
it's cheaper to import, and it can be mocked with `MockTransport`), pydantic v2 + pydantic-settings, typer,
loguru + rich, numpy. Dev: pytest, pytest-cov, hypothesis, ruff, pyright.
