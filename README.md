# jev-bench

A benchmark of **Jev** (`typesafe/jev-1.13`, TypeSafe's decision model) on email triage, against a Claude
chat model, a GPT chat model and an embedding-similarity baseline. Every model answers the same typed
questions about the same synthetic emails and returns a probability for every answer. The benchmark
compares **speed, cost and quality**: how closely each model agrees with the generator's intended answers,
with your own labels and with the other models. All calls go through [OpenRouter](https://openrouter.ai).

## Quick start

```bash
cp .env.example .env        # optional: put OPENROUTER_API_KEY here, or enter a key in the browser later
./run.sh                    # installs uv and Python 3.14 only if missing, then serves http://127.0.0.1:8000
```

With [uv](https://docs.astral.sh/uv/) already installed, `uv sync && uv run jev-bench serve` does the same.

Generations, runs and analyses make **paid** OpenRouter calls: each run card shows an estimated cost
before you start. To get a key: [create one](https://openrouter.ai/keys) and
[add credits](https://openrouter.ai/settings/credits).

## Workflow

| Page | What you do there |
|---|---|
| **Generations** | Create a set of synthetic emails. A seeded plan spreads them over categories, urgency, length and prompt-injection content; several generator models write them and record the answers they intended (the *reference*). |
| **Benchmark** | Run each column on the selected generations and read the comparison. Each card shows the run's quality (mean κ against the reference and your labels), cost and speed; the report below adds per-question statistics, pairwise agreement with bootstrap confidence intervals and Fleiss' κ. |
| **Explorer** | Browse every email with the reference and every run's answer, sort by disagreement, read an email's full text and probabilities, and save your own labels (they become the *Human* rater). |
| **Runs** | Every stored run with its totals; select runs to compare on the Benchmark page or to explore. |
| **Analyze** | Let an LLM analyst read a whole comparison and write a Markdown assessment, with links to the emails it cites. |
| **Help** | What every metric means, the methodology and the live question set. |

## What is measured

**Columns** (`config/benchmark.toml`):

| Column | Kind | Default model |
|---|---|---|
| Jev | Decisions API | `typesafe/jev-1.13` |
| Anthropic | chat, one email per request or all in one | `anthropic/claude-sonnet-5` |
| OpenAI | chat, one email per request or all in one | `openai/gpt-5.6-terra` |
| Embeddings | similarity of the email to each option's text, fixed temperature | `openai/text-embedding-3-large` |
| Kev (optional, swaps with Embeddings in one card) | Decisions API, like Jev | `jaredpalmer/kev-4b` |

**Questions** (`config/questions.toml`, 18 in the shipped set): a multi-label `category`; four ordered
scales (`urgency`, `importance`, `sentiment`, `confidentiality`) that also get a 0–100 score; and thirteen
yes/no questions such as `needs_reply`, `malicious` and `sensitive_data`. Every answer from every source
becomes a probability distribution over the question's options, so all sources are compared the same way.

**Metrics.** The headline is each run's mean κ against the reference, averaged over questions with every
question weighing the same. For ordered scales κ is quadratic, so near misses count less than far ones; for
the multi-label category it is averaged over labels. Beside κ the report shows agreement, JSD, Pearson r,
Brier, Jaccard and F1, as well as time and cost relative to Jev. The Help page explains each one in plain
language.

**Caveats.** The emails are synthetic and the reference is itself a model's judgment, so "agrees with the
reference" means "agrees with the generator". Chat-model probabilities are self-reported. Compare runs on
the same generations, and prefer larger generations for firm conclusions.

## Command line

```bash
uv run jev-bench serve [--host 127.0.0.1] [--port 8000] [--reload]
uv run jev-bench generate [--count 200] [--name generation] [--seed N] [--model ID ...]
uv run jev-bench run COLUMN -g GENERATION_ID [-g ...] [--model ID] [--mode per_email|all_in_one]
```

`COLUMN` is a column id from `config/benchmark.toml` (`jev`, `anthropic`, `openai`, `embeddings`, `kev`). The CLI
and the web UI share the same jobs and data. A run started from the CLI appears in the UI after a page
reload.

## Configuration

Prompts, questions, traits and default models live in `config/*.toml`, not in code. They are re-read at
every run, generation or analysis start, and each run records a snapshot of what it used.

| File | Holds |
|---|---|
| `config/questions.toml` | The question set: ids, types, options and instructions. It is the single source for the Jev payloads, the LLM JSON schemas, the embedding option texts, the metrics and the UI. |
| `config/benchmark.toml` | Columns, chat and embedding prompts, concurrency and token budgets. |
| `config/generation.toml` | Generator models, prompts and the trait plan. |
| `config/analysis.toml` | Analyst model, limits and the prompt templates of the Analyze tab. |

Environment variables (or `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | unset | Server key. A key entered in the browser overrides it for the jobs that browser starts. |
| `JEV_BENCH_DATA_DIR` | `data` | Where generations, runs, labels and analyses are stored. |
| `JEV_BENCH_CONFIG_DIR` | `config` | Where the TOML files are read from. |
| `JEV_BENCH_REQUEST_TIMEOUT_S` | `60` | Per-request timeout for OpenRouter calls. |
| `JEV_BENCH_MAX_RETRIES` | `3` | Retries of an OpenRouter call after a transient error (timeouts, 429, 5xx). |

**API keys are never persisted or logged.** A browser key stays in that browser's local storage and is sent
only on the requests that start a job.

## Data

Everything is plain JSON/JSONL under `data/`:

```
data/generations/<id>/   emails with their reference answers and the question-set snapshot
data/runs/<id>/          run metadata, predictions and raw responses
data/labels/             your human labels
data/analyses/           analyst reports with the exact prompts used
data/embeddings/         vector cache per embedding model (gitignored)
```

When the web server starts, it marks runs and generations that were left `running` as `interrupted`.

## Published snapshot

A read-only snapshot of the committed results is at https://yarnaid.github.io/jev-benchmark/. GitHub
Actions rebuilds it on every push to `main` with `jev-bench export-site`. It shows:
- the Benchmark comparison, for the latest runs with either Embeddings or Kev;
- the Explorer and the saved analyses;
- every step of the label threshold.

It cannot start runs, hold a key or save labels. Run the app locally for that, or to compare any other
combination of runs.

## Development

```bash
uv run pytest                                  # default suite: no network, < 5 s
uv run pytest --cov --cov-fail-under=95        # coverage gate
uv run pytest -m integration                   # real, PAID OpenRouter calls; needs OPENROUTER_API_KEY
uv run ruff check --fix && uv run ruff format && uv run pyright
node --check src/jev_bench/web/static/js/*.js  # JS syntax (Node only, no npm)
node --test tests/js/                          # JS unit tests
```

The UI is build-free: plain ES modules with Bootstrap 5.3 and Chart.js from jsDelivr (with SRI). Email
bodies are treated as hostile, so the DOM is built with `h()` from `js/dom.js` and never from HTML strings.

`CLAUDE.md` describes the architecture and the invariants to keep. The design specs and implementation
plans are in `docs/superpowers/`.
