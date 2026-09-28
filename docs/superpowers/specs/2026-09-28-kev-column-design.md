# Kev column (optional, swappable with Embeddings)

Adds [Kev](https://huggingface.co/spaces/jaredpalmer/kev) as an optional benchmark column. Kev
(`jaredpalmer/kev-4b`, `jaredpalmer/kev-0.8b`, Apache-2.0) is an open reimplementation of TypeSafe's
`/v1/systemone` Decisions contract on Qwen3.5: one state and typed questions in, a probability per option
out, from one forward pass. It is reached through its public Hugging Face Space. On the Benchmark page it
shares one card slot with Embeddings, so the row keeps four cards.

## 1. Decisions

| Topic | Decision |
|---|---|
| Transport | The public HF Space over Gradio's REST API (`https://jaredpalmer-kev.hf.space`). Self-hosted `kev.serve` is out of scope. |
| Client library | Raw REST over the shared `httpx2` client. The alternative, `gradio_client`, was rejected: it depends on `httpx` (the project uses `httpx2`), costs more to import, and wraps only two HTTP calls. |
| Column kind | A new kind `kev`, not a `decisions` column with another URL: `report-summary.js` treats the first `decisions` run as the Jev ×-baseline. |
| Probabilities | Calibrated (`calibrated = true`): the checkpoint's fitted temperature, which is `kev.serve`'s default. It leaves the argmax unchanged (Space README: ECE 0.12 → 0.05 out of domain on Kev-4B). Snapshotted into every run. |
| `date_facts` | Always off: Jev gets no date preprocessing either. |
| HF token | Handled like the OpenRouter key: `HF_TOKEN` in the environment or `.env` is the server default, and a non-blank browser value (`X-HF-Token`) overrides it. It is optional: without one, calls are anonymous (ZeroGPU: 2 min/day; a free account gets 5 min, PRO 40 min). |
| OpenRouter key | Not needed for Kev runs; still required for every other column. |
| Default comparison | "Latest" compares the latest completed run per **visible** column. Hidden columns' runs stay pickable in the Runs checklist. |
| Cost | $0 per request (`cost_estimated`); the ZeroGPU quota is the real budget. |

## 2. Facts verified against the live Space (2026-09-28)

- `GET /gradio_api/info` → `named_endpoints["/decide"]` with the parameters `state_text`, `questions_json`,
  `model_choice` (enum `Kev-4B` | `Kev-0.8B` | `Both`), `calibrated`, `date_facts`, `check_stability` and
  `n_perm`. Gradio 6.28, protocol `sse_v3`, API prefix `/gradio_api`. This call does not use the GPU.
- `POST /gradio_api/call/decide` with `{"data": [...]}` → `{"event_id": "…"}`; then
  `GET /gradio_api/call/decide/{event_id}` → `text/event-stream`.
- Success: `event: complete`, `data: [rendered_html, response, report]`. `response` is the `/v1/systemone`
  body `{"model": "jaredpalmer/kev-4b", "answers": {...}, "usage": {"input_tokens", "output_tokens"},
  "latency_ms"}`, and its `answers` have exactly the shape `classifiers.jev.parse_decisions` reads:
  - `noul`;
  - `choice` with `probabilities` by option name;
  - `score` with `probabilities` keyed by level index.
- Failure: `event: error`, `data: {"error": "<message>", "duration": …, "visible": true, "title": "Error"}`.
  The Space raises `gr.Error` for invalid input, too-long states and (ZeroGPU) an exhausted quota.
- `heartbeat` events arrive every 15 s while queued.
- An unknown endpoint name → HTTP 500, not 404.
- Timing: an anonymous round trip took about 1.2 s, of which 117 ms was GPU time. The Space runs one request at
  a time (Gradio queue, default concurrency limit 1).
- `state_text` starting with `{` is parsed as JSON and rendered as `field: value` lines, so sending
  `json.dumps(email.to_state())` gives Kev the same six fields Jev gets.
- Quota is consumed by the *effective* GPU duration (HF ZeroGPU docs). An HF token is sent as
  `Authorization: Bearer <token>` (Gradio "querying with curl" guide).

## 3. Configuration

`config/benchmark.toml`:

```toml
[kev]
space_url = "https://jaredpalmer-kev.hf.space"
api_name = "decide"
calibrated = true
concurrency = 2
timeout_s = 300

[[columns]]
id = "kev"
title = "Kev"
kind = "kev"
models = ["Kev-4B", "Kev-0.8B"]
default_model = "Kev-4B"
slot = "embeddings"
```

- **`KevParams`** (`kind = "kev"`) joins `RunParams`, the discriminated union.
- **`concurrency = 2`**: the Space serves one request at a time, so a higher value only waits in its queue.
  2 keeps one request queued while the other runs.
- **`timeout_s`** bounds one attempt, including the wait in the queue.
- **`ColumnConfig` changes:**
  - `models: tuple[str, ...] = ()` is a static catalog: required and non-empty for `kev` (and
    `default_model` must be one of them), forbidden for the other kinds.
  - `slot: str | None = None` groups the columns that share one card position. The effective slot is
    `slot or id`.
  - `modality` becomes optional: required for `decisions` / `chat` / `embeddings`, forbidden for `kev`.
    `prefix` is forbidden for `kev`.
- **Catalog:** `Catalog.for_column` returns `ModelInfo(id=m, name=m)` (prices 0, no context length) for each
  static model, without calling OpenRouter.

## 4. Components

- **`kev_space.py`** exports `KevSpaceClient` and `KevSpaceError`.
  - `info()` reads `GET {space_url}/gradio_api/info`.
  - `decide(state_json, questions_json, model, calibrated, token)` does the two-step call and returns
    `ApiResponse(body=<systemone response>, latency_ms=<client round trip>)`.
  - Both calls use absolute URLs on the shared `httpx2.AsyncClient`, whose `base_url` is OpenRouter's.
  - The token is passed per call and sent only when present.
  - `KevSpaceError` subclasses `OpenRouterError`, so `runner._classify`, `failures.failure_text` and
    `log_job_failure` handle it unchanged (fatal → abort without a traceback). A rename to a neutral
    `ProviderError` was considered and left out as a wider refactor.
- **Retries:** `openrouter.py`'s private `_retrying` loop becomes a public module function, `with_retries`,
  used by both `OpenRouterClient` and `KevSpaceClient`, so the backoff has a single implementation.
- **`classifiers/kev.py`** exports `KevClassifier`, which implements `Classifier`.
  - Reuses `questions_payload` and `parse_decisions` from `classifiers/jev.py`: multi-label questions are sent
    as `choice` exactly as for Jev.
  - One email per request; `concurrency` from `KevParams`; budget from `tokens` (no catalog context length, so
    the fallback applies).
  - `prepare()` is the preflight: `info()` must list `/decide`, and the requested model must be in its
    `model_choice` enum. Otherwise a fatal `KevSpaceError` is raised before any email is sent.
  - `classify()` sends the email, parses `answers`, and gets usage through `usage_from_body` (cost 0,
    estimated). `resolved_model` is the body's `model`, and `raw` is the body.
- **`run_launcher.py`:**
  - `build_classifier` and `_params` get a `kev` branch.
  - `launch_run(request, api_key: str | None, hf_token: str | None, services)` checks the key per kind: `kev`
    needs no OpenRouter key; every other kind raises `RunLaunchError` with the "API key is not configured"
    text (the UI's `needsKey` still matches it).
  - The `mode` rule is unchanged: `kev` runs are `per_email`.
- **`settings.py`:** `hf_token: SecretStr | None` from `HF_TOKEN` (blank → None) and `server_hf_token()`.
- **`services.py`:** `hf_token(supplied)`, with the same precedence as `api_key`.
- **`web/deps.py`:**
  - `OptionalApiKeyDep`: the OpenRouter key or None, without the 400.
  - `HfTokenDep`: the `X-HF-Token` header, else the server token, else None.
  - `POST /api/runs` uses both. The other job routes keep `ApiKeyDep`.
- **`/api/status`** gains `server_hf_token: bool`.
- **`/api/catalog`** gains `slot` (the effective slot) and `calibrated` (for `kev`) on `CatalogColumn`.
- **CLI:** `cli_jobs` passes `services.hf_token(None)`. A Kev run from the CLI needs no OpenRouter key.

## 5. UI

- **`js/slots.js`** (pure, tested) exports `slotGroups(catalog)` and `visibleColumns(catalog, picks)`.
  - `slotGroups` returns `[{slot, columns}]`, positioned where the slot first appears in config order.
  - `visibleColumns` returns one column per slot: the stored pick when it is still a member, else the slot's
    first column. Embeddings therefore stays the default.
- **Benchmark page:**
  - One card per slot.
  - A slot with more than one column shows a segmented control (`Embeddings | Kev`) in the card header in
    place of the plain title. Switching stores `benchmark.slot.<slot>`, re-renders that card, and, unless
    runs were picked explicitly, resets "Latest".
  - `defaultRunIds` gets the visible columns. `orderByColumn` still ranks by the full catalog.
- **Analyze page:** its "Latest" uses the same `visibleColumns`, since the pref is shared.
- **Kev card:**
  - the model select, labelled "free";
  - an extras line: `calibrated · HF Space · quota: your token | server token | anonymous (2 min/day)`, with
    a (?) tooltip;
  - the estimate reads "free (ZeroGPU quota)" with the email and request counts;
  - the same stats rows as Jev.
- **Keys:**
  - The key dialog gains an optional "Hugging Face token (for Kev)" field, kept in localStorage (`hf-token`)
    and sent as `X-HF-Token` only by `api.createRun`.
  - The navbar badge keeps showing the OpenRouter key state.
  - The Help page's API-keys section describes both keys and how to create an HF token
    (<https://huggingface.co/settings/tokens>, read access is enough).
- **Colors:** the eight validated slots are all in use, and a ninth hue is never generated, so Kev takes the green
  spare `#008300` (both modes). It goes into `js/palette.js` as `SERIES.kev`, and into `app.css` as
  `--jb-col-kev` / `.accent-kev`. Red stays the only spare.
  - Validator results for both orders (Kev beside Embeddings, and Kev swapped in for it):
    - light: every check passes;
    - dark: CVD ΔE 6.9 against the reference's amber. That band is legal only with secondary encoding, which
      every chart has (legend, tooltips, named table rows).
  - Red fails: normal-vision ΔE 13.2 (light) and 7.8 (dark) against Embeddings' magenta.
- **Glossary:** new keys for the quota line and the Kev estimate. The node test that every
  `withHelp` / `helpIcon` key exists covers them.

## 6. Error handling

| Situation | Handling |
|---|---|
| Preflight: `/decide` missing or model not in its enum | fatal `KevSpaceError`: the run fails before any email is sent |
| HTTP 401 / 403 (bad HF token), 404 | fatal |
| `error` event whose text contains "quota" (case-insensitive) | fatal: an exhausted quota fails every remaining email. Matching on the message text is a heuristic. |
| Other `error` event (invalid input, state too long) | that email fails with the Space's message |
| Transport error, HTTP 429 / 5xx, stream ended without `complete` / `error` | retried with the shared backoff, then that email fails |
| Attempt exceeds `timeout_s` | that email fails, not retried (a stuck public queue) |
| `complete` data malformed (not a list, `response` not an object) | that email fails ("malformed Space response") |

Every `KevSpaceError` message is masked for `hf_…` tokens (the last four characters are kept), like the
`sk-…` masking in `openrouter.py`.

## 7. Invariants kept

- Kev sees only `Email.to_state()`, as JSON: no email ids, traits, reference answers or generator model.
- The HF token is held as `SecretStr` in the client and the classifier, is never persisted (run meta, params,
  responses) and is never logged. The sentinel test in `tests/test_web_routes_runs.py` gains an HF sentinel.
- The header is sent only on `POST /api/runs`.
- No `innerHTML`; no inline scripts; no new CDN dependency.
- Config is re-read on every run start; runs snapshot `KevParams` and the column config.

## 8. Testing

**Unit tests** (one file per module, `pytest.param` tables, the Space mocked with `httpx2.MockTransport`):

- `tests/test_kev_space.py`:
  - event mixes: `heartbeat` then `complete`; `error`; a quota `error`; a stream ending early;
  - HTTP 401 / 404 / 429 / 500 / transport error;
  - the timeout;
  - the token header present or absent;
  - the token masked in every error message;
  - `info()` parsing.
- `tests/test_classifiers_kev.py`:
  - the request's `data` (state JSON, `questions_payload` JSON, model, calibrated, `false, false, 4`);
  - parsing through `parse_decisions`; usage cost 0;
  - preflight pass and fail.
- **New rows in existing tables:**
  - `test_benchmark_config.py`: the `models` / `slot` / `modality` / `prefix` rules;
  - `test_catalog.py`: the static catalog;
  - `test_run_launcher.py`: Kev without an OpenRouter key; the others still require one; params snapshot;
  - `test_settings.py`, `test_services.py`: HF token precedence;
  - `test_web_routes_runs.py`: headers, and the HF sentinel never stored or logged;
  - `test_web_routes_status.py`, `test_web_routes_catalog.py`;
  - `test_openrouter.py`: `with_retries`;
  - `test_cli_jobs.py`.
- **JS:**
  - `tests/js/slots.test.js`;
  - new rows for `palette` (a Kev color) and `selection` (default runs over visible columns).

**Integration:** `tests/test_integration_kev.py` (`-m integration`, opt-in) makes one real Space call on a
sample state and asserts every question parses. It is free but uses ZeroGPU quota.

**Gates:** `ruff`, `pyright` (0 errors), the default suite under 5 s, coverage at least 95%, `node --check`,
`node --test tests/js/`.

## 9. Docs

- `CLAUDE.md`: the Kev column in Architecture (transport, static catalog, slots); the HF token under Key safety
  and Invariants.
- `README.md`: configuration, keys, and the swap.

## 10. Out of scope

- Self-hosted `kev.serve` (`POST /v1/systemone`).
- The Space's `Both` model choice.
- `date_facts`.
- The option-order stability check.
- Kev in the Analyze model picker (Kev is not a chat model).
