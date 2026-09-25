# UI & Benchmark upgrades (B) and the Analyze tab (C)

Sub-projects **B** and **C** of the 2026-09-25 improvement request; follows
`2026-09-25-question-model-v2-design.md` (A, merged). The user asked to implement B and C directly on `main`
without further review gates, so this document records the decisions made while implementing. Every item
traces to the original request.

## B. UI and Benchmark upgrades

1. **Your own key, even with a server key.**
   - `Services.api_key(supplied)` now prefers a non-blank browser key (`X-OpenRouter-Key`) over
     `OPENROUTER_API_KEY`. This reverses the previous "server key wins" invariant.
   - The header is still sent only on job-starting POSTs: runs, generations and analyses.
   - Keys are still never persisted or logged.
   - The navbar badge shows which key is in effect ("server key" / "your key").
   - The key dialog explains the override and how to get a key:
     - create a key at <https://openrouter.ai/keys>;
     - add credits at <https://openrouter.ai/settings/credits>;
     - optional provider keys (BYOK) at <https://openrouter.ai/workspaces/default/byok>.
2. **Preliminary cost estimate per run.** `POST /api/runs/estimate` takes a `RunRequest` and returns
   `RunEstimate`:
   - request, email and token counts;
   - a **token estimate**: the same request plan the runner uses, priced with the catalog's per-token prices.
     It is an upper bound: Jev's output reserve is pessimistic and cached embeddings are ignored;
   - a **history estimate**: the mean cost per email of past completed runs with the same column, model and
     mode (embeddings use `cold_cost`), × the number of emails.

   No key is needed. Each column card shows the estimate and refreshes when the model, mode or generations
   change.
3. **Run selection on the Benchmark page.**
   - A "Runs" checklist lists the completed runs on exactly the selected generations. It defaults to the
     latest completed run per column (`selection.js`) and is synced to `?runs=`.
   - The private duplicates in `benchmark.js` are removed.
4. **Speed and cost relative to Jev.** The raters summary shows "Time vs Jev" and "Cost vs Jev" as `×N`
   relative to the selected Jev run (Jev = ×1; "—" without a Jev run), next to the absolute duration and cost,
   plus cost per email.
5. **Tooltips.**
   - A (?) icon next to every metric, written for a non-technical manager: what it is and how to read it.
   - All texts live in one module, `js/glossary.js`, which also feeds the Help page.
   - Bootstrap tooltips are initialized once, delegated from `document.body`.
6. **Help page** (`help.html`), with sections:
   - About this benchmark;
   - Reading the Benchmark page;
   - Reading the Explorer;
   - The Analyze tab;
   - the Metrics glossary;
   - Methodology (data generation, the reference, probabilities, the multi-label rule, scores, confidence
     intervals, Fleiss' κ, disagreement, cost figures, caveats);
   - the question set, live from `GET /api/questions`;
   - API keys.

   The navbar gains "Analyze" and "Help".
7. **Explorer usability.**
   - The detail panel is narrower (`min(680px, 100vw)`), and the header rows get icons (From, To, Cc, Sent,
     Generator, Traits, Id).
   - Each question's options are ordered by the reference labels first, then by the highest probability any
     run gave. Options below 5% everywhere that are not in the reference are folded into a "Show N more" row.
   - Reference rows are highlighted.
   - A run's chosen/applied option is green when it is in the reference and red when it is not.
   - The ordering is a pure, tested helper (`js/option-order.js`).
8. **Colors and dark theme.**
   - One professional accent per column, reused in cards and charts: Jev teal, Anthropic amber, OpenAI green,
     Embeddings violet, reference slate.
   - Tinted card headers, colored summary badges.
   - The dark theme uses a near-black background (`#0a0b0d`) with slightly lighter surfaces.

## C. The Analyze tab

- **Config:** `config/analysis.toml` holds:
  - `default_model` (`anthropic/claude-sonnet-5`, 1M context);
  - `max_disputed_emails` (12) and `max_output_tokens` (8000);
  - `system_prompt` and `user_prompt`: `string.Template` texts with the placeholders `$generations`, `$runs`,
    `$questions`, `$report`, `$emails` and `$disputed`.
- **Inputs to the analyst:**
  - the aggregate comparison report (rounded, compact JSON);
  - a compact per-email table: each run's top answer and confidence per question, plus the reference;
  - the full text of the N most-disputed emails, by the disagreement index.

  That is the user's choice from the planning Q&A. Emails are addressed as `e001…`; no email ids and no
  generator models are sent.
- **Prompt editing:**
  - the page prefills the config prompts and offers "reset to default";
  - edits are kept per browser;
  - the prompt actually sent is snapshotted in the analysis record.
- **API:**
  - `GET /api/analysis/defaults`;
  - `POST /api/analyses/estimate` (input tokens and cost, and whether it fits the model's context);
  - `POST /api/analyses` (a background job; key required);
  - `GET /api/analyses`, `GET /api/analyses/{id}` (with live progress), `POST /api/analyses/{id}/cancel`.
- **Persistence:** `data/analyses/<id>.json` (meta, the prompts used, the result Markdown, usage, cost,
  duration, status). Interrupted analyses are swept on server start, like runs.
- **Rendering:** the result is Markdown, rendered by a small safe parser (`js/markdown.js`: headings,
  paragraphs, lists, tables, code, bold/italic/inline code) into `h()` nodes. There is no `innerHTML`.
- **Model call:** a streamed chat completion (progress = characters received). Unsupported sampling
  parameters are not sent (`temperature` and `reasoning` are left out when the catalog omits them). A 401,
  402, 403 or 404 fails the job with the provider's message.

## Out of scope

CLI commands for analyses; editing TOML config from the UI (pre-existing non-goal); a cost estimate for
generations.
