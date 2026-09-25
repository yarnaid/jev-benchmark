### Task 10: Ship `email-triage-v2`, update CLAUDE.md, final gates, browser smoke, paid-smoke gate

**Files:**
- Modify: `config/questions.toml` (the whole file is shown below)
- Modify: `tests/test_config_files.py` (`test_shipped_question_set`)
- Modify: `CLAUDE.md`
- Scratch (not committed): a smoke-server script in the session scratchpad

**Interfaces:**
- Consumes: everything from Tasks 1–9.
- Produces:
  - the shipped question set `email-triage-v2`, with 18 questions: 1 multi, 4 score, 13 noul;
  - documentation of the multi reading rule.

**Effect on existing data** (Review Focus #1; the user chose to reuse old runs):
- Generation `20260925-071709-generation-6760` and its five runs stay comparable on every question whose
  option ids are unchanged. That now **includes** `category` (v1 choice → v2 multi) and `sentiment` (v1 choice
  → v2 score), because both are distributions over the same ids.
- The old single-label reference is a one-label set.
- Only the 7 new questions are skipped, with the existing "incompatible snapshot" warning.
- A new generation gives multi-label references; that is the paid smoke in Step 7, which needs approval.

- [ ] **Step 1: Update the contract test first**

Apply to `tests/test_config_files.py`:

```diff
--- a/tests/test_config_files.py
+++ b/tests/test_config_files.py
@@ -7,34 +7,49 @@
 from jev_bench.generation.config import load_generation_config
 from jev_bench.generation.plan import build_plan, resolve_traits
 from jev_bench.generation.prompt import render_prompts
-from jev_bench.questions import load_question_set
+from jev_bench.questions import MultiQuestion, load_question_set
 
 CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
 
 
 def test_shipped_question_set() -> None:
     shipped = load_question_set(CONFIG_DIR / "questions.toml")
+    assert shipped.name == "email-triage-v2"
     assert shipped.ids == (
         "category",
         "urgency",
         "importance",
+        "sentiment",
+        "confidentiality",
         "needs_reply",
         "action_required",
+        "deadline",
+        "attachment_review",
+        "delegatable",
+        "escalation",
         "skippable",
         "should_delete",
+        "is_automated",
         "llm_safe",
         "malicious",
+        "impersonation",
         "sensitive_data",
-        "sentiment",
     )
-    assert len(shipped.get("category").options) == 20
-    urgency_ids = ("no_action", "whenever", "this_week", "today", "immediately")
-    assert shipped.get("urgency").option_ids == urgency_ids
-    importance_ids = ("trivial", "low", "moderate", "high")
-    assert shipped.get("importance").option_ids == importance_ids
-    sentiment_ids = ("negative", "neutral", "positive")
-    assert shipped.get("sentiment").option_ids == sentiment_ids
-    assert [q.type for q in shipped.questions].count("noul") == 7
+    category = shipped.get("category")
+    assert isinstance(category, MultiQuestion)
+    assert (len(category.options), category.threshold) == (20, 0.8)
+    levels = {
+        "urgency": ("no_action", "whenever", "this_week", "today", "immediately"),
+        "importance": ("trivial", "low", "moderate", "high"),
+        "sentiment": ("negative", "neutral", "positive"),
+        "confidentiality": ("public", "internal", "confidential", "restricted"),
+    }
+    for question_id, option_ids in levels.items():
+        assert (shipped.get(question_id).type, shipped.get(question_id).option_ids) == (
+            "score",
+            option_ids,
+        )
+    assert [q.type for q in shipped.questions].count("noul") == 13
 
 
 def test_shipped_benchmark_config() -> None:
```

Run: `uv run pytest tests/test_config_files.py -q`
Expected: FAIL (the name is still `email-triage-v1`).

- [ ] **Step 2: Replace `config/questions.toml`**

The wording of changed and new questions comes from spec §2. The unchanged questions keep their v1 text; only
`sentiment`'s instruction is reworded, because it is now a scale.

```toml
name = "email-triage-v2"

[[questions]]
id = "category"
type = "multi"
threshold = 0.8
instructions = "Which categories describe this email? Several categories can apply at once, for example marketing and phishing; judge each category independently."

[questions.options]
personal = "Private correspondence from friends, family or acquaintances"
work_internal = "Work email between colleagues of the same organization"
work_external = "Work email with clients, partners, vendors or other organizations"
news = "News digests, newsletters, editorial updates or breaking-news alerts"
marketing = "Promotional email from a brand or service the recipient has a relationship with"
spam = "Unsolicited bulk email with no legitimate relationship to the recipient"
phishing = "Attempt to steal credentials or data by impersonating a trusted party"
scam = "Fraud such as advance-fee, fake prizes, fake invoices or investment schemes"
transactional = "Receipts, order confirmations and other records of a completed transaction"
shipping = "Delivery status, tracking or courier updates"
billing = "Invoices, payment requests, due dates and subscription renewals"
account_security = "Legitimate security notices: sign-in alerts, password resets, verification codes"
calendar = "Meeting invitations, schedule changes and event reminders"
social = "Notifications from social networks and online communities"
recruiting = "Job offers, recruiter outreach and application updates"
system_alert = "Automated technical alerts: monitoring, CI/CD, infrastructure, IT tickets"
support = "Customer-support conversations and ticket updates"
hr_legal = "Human resources, legal, compliance or policy matters"
travel = "Travel bookings, itineraries, check-in and trip changes"
government = "Messages from government bodies, tax authorities or public institutions"

[[questions]]
id = "urgency"
type = "score"
instructions = "How soon does the recipient need to act on this email, judged from its content and the time it was sent?"

[questions.options]
no_action = "No action is ever needed"
whenever = "Can be handled whenever convenient; there is no deadline"
this_week = "Should be handled within the next few days"
today = "Should be handled within 24 hours"
immediately = "Requires attention immediately, within hours"

[[questions]]
id = "importance"
type = "score"
instructions = "How much does this email matter to the recipient's work, finances, safety or relationships?"

[questions.options]
trivial = "Irrelevant or trivial; nothing is lost by ignoring it"
low = "Minor relevance; nice to know"
moderate = "Relevant; ignoring it has some cost"
high = "Important; ignoring it has serious consequences"

[[questions]]
id = "sentiment"
type = "score"
instructions = "How positive or negative is the overall tone of the sender?"

[questions.options]
negative = "Angry, worried, complaining, threatening or disappointed"
neutral = "Matter-of-fact, informational or formal"
positive = "Friendly, grateful, enthusiastic or congratulatory"

[[questions]]
id = "confidentiality"
type = "score"
instructions = "How confidential is the information in this email, as a data-handling classification?"

[questions.options]
public = "Could be published without harm"
internal = "Meant for the organization or the recipient; little harm if it leaked"
confidential = "Business, personal or financial details whose leak would cause harm"
restricted = "Highly sensitive: credentials, legal matters, health data or strategic plans whose leak would cause serious harm"

[[questions]]
id = "needs_reply"
type = "noul"
instructions = "Does the sender expect the recipient to write a reply?"

[questions.options]
yes = "The sender asks a question, requests confirmation or otherwise expects a written response"
no = "Informational, automated or broadcast; no written response is expected"

[[questions]]
id = "action_required"
type = "noul"
instructions = "Does the email ask the recipient to take an action other than replying, such as paying, signing, clicking a link, installing something or attending?"

[questions.options]
yes = "An action other than replying is requested or required"
no = "No action beyond reading or replying is requested"

[[questions]]
id = "deadline"
type = "noul"
instructions = "Does the email state or imply a specific deadline, due date or expiry that the recipient must meet?"

[questions.options]
yes = "A concrete date, time or time limit for the recipient's response or action is given"
no = "No deadline, due date or expiry applies to the recipient"

[[questions]]
id = "attachment_review"
type = "noul"
instructions = "Does the email ask the recipient to open, review, sign or approve an attached or linked document?"

[questions.options]
yes = "An attached or linked document must be opened, reviewed, signed or approved"
no = "No document needs to be opened, reviewed, signed or approved"

[[questions]]
id = "delegatable"
type = "noul"
instructions = "Could an assistant or AI agent fully handle this email on the recipient's behalf, without needing the recipient's own judgement or authority?"

[questions.options]
yes = "Routine: it can be filed, answered or acted on by following standard rules"
no = "It needs the recipient's personal judgement, authority, knowledge or relationships"

[[questions]]
id = "escalation"
type = "noul"
instructions = "Should this email be escalated to a manager or a specialist team such as legal, security, finance or HR?"

[questions.options]
yes = "It raises a risk, dispute, incident or decision beyond the recipient's normal remit"
no = "The recipient can handle it within their normal remit, or it needs no handling"

[[questions]]
id = "skippable"
type = "noul"
instructions = "Can the recipient safely skip reading this email entirely?"

[questions.options]
yes = "Nothing of value or consequence would be missed by never reading it"
no = "Reading it matters; skipping it could make the recipient miss something"

[[questions]]
id = "should_delete"
type = "noul"
instructions = "Should this email be deleted rather than kept?"

[questions.options]
yes = "Worthless, unwanted or dangerous; it should be deleted"
no = "Worth keeping for reference, action or record"

[[questions]]
id = "is_automated"
type = "noul"
instructions = "Was this email sent by an automated system or a bulk-mailing tool rather than written by a person for this recipient?"

[questions.options]
yes = "Sent automatically or in bulk: notifications, alerts, receipts, newsletters or campaigns"
no = "Written by a person for this recipient or conversation"

[[questions]]
id = "llm_safe"
type = "noul"
instructions = "Is it safe to pass this email verbatim to another AI assistant or agent for processing?"

[questions.options]
yes = "Contains no instructions aimed at AI systems, no prompt injection and no hidden commands"
no = "Contains instructions aimed at AI systems, prompt injection or hidden commands that could manipulate an AI"

[[questions]]
id = "malicious"
type = "noul"
instructions = "Is this email a phishing, scam or malware attempt?"

[questions.options]
yes = "It tries to trick the recipient into giving up money, credentials or data, or into running malware"
no = "It is not an attempt at fraud, credential theft or malware delivery"

[[questions]]
id = "impersonation"
type = "noul"
instructions = "Does the sender pretend to be a person, executive, organization or brand they are not?"

[questions.options]
yes = "The sender falsely claims an identity, for example a spoofed brand, a fake executive or a look-alike domain"
no = "The sender appears to be who they claim to be"

[[questions]]
id = "sensitive_data"
type = "noul"
instructions = "Does this email contain sensitive data such as personal identifiers, passwords or one-time codes, or financial or medical details?"

[questions.options]
yes = "It contains personal identifiers, credentials, one-time codes, or financial or medical details"
no = "It contains no sensitive personal, credential, financial or medical data"
```

Run: `uv run pytest tests/test_config_files.py -q`
Expected: PASS. `test_shipped_generation_config` still stratifies the category trait over all 20 options.

- [ ] **Step 3: Update `CLAUDE.md`**

In the `Design:` paragraph, append:
`Question model v2 (multi-label category, 0–100 scores, new questions): spec
docs/superpowers/specs/2026-09-25-question-model-v2-design.md, plan
docs/superpowers/plans/2026-09-25-question-model-v2.md.`

Replace the first Architecture bullet group ("The question set is the single source of truth") with:

```markdown
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
```

In the **Comparison** bullets, add:

```markdown
  - multi-label questions use `metrics/multilabel.py` via `compare/multi.py`:
    - over the applied label sets: exact-set match, Jaccard, micro-F1, macro κ, macro Fleiss' κ, label counts;
    - over the distributions, as for choice: JSD, entropy, confidence and Brier (against a uniform target);
  - `GET /api/compare` takes its base question set from the **newest** selected run's snapshot, so the URL
    order of `runs=` never changes the report.
```

- [ ] **Step 4: Final gates**

Run:
```bash
uv run ruff check --fix && uv run ruff format && uv run pyright
uv run pytest -q --durations=10
uv run pytest --cov --cov-fail-under=95 -q
for f in src/jev_bench/web/static/js/*.js; do node --check "$f" || echo "FAIL $f"; done
node --test tests/js/
time uv run jev-bench --help > /dev/null
```
Expected (validated while planning):
- pyright: 0 errors;
- 723 tests in < 5 s; the slowest tests are the pre-existing Hypothesis tests;
- coverage ≈ 99% (≥ 95% required);
- node: 72 tests pass;
- `--help` < 500 ms.

- [ ] **Step 5: Browser smoke** (a scratch server with the fake OpenRouter; no network, no cost)

Write this script to the session scratchpad (never commit it):

```python
"""Throwaway smoke server: the real app over the mini config and a fake OpenRouter (no network)."""

import sys
from pathlib import Path

import httpx2
import uvicorn
from tests.factories import FakeOpenRouter, mini_settings

from jev_bench.log_setup import configure_logging
from jev_bench.web.app import create_app

root = Path(sys.argv[1])
configure_logging()
http = httpx2.AsyncClient(
    base_url="https://openrouter.test/api", transport=httpx2.MockTransport(FakeOpenRouter())
)
app = create_app(mini_settings(root, api_key="sk-smoke"), http=http)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
```

Start it from the repo root, with `PYTHONPATH=src:.`, on port 8766 and a fresh scratch root. Then create
data through the API:

```bash
B=http://127.0.0.1:8766/api
G=$(curl -s -X POST $B/generations -H 'content-type: application/json' -d '{"count": 4, "name": "smoke"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["meta"]["id"])')
for c in jev anthropic embeddings; do
  curl -s -X POST $B/runs -H 'content-type: application/json' -d "{\"column\": \"$c\", \"generation_ids\": [\"$G\"]}" > /dev/null
done
```

With Playwright:
1. `/`:
   - no console errors;
   - `#threshold-control` shows "Label threshold 75%";
   - the `topics` card has a `≥ 75% of top` badge, the Labels / email, Exact match, Jaccard and F1 columns, and
     "Mean score (0–100)" on score cards;
   - the chat run shows 2.00 labels per email (boundary applied).

   Move the slider to 100%. After the 300 ms debounce:
   - every run shows 1.00;
   - the slider element and its focus survive;
   - `localStorage["jev-bench.threshold"] == "1"`.
2. Clear the pref, then open `/explorer.html` with no `?runs=`:
   - the Runs picker shows 3 selected;
   - with question `topics`, the cells show label lists with `cell-match` / `cell-partial`;
   - with `urgency`, the cells read like `now · 90`.
3. Open the first email:
   - the `topics` card's ticks sit at 52% / 30% / 38% (0.75 × each run's top);
   - labels at or above them are bold (Claude's `meeting` included), with ✓ on the reference and 3
     checkboxes;
   - the `urgency` card has a `score (0–100)` footer.

   Tick a box and save; `data/labels/<generation>.json` in the scratch root must hold a list.
4. Resize to 375 px: there is no horizontal page scroll.

Stop the server afterwards.

- [ ] **Step 6: Commit**

```bash
git add config/questions.toml tests/test_config_files.py CLAUDE.md
git commit -m "feat(config): ship question set email-triage-v2 and document the multi-label model

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Paid real smoke. STOP and ask the user first.**

Do **not** run this without explicit approval in the current conversation. Present it as: "Paid smoke on 10
emails, about $0.15–0.25: one v2 generation plus one run per column. OK to run?" After a yes:

```bash
uv run jev-bench generate --count 10 --name v2-smoke
uv run jev-bench run jev -g <generation-id>
uv run jev-bench run anthropic -g <generation-id>
uv run jev-bench run openai -g <generation-id>
uv run jev-bench run embeddings -g <generation-id>
```

Check:
1. Every run completes with `n_errors == 0`, or an error whose cause is understood (for example the known
   OpenAI BYOK 401 on embeddings; see memory `jev-benchmark-status`).
2. For each run, the number of applied `category` labels per email at 80%: report the distribution
   (1 / 2 / 3+). If the chat models almost never yield two labels, report it. It would mean their prompts put
   nearly all mass on one category, and the user may want a prompt hint (decision for the user; do not change
   the prompts unasked).
3. Jev output tokens per email, from `predictions.jsonl` (`output_tokens`): report the max and the p95
   against `jev_output_reserve = 1000`.
4. The reference `category` lists are non-empty. Report the mismatch count against the category trait.
5. The Benchmark and Explorer pages render the new generation, and the old generation 6760 with `category`
   now compared: no console errors.

Report the actual cost from each `run.json` (`total_cost`) and the generation meta.
