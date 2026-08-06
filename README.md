# Idea Ledger

A local tool that evaluates business ideas against **your** rules and keeps the
scores honest as facts change.

You describe what to look at, in your own words. Candidates are generated, gated
against rules you control, checked against real competitors with citable urls,
then scored. The scoring is plain code — no model ever assigns a number — so the
same evidence always produces the same score, and a rejection can always name
the rule that caused it.

Everything runs on your machine. No account, no server, no third-party Python
packages.

---

## Why it works this way

This began as [imirchev/business-idea-agent](https://github.com/imirchev/business-idea-agent),
a prompt pack: 281 lines of instructions plus 5,262 lines of the output they
produced. Reading that output against itself showed the problem. Across 54
scorecards, one dimension used 3 of its 10 possible values; totals spanned 35–47
with a standard deviation of 2.39. The measurement error was larger than the
discrimination — the scores could not separate the ideas they were meant to
rank. One idea's whole case rested on a licence nobody had checked, and nothing
in the system could answer "what else did that assumption hold up?"

None of that is fixable with a better prompt, because none of it is a language
problem. So the parts that must not vary became code, and the model kept only
the parts that genuinely need judgement:

| Deterministic (code) | Probabilistic (model) |
|---|---|
| Gates and constraint predicates | Proposing candidates |
| Scoring, thresholds, ranking | Gathering evidence |
| Provenance and status transitions | Attempting refutation |
| Assumption invalidation | Rendering prose |
| Calibration | |

The model is an adapter, not the core. It cannot skip a gate, cannot decide it
has searched enough, and never sees a score.

---

## What it does

- **Talk to it normally.** No command grammar — "lets look at custom wooden
  furniture in Florida", "why was that one rejected?", "add a rule rejecting
  anything over $5k".
- **Gates run in code, before any model judges.** A rejection names the rule.
- **It refuses to score thin evidence.** Too few competitors with resolvable
  urls and an idea comes back `under_researched` rather than scored on nothing.
- **Every score carries a falsifier** — the observation that would move it.
  That is the column you argue with.
- **Overrides are recorded and quarantined.** Your judgement calls never teach
  the system they were measurements.
- **Assumption graph.** Mark one premise false and it names every score, idea
  and document downstream of it.
- **Capability negotiation.** If your provider cannot search, the evidence stage
  is refused with a reason rather than quietly inventing competitors.
- **Self-diagnosis.** A Check tab that tests storage, credentials, connection,
  model, capabilities and transport against your real key.
- **Archives you can read without this app.** Dated folders holding
  `dossier.md` and `snapshot.json`.

---

## Running it

**Prerequisite:** Python 3.11 or newer. Nothing else — no `pip install`, no
virtualenv. The whole thing is standard library.

```bash
git clone https://github.com/arsurvey82/idea-ledger.git
cd idea-ledger
python -m app
```

Your browser opens at `http://127.0.0.1:8420`.

```bash
python -m app --port 9000 --no-browser     # if 8420 is taken
```

Data lives in `~/.idea-ledger` (override with `IDEA_LEDGER_HOME`). Your API key
is **never** written there — it goes to your operating system's secret store:
Windows DPAPI, macOS Keychain, or `secret-tool` on Linux. If no store is
available, saving is refused rather than writing plain text.

### First run

1. **Setup** → pick a provider → paste your key → name it if you like →
   **Save & test**.
2. **Check** → **Run checks**. Eleven checks against your real key; the first
   failure is usually the only one worth reading.
3. **Facts** → set `location` and `licences_held` to your actual situation. The
   shipped values are placeholders, and gates read these fields by name.
4. Type something in the chat box.

### Choosing a provider

Capabilities decide which pipeline stages can run, so this is a real choice
rather than a preference:

| Provider | All five stages | Notes |
|---|---|---|
| **Anthropic** | yes | Native web search and schema enforcement. Defaults to `claude-sonnet-5`; the model is selectable. |
| **Google Gemini** | no | Genuinely free tier. Generation, rules and rendering work. Search grounding is not exposed on the OpenAI-compatible endpoint, so evidence and refutation are refused. |
| **OpenAI** | no | This build does not use the Responses API surface, so it cannot search. |
| **OpenRouter** | route-dependent | Capabilities belong to the route, not the gateway. **Find one that works** filters on published support for structured output and tools, then places a real call against each candidate — free routes first, then cheapest paid. |

Several keys can be saved and named; switching between them switches provider
with them.

---

## Reading the output

| Outcome | Means |
|---|---|
| `scored` | Passed your gates, found enough cited competitors, got a number |
| `gate_rejected` | Killed by one of your rules, in code, before any model judged it |
| `under_researched` | **Not a failure.** Refused to score rather than score thin evidence |
| `refuted` | A check found the premise does not hold |
| `duplicate` | Already on your reject list; reviving one is a deliberate act |

Scores compare only within a track and within a rubric version. The dossier says
so in the file, so the number cannot be carried somewhere it means nothing.

---

## How rules work

A rule is **data**, never executed as code. Symbolic rules are predicates over
your fields, read by a small interpreter — no `eval`, no `exec`, no attribute
traversal:

```json
{
  "field": "capital_required_usd",
  "op": "lte",
  "other": "budget_ceiling_usd"
}
```

Operators are `lte`, `gte`, `lt`, `gt`, `eq`, `ne`, `falsy`, `truthy`, combined
with `all`, `any`, `not`. `other` compares one field against another, so a rule
that reads your budget keeps working when you change the budget.

A rule referencing a missing field **fails closed** — nothing passes — rather
than silently letting ideas through.

Add one in the Rules tab or just ask in chat. Either way it is dry-run against
the whole ledger first and reports its blast radius; activation is a separate,
explicit act.

---

## Layout

```
app/
  core/            no I/O, no clock, no model
    types.py       provenance, status transitions, evidence
    rules.py       the predicate interpreter
    scoring.py     totals, thresholds, rank-within-track
    assumptions.py the invalidation graph
    calibration.py measured bias, with a sample floor
    manifest.py    the pipeline's self-description
  providers/       one adapter per wire format
  pipeline.py      ST0-ST6, deterministic
  evaluator.py     the model-backed stages
  web.py           stdlib HTTP server
  static/          the single-page interface
tests/             176 tests, no key and no network required
e2e_check.py       clean-install walkthrough against a real provider
```

Ports and adapters. The core has no I/O, no clock and no model, which is why
most of the suite needs neither a key nor a network.

---

## Testing

```bash
python -m unittest discover -s tests -t .   # 176 tests, offline
python e2e_check.py                          # real provider, clean install
```

`e2e_check.py` starts a second server on an empty home directory and drives the
same HTTP surface the browser uses, against your real key. It keeps its failures
in the report.

---

## Modifying it with Claude

Paste this at the start of a session:

> I'm working on Idea Ledger, a local neuro-symbolic tool for evaluating
> business ideas. Read `README.md` first, then `app/core/manifest.py` and
> `defaults/manifest.json` — the manifest is the pipeline's self-description and
> drives both rule placement and capability negotiation.
>
> Load-bearing constraints. Please do not break them:
>
> - **Standard library only.** No third-party packages anywhere, including
>   tests. Someone should be able to clone and run with only Python installed.
> - **`app/core/` is pure.** No I/O, no clock, no model, no randomness. If a
>   change needs any of those, it belongs above the core.
> - **Gates and scoring never consult a model.** The model proposes, gathers
>   evidence, attempts refutation, and renders prose. Nothing else.
> - **Rules are data.** The predicate interpreter must never gain `eval`,
>   `exec`, `getattr` traversal, or anything that executes rule content.
> - **Capabilities are checked, not claimed.** If a provider might not support
>   something, verify it against that provider's own metadata and declare the
>   truth. A stage that cannot run is refused with a reason. Silently degrading
>   is the specific failure this design exists to prevent.
> - **Opaque provider state round-trips unmodified.** OpenRouter's
>   `reasoning_details` and Gemini's `thought_signature` are carried, never
>   interpreted, never dropped.
> - **A key is never written to the repository, the config file, a log, or a
>   report.** The config stores a reference, not a secret.
> - **ASCII in anything printed.** Reports get read on Windows consoles using
>   cp1252, where an em-dash is a crash.
>
> When adding a provider, register it in every table that needs it — the
> registry, `ENDPOINTS`, `connectivity.PROBES`, `_COMPAT` — and add a
> `diagnose()` check. Forgetting one produces a silent downgrade rather than an
> error; that has happened three times.
>
> Run `python -m unittest discover -s tests -t .` before and after. When you
> change `app/static/index.html`, `tests/test_page_script.py` executes it against
> a stub DOM — that suite exists because the Python tests cannot see the browser
> and two UI outages shipped unnoticed.

---

## Known limits

Stated plainly so you can tell a limitation from a bug:

- Google's search grounding is not reachable through its OpenAI-compatible
  endpoint. Verified rather than assumed: `google_search` is rejected in all
  three documented forms.
- The pipeline has not yet been observed completing end to end on real data.
  The transports are verified individually; the full five-stage loop is not.
- Rule-change-by-conversation works, but has been seen on few phrasings.
- Free tiers rate-limit quickly. Fine for exploring, not for a long session.

---

## Credit

The rubric, the six dimensions, the track split and the original framing come
from [imirchev/business-idea-agent](https://github.com/imirchev/business-idea-agent).
This is a reimplementation of those ideas as a system rather than as a prompt.
