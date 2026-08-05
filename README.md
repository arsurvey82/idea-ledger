# Idea Ledger

A decision ledger with a language model attached to it.

It holds business ideas, facts about you, and evidence-backed claims. It can propose
new ideas — but its main job is keeping scores honest as facts change, because that
is what the work actually consists of: one generation pass, then many corrections.

**It is not** a chatbot, a service, or an autonomous agent. Control flow is ordinary
code. The model sits at the edge, alongside the database, and is trusted about as
much as any other remote service.

---

## Status

This is under construction. The table says what runs today, so you can tell the
difference between a feature and a plan.

| Component | State |
|---|---|
| Domain types, provenance, status transitions | working |
| Constraint predicates and gate evaluation | working |
| Scoring, thresholds, rank-within-track | working |
| Assumption graph and invalidation | working |
| Calibration loop | working |
| Provider capability negotiation | working |
| Configuration and key resolution | working |
| Pipeline orchestrator (ST0–ST6) | working |
| Rule intake: triage, placement, dry-run, confirmation | working |
| SQLite persistence | working |
| Web interface + live stage stream | working |
| Anthropic transport (request/response) | written, **not yet called live** |
| Rule compile step (plain language → typed rule) | not yet — the stages after it are |

The whole test suite runs with no API key, no network, and no third-party
packages. That is deliberate: everything the system is *trusted* for lives in a
pure core that a model cannot reach.

---

## Prerequisites

- **Python 3.11 or newer.** That is the only thing you must install.
- **An API key** for one provider — Anthropic, OpenAI, or OpenRouter — once you
  want to run an evaluation. Nothing before that point needs one.

No Node, no npm, no build step, no database server.

---

## Install

```bash
git clone <your-fork-url> idea-ledger
cd idea-ledger
python -m unittest discover -s tests -t .   # should pass before you have a key
python -m app                               # opens http://127.0.0.1:8420
```

With no key configured the interface still works: **Run** uses a labelled demo
evaluator so you can watch the pipeline gate, reject, score, and archive without
spending anything or calling a model.

---

## Configure your key

**Your key never goes in this repository.** The config file stores the *name* of
an environment variable, never the value, so a config file committed by accident
leaks a variable name and nothing else.

Set the variable your provider uses:

```powershell
# PowerShell
$env:ANTHROPIC_API_KEY = 'your-key'
```

```bash
# bash / zsh
export ANTHROPIC_API_KEY='your-key'
```

| Provider | Variable |
|---|---|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

To check what the app can see — this prints where the key came from and its last
four characters, never the key:

```bash
python -c "from app.config import Config; print(Config().with_provider('anthropic').describe())"
```

---

## Choosing a provider

Providers are interchangeable, but not identical, and the difference matters.

Some stages need capabilities a backend may not have. The clearest case is
**server-side web search**: without it, a model asked to name competitors will
invent them, which is the single largest error in the system this replaces. So
capabilities are negotiated once, at setup, and a backend that cannot do the job
is told so before you run anything:

```
openrouter / (route not chosen)
  generate       ok          provider supports every capability this stage needs
  evidence       cannot run  missing server_search - without search the model invents
                             competitors instead of finding them
  refute         cannot run  missing server_search - a refutation with no source is an
                             opinion, not a check
  render         ok          provider supports every capability this stage needs
  compile_rule   ok          provider supports every capability this stage needs
```

Two ways to clear that:

1. **Point at a route that can search.** OpenRouter passes through to whatever
   model you name, so declare that route's real capabilities and re-negotiate.
2. **Add a search adapter.** A standalone search API supplies what the model
   provider lacks; the stage then runs as *compensated*, and the report says so.

Swap backends freely. Where one genuinely cannot do the job, you find out at
setup in one sentence — never by reading a plausible answer that turns out to be
invented.

---

## Your data versus this repository

The split is the privacy boundary, and on a public repo it is the whole point.

| Ships here, read-only | Created on your machine, never committed |
|---|---|
| Rubric: dimensions and definitions | Your fact base — who you are, your constraints |
| Gate thresholds, track definitions | Idea ledger, scores, statuses |
| Output schemas, prompt templates | Evidence, assumptions, run log |
| Help content | Overrides, archive, exports |
| An example fact base — **fictional** | Reject list — starts empty |

Operator data lives in `~/.idea-ledger/`, which `.gitignore` blocks. A second
person cloning this gets the framework and an empty ledger. Nobody inherits
anyone else's profile, income, or family details.

Override the location with `IDEA_LEDGER_HOME` if you keep more than one profile.

---

## Concepts worth knowing before you use it

**Provenance.** Every number is `derived` (computed from evidence), `overridden`
(you edited it; the original is kept), or `seeded` (imported, no evidence). The
dossier shows which. Human overrides are excluded from calibration — counting
your corrections as model error would make the system measure you instead of it.

**Tracks.** Scores rank *within* a track only. A physical-goods idea and a
service idea are not on one scale, and the system refuses to sort them together
rather than quietly doing it.

**Rubric versions.** Adding, removing, or redefining a dimension is a breaking
change: every stored score stops being comparable to every future one, and the
missing evidence cannot honestly be backfilled. The ledger versions rather than
pretends, and offers a costed re-run instead of guessing.

**Assumptions.** When a claim is used, the dependency is recorded. Later you can
ask what breaks if it is wrong, and get an answer instead of doing archaeology:

```
4 artifact(s) depend on this assumption:
  "Partner holds a Florida licence"

  dossier (1)
    bayline-advantages           ranked the credential first
  idea (1)
    casa-sello                   scored as licence-gated
  score (1)
    casa-sello-v1                competition dimension assumed the moat
```

**Calibration.** Every rescore is measured. Once there are enough observations,
the measured bias is fed back into the prompts. Below that floor the system
reports a direction and refuses to state a number — a confident prior computed
from three data points is the overconfidence it exists to correct.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `no key found; expected environment variable X` | The variable is unset in *this* shell. Setting it in another window does not carry over |
| A stage reports `cannot run` | The provider lacks a capability that stage needs. Change route or add a compensator — see above |
| `NotComparable: track spans rubric versions` | Ideas scored under different rubrics. View separately, or re-run the evidence pass |
| `IllegalTransition: rejected -> active` | Rejected ideas revive to *on hold* for review, never straight back to active |
| `UnresolvedInput` on a gate | A rule reads a field your fact base does not define. Add the field, or drop the rule |

---

## Licence

Add one before publishing.
