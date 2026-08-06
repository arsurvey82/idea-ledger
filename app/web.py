"""Local web interface.

Standard library only. FastAPI would be pleasanter to write, but it would make
the operator install a web framework to read their own ledger, and the locked
decision was one prerequisite: Python.

Threading server plus a small broadcaster gives live stage events over SSE,
which matters because a run takes minutes and four minutes of silence reads as
a hang.
"""

from __future__ import annotations

import json
import queue
import time
import threading
import webbrowser
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlparse

from dataclasses import replace

from .config import (
    DEFAULT_ENV_VARS, Config, KeySource, key_advisory, user_dir, validate_key,
)
from .core.assumptions import AssumptionGraph, summarise as summarise_assumptions
from .core.manifest import Manifest
from .core.rules import ChangeClass, Rule, RuleTarget
from .core.scoring import Threshold, score
from .core.types import FactBase, Idea, Provenance, Rubric, Status
from .pipeline import Candidate, CoveragePolicy, Outcome, Pipeline, StageEvent
from .providers import (
    REGISTRY, default_requirements, get as get_provider, negotiate, with_capabilities,
)
from .rule_intake import ImpactPreview, RuleIntake, RuleProposal
from .store import Store

#: Providers whose routes publish a capability manifest we can read.
_COMPAT = {"google", "openai", "openrouter"}

#: Providers where probing routes one at a time tells you something. A broker's
#: route can be listed, be free, and still be refused by your account, which only
#: a real call reveals. Google serves its own models, so there is nothing to
#: discover that way - this is a deliberate subset of _COMPAT, not an oversight.
_PROBEABLE = {"openai", "openrouter"}

#: Providers whose model list is small and self-served enough to pick from
#: automatically. OpenRouter is excluded on purpose: it fronts hundreds of
#: routes with wildly different prices, so choosing one silently is not a
#: favour - that is what the explicit "Find one that works" button is for.
_AUTO_MODEL = {"google", "openai"}

STATIC = Path(__file__).resolve().parent / "static"
RUBRIC = Rubric(
    version=1,
    dimensions=("demand", "competition", "ease", "capital", "profit", "solo_marketing"),
    tracks=("service", "physical"),
)
THRESHOLDS = (Threshold("ease", 7), Threshold("solo_marketing", 7))

#: Used when the operator has not named a route. OpenRouter's free tier means a
#: working chat before anyone has spent anything.
#:
#: Google is deliberately absent. Any id here is a guess with a shelf life, and
#: the guess that was here - gemini-2.5-flash - had already been retired for new
#: accounts, so a valid key reported "no route called gemini-2.5-flash". Google's
#: model is resolved from the key on save instead; see _AUTO_MODEL.
DEFAULT_ROUTE = {"openrouter": "openrouter/free", "openai": "gpt-4o-mini"}

#: Anthropic's default, kept beside the others rather than imported at module
#: scope so the transport stays lazily loaded.
ANTHROPIC_MODEL = "claude-opus-5"


class Broadcaster:
    """Fan out stage events to every open SSE connection."""

    def __init__(self) -> None:
        self._clients: list[queue.Queue[str]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue()
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(payload)
        with self._lock:
            for q in list(self._clients):
                q.put(line)


class Workspace:
    """Everything the handlers need. One instance per process."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = home or user_dir()
        self.store = Store.open(self.home)
        self.config = Config.load(self.home)
        self.manifest = Manifest.load()
        self.bus = Broadcaster()
        self.pending: dict[str, tuple[RuleProposal, ImpactPreview]] = {}
        self.logs: list[str] = []
        self.history: list[dict[str, Any]] = []
        self._seed_if_empty()

    # -- state -----------------------------------------------------------
    def _seed_if_empty(self) -> None:
        """First run gets a fictional fact base and two starter gates.

        The gates are shipped defaults, attributed as such, and revertable in
        one click. They exist so the pipeline demonstrably *does* something on
        a fresh install; a ledger with no gates rejects nothing, which reads as
        the gate mechanism being broken.

        Both read the operator's own fields rather than copying values into the
        rule, so editing the fact base changes what the gate enforces.
        """
        if not self.store.load_facts().fields:
            self.store.save_facts(
                FactBase(
                    {
                        "budget_ceiling_usd": 20000,
                        "hours_per_week": 12,
                        "location": "unset",
                        "licences_held": [],
                        "accepts_live_negotiation": False,
                    }
                )
            )
        if not self.store.load_rules():
            self.store.save_rule(
                Rule(
                    id="R-budget",
                    description="capital required must not exceed your budget ceiling",
                    target=RuleTarget.SYMBOLIC,
                    stage="gate",
                    predicate={
                        "field": "capital_required_usd",
                        "op": "lte",
                        "other": "budget_ceiling_usd",
                    },
                    author="shipped",
                    active=True,
                )
            )
            self.store.save_rule(
                Rule(
                    id="R-negotiation",
                    description="delivery must not require live negotiation you have ruled out",
                    target=RuleTarget.SYMBOLIC,
                    stage="gate",
                    predicate={
                        "any": [
                            {"field": "accepts_live_negotiation", "op": "truthy"},
                            {"field": "requires_live_negotiation", "op": "falsy"},
                        ]
                    },
                    author="shipped",
                    active=True,
                )
            )

    def facts(self) -> FactBase:
        return self.store.load_facts()

    def rules(self) -> list[Rule]:
        return self.store.load_rules()

    def state(self) -> dict[str, Any]:
        facts = self.facts()
        ideas = self.store.load_ideas()
        graph = self.store.load_graph()
        calibration = self.store.load_calibration()

        rows = []
        for idea in ideas:
            row: dict[str, Any] = {
                "id": idea.id,
                "name": idea.name,
                "track": idea.track,
                "status": idea.status.value,
                "total": None,
                "max": RUBRIC.max_total,
                "overridden": False,
                "thresholds_met": None,
            }
            if idea.vector and not RUBRIC.validates(idea.vector):
                s = score(idea, RUBRIC, THRESHOLDS)
                row.update(
                    total=s.total,
                    overridden=s.has_override,
                    thresholds_met=s.thresholds_met,
                    confidence=round(s.mean_confidence, 2),
                )
            rows.append(row)

        return {
            "facts": dict(facts.fields),
            "ideas": rows,
            "rules": [
                {
                    "id": r.id,
                    "description": r.description,
                    "target": r.target.value,
                    "stage": r.stage,
                    "active": r.active,
                    "author": r.author,
                    "change_class": r.change_class.value,
                }
                for r in self.rules()
            ],
            "assumptions": summarise_assumptions(graph),
            "calibration": calibration.summary(),
            "rejections": self.store.rejections()[:20],
            "manifest": self.manifest.report(),
        }

    def _route_capabilities(self, fallback: Any) -> frozenset:
        """What the *named route* can do, asked of the live adapter.

        Falls back to the gateway's declaration when there is no key to ask
        with, since the route manifest is an authenticated read.
        """
        key = self.config.key(home=self.home)
        if not (key and self.config.model_id and self.config.provider in _COMPAT):
            return fallback.capabilities()
        from .providers.compat_adapter import CompatProvider

        try:
            return CompatProvider(
                provider=self.config.provider, api_key=key, model=self.config.model_id
            ).capabilities()
        except Exception:
            return fallback.capabilities()

    def setup(self) -> dict[str, Any]:
        from . import secrets as secret_store

        provider_name = self.config.provider or "anthropic"
        try:
            backend = get_provider(provider_name)
            if self.config.model_id:
                # A named route is part of the provider's identity, so the
                # capability report must show it rather than "(route not chosen)".
                #
                # The capabilities must be re-derived for that route, not copied
                # from the gateway. Passing backend.capabilities() back in was a
                # no-op that relabelled the report while leaving the gateway's
                # optimistic claims intact - which is how a route with no
                # response_format support was reported as ready to generate.
                backend = with_capabilities(
                    backend, self._route_capabilities(backend), self.config.model_id
                )
            n = negotiate(backend, default_requirements())
            report, ok = n.report(), n.ok
        except KeyError as exc:
            report, ok = str(exc), False

        source, detail = self.config.key_status(home=self.home)
        # A key stored before validation existed - or set in a shell by hand -
        # can still be nonsense. Re-check what is actually there on every read,
        # so the page can say "this is not a key" instead of leaving the
        # operator to infer it from a column of provider rejections.
        stored_key = self.config.key(home=self.home)
        key_complaint = (
            validate_key(self.config.provider, stored_key)[1]
            if stored_key and self.config.provider
            else ""
        )
        store_info = secret_store.describe()
        return {
            "key_complaint": key_complaint,
            "saved_keys": [k.as_dict() for k in secret_store.saved(self.home)],
            "active_account": self.config.key_ref.account,
            # Empty when runs will be real. The page needs this to say so up
            # front rather than after someone has read three fictional results.
            "demo_reason": self.why_demo(),
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "providers": sorted(REGISTRY),
            "env_var": self.config.key_ref.env_var
            or DEFAULT_ENV_VARS.get(provider_name, ""),
            "key_source": source.value,
            "key_detail": detail,
            "has_key": source.value != "absent",
            "store_backend": store_info.backend,
            "store_detail": store_info.detail,
            "store_available": store_info.available,
            "negotiation": report,
            "capabilities_ok": ok,
            "next_step": self.config.next_step(home=self.home).value,
            "home": str(self.home),
            "config_text": self.config.describe(home=self.home),
        }

    def save_setup(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Choose a provider and, optionally, hand a key to the OS secret store."""
        from . import secrets as secret_store

        provider = str(payload.get("provider", "")).strip().lower()
        if provider and provider not in REGISTRY:
            return {"error": f"unknown provider {provider!r}"}

        cfg = self.config.with_provider(provider) if provider else self.config
        model = str(payload.get("model_id", "")).strip()
        if model:
            cfg = replace(cfg, model_id=model)

        stored = advisory = ""
        key = str(payload.get("key", "")).strip()
        if key:
            # Checked before it reaches the store, so a stray paste is refused
            # while the operator still has the clipboard, rather than surfacing
            # later as a wall of "rejected the key" from the provider.
            key, complaint = validate_key(cfg.provider, key)
            if complaint:
                return {"error": complaint, **self.setup()}
            advisory = key_advisory(cfg.provider, key)
            label = str(payload.get("label", "")).strip()
            account = secret_store.account_for(cfg.provider, label)
            # Saving the same value again is not an error and not a change. Say
            # which of the three happened, so "saved" always means something.
            already = secret_store.fetch(self.home, account) == key
            try:
                info = secret_store.store(self.home, account, key)
                secret_store.remember(self.home, account, cfg.provider, label, key)
                cfg = replace(
                    cfg,
                    key_ref=replace(cfg.key_ref, source=KeySource.KEYRING, account=account),
                )
                stored = (
                    f"already saved as {label or account}, unchanged"
                    if already else
                    f"{info.backend}: {info.detail}"
                )
            except secret_store.SecretStoreUnavailable as exc:
                # Refuse rather than silently write plaintext.
                return {"error": f"not stored: {exc}", "env_var": cfg.key_ref.env_var}

        # Resolve a model from the key rather than a constant. Shipping
        # "gemini-2.5-flash" as a default meant a perfectly valid key answered
        # "no route called gemini-2.5-flash", because Google had retired it for
        # new accounts. Only the account can say what it can run.
        picked = ""
        if cfg.provider in _AUTO_MODEL and not cfg.model_id and cfg.key(home=self.home):
            from .providers.openai_compat import pick_default_model

            self.log(f"resolving a model for {cfg.provider}")
            picked, _ = pick_default_model(
                cfg.key(home=self.home) or "", cfg.provider,
                on_try=lambda m, note: self.log(f"  {m}: {note}"),
            )
            if picked:
                cfg = replace(cfg, model_id=picked)
                self.log(f"model resolved to {picked}")

        cfg.save(self.home)
        self.config = cfg
        return {"ok": True, "stored": stored, "advisory": advisory,
                "picked_model": picked, **self.setup()}

    # -- connectivity ----------------------------------------------------
    def test_connection(self) -> dict[str, Any]:
        from . import connectivity

        provider = self.config.provider
        probe = connectivity.check(provider, self.config.key(home=self.home))
        self.log(("connected to " if probe.ok else "connection failed: ") + probe.headline)
        return probe.as_dict()

    # -- logs ------------------------------------------------------------
    def log(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.logs.append(f"{stamp}  {line}")
        del self.logs[:-400]
        self.bus.publish({"type": "log", "line": f"{stamp}  {line}"})

    # -- chat ------------------------------------------------------------
    def chat(self, message: str, evaluator: Any) -> dict[str, Any]:
        """Interpret one operator message and act on it.

        Deliberately a command surface with an agentic presentation rather than
        a language model pretending to be one. Every action it takes is a real
        call into the pipeline, and it says so. Once a key is verified this same
        surface is where model-driven interpretation lands.
        """
        text = message.strip()
        low = text.lower()

        # With a working key this is a real conversation. Without one it falls
        # back to commands, and says so, rather than pretending to understand.
        #
        # The provider set is _COMPAT, not a literal. Spelling it out here left
        # google excluded, so every message from a working Gemini key fell
        # through to the keyword grammar and answered "I do not have a command
        # for that" - the exact app-like behaviour this surface exists to avoid.
        # Chat needs a chat endpoint and nothing else; search and schema support
        # are stage requirements, not conversation requirements.
        if text and self.config.key(home=self.home) and self.config.provider in _COMPAT:
            try:
                return self._agent_chat(text, evaluator)
            except Exception as exc:
                self.log(f"agent failed: {exc}")
                self.bus.publish({
                    "type": "chat", "role": "agent", "kind": "text",
                    "text": f"**The model call failed.** {exc}\n"
                            "Falling back to commands for this message — try `help`.",
                })

        say = lambda t: self.bus.publish({"type": "chat", "role": "agent", "kind": "text", "text": t})
        tool = lambda name, state, detail="": self.bus.publish(
            {"type": "chat", "role": "agent", "kind": "tool",
             "tool": name, "state": state, "detail": detail}
        )
        self.log(f"chat: {text[:80]}")

        if not text:
            return {"ok": True}

        if low in {"help", "?"}:
            say(
                "Things you can ask me:\n"
                "  run <brief>      evaluate candidates against your rules\n"
                "  rules            show every rule currently loaded\n"
                "  facts            show your fact base\n"
                "  test             check the saved key actually connects\n"
                "  status           where setup has got to\n"
                "Anything else, I will tell you plainly that I cannot do it yet "
                "rather than guessing."
            )
            return {"ok": True}

        if low.startswith(("run", "evaluate")):
            brief = text.split(" ", 1)[1].strip() if " " in text else "an idea"
            demo = self.why_demo()
            say(
                f"Evaluating candidates for **{brief}**."
                + (f"\n\n**Using demo candidates.** {demo}" if demo else "")
            )
            tool("run_pipeline", "started", brief)
            try:
                # live first, demo only as a fallback - the same choice the
                # agent's run_pipeline tool makes. This path took whatever was
                # handed in, and /api/chat hands in the demo evaluator
                # unconditionally, so `run <brief>` produced fictional
                # candidates even with a working provider, and said nothing.
                results = self.run(brief, self.live_evaluator() or evaluator)
            except Exception as exc:
                # A provider refusing mid-run is a normal thing to report, not a
                # reason to fail the whole HTTP request. Left uncaught this
                # returned a bare 400 and the chat simply stopped responding.
                message, fix = _explain_failure(exc)
                tool("run_pipeline", "failed", message[:90])
                say(f"**The run stopped.** {message}" + (f"\n\n{fix}" if fix else ""))
                return {"ok": False, "error": message, "fix": fix}
            tool("run_pipeline", "done", f"{len(results)} candidate(s)")
            scored = [r for r in results if r["outcome"] == "scored"]
            cut = [r for r in results if r["outcome"] != "scored"]
            lines = [f"{len(scored)} scored, {len(cut)} cut."]
            for r in cut:
                lines.append(f"  {r['name']} - {r['why']}")
            if cut:
                lines.append(
                    "Those were rejected by code, before any model was asked whether "
                    "they were good ideas."
                )
            say("\n".join(lines))
            return {"ok": True, "results": results}

        if low.startswith("rule") and low != "rules":
            say(
                "Rule intake needs the compile step, which lands with a verified key. "
                "For now use the Rules panel on the right: it does the triage, "
                "placement, dry run and confirmation - only the plain-language parsing "
                "is missing."
            )
            return {"ok": True}

        if low in {"rules", "show rules"}:
            tool("read_rules", "started")
            rules = self.rules()
            tool("read_rules", "done", f"{len(rules)} loaded")
            if not rules:
                say("No rules loaded.")
            else:
                say(
                    f"{len(rules)} rule(s) loaded:\n"
                    + "\n".join(
                        f"  {'on ' if r.active else 'off'}  {r.id:<16} "
                        f"{r.stage}/{r.target.value:<9} {r.description}  [{r.author}]"
                        for r in rules
                    )
                )
            return {"ok": True}

        if low in {"facts", "show facts"}:
            tool("read_facts", "started")
            facts = self.facts()
            tool("read_facts", "done", f"{len(facts.fields)} field(s)")
            say("\n".join(f"  {k}: {v!r}" for k, v in sorted(facts.fields.items())))
            return {"ok": True}

        if low in {"test", "test connection", "connect"}:
            tool("test_connection", "started", self.config.provider or "(no provider)")
            probe = self.test_connection()
            tool("test_connection", "done" if probe["ok"] else "failed", probe["headline"])
            say(
                f"**{probe['headline']}.** {probe['detail']}"
                + (f"\n{probe['fix']}" if probe.get("fix") else "")
            )
            return {"ok": True, "probe": probe}

        if low in {"status", "setup"}:
            s = self.setup()
            say(
                f"provider: {s['provider'] or 'not chosen'}\n"
                f"key: {s['key_detail']}\n"
                f"next step: {s['next_step'].replace('_', ' ')}\n"
                f"ledger: {s['home']}"
            )
            return {"ok": True}

        # Reached two ways: no key at all, or a model call that failed above.
        # They need different sentences - claiming "no key saved" to someone
        # whose key is stored and working is simply false.
        rules = self.rules()
        active = [r for r in rules if r.active]
        has_key = bool(self.config.key(home=self.home))
        why = (
            "the model call above failed, so nothing could read it"
            if has_key
            else "there's no key saved, so nothing is reading your words"
        )
        say(
            f"I can't interpret “{text}” — {why}. That is the only reason this "
            "reply is a menu instead of an answer.\n\n"
            "**What this is.** A ledger for business ideas. You describe what you "
            "want to evaluate; candidates are generated, gated against your own "
            "rules, checked for real competitors, then scored. The scoring is "
            "plain code — no model assigns a number, so the same evidence always "
            "produces the same score.\n\n"
            f"**What's already yours.** {len(active)} active rule(s) and "
            f"{len(self.facts().fields)} facts, both editable in the panels on the "
            "right. Gates run before any model sees an idea, which is why a "
            "rejection can name the rule that caused it.\n\n"
            + ("**To get past this:** open Setup and press Save & test; the model "
               "is re-resolved from your key each time, so a retired one is replaced. "
               if has_key else
               "**To get past this:** open Setup, choose a provider and save a key. ")
            + "Then say what you just said, in your own words, and it will be read "
              "properly. Meanwhile `run <brief>`, `rules`, `facts` and `status` work."
        )
        return {"ok": True}

    def _agent_chat(self, text: str, evaluator: Any) -> dict[str, Any]:
        """Route the message through the model, with the ledger's tools bound."""
        from .chat_agent import ChatAgent
        from .providers.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(
            provider=self.config.provider,
            api_key=self.config.key(home=self.home) or "",
            model=self.config.model_id or DEFAULT_ROUTE.get(self.config.provider, ""),
            on_retry=lambda note: (
                self.log(note),
                pub_retry({"type": "chat", "kind": "note", "text": note}),
            ),
        )
        pub = self.bus.publish
        pub_retry = self.bus.publish
        agent = ChatAgent(
            client=client,
            tools={
                # why_demo() rides along so the model can tell the operator the
                # cause instead of reporting fictional names as findings. It is
                # the difference between "3 candidates" and "3 fictional
                # candidates, because this provider cannot search".
                "run_pipeline": lambda brief="an idea": {
                    "results": self.run(brief, self.live_evaluator() or evaluator),
                    "evaluator": "model" if self.live_evaluator() else "demo (fictional)",
                    "demo_reason": self.why_demo(),
                },
                "list_rules": lambda: {"rules": self.state()["rules"]},
                "read_facts": lambda: {"facts": dict(self.facts().fields)},
                "list_ideas": lambda: {"ideas": self.state()["ideas"]},
                "read_idea": lambda id="": self.dossier(id),
                "check_connection": self.test_connection,
                "system_status": self.setup,
                "explain_rubric": self.explain_rubric,
                "activate_rule": lambda rule_id="", breaking_confirmed=False: (
                    self.activate_rule(rule_id, confirmed=True, breaking=bool(breaking_confirmed))
                ),
                "preview_rule": self._preview_from_agent,
            },
            emit_text=lambda t: pub(
                {"type": "chat", "role": "agent", "kind": "text", "text": t, "final": True}
            ),
            emit_delta=lambda d: pub({"type": "chat", "kind": "delta", "text": d}),
            emit_think=lambda d: pub({"type": "chat", "kind": "reasoning", "text": d}),
            emit_tool=lambda name, state, detail: pub(
                {"type": "chat", "role": "agent", "kind": "tool",
                 "tool": name, "state": state, "detail": detail}
            ),
            history=self.history,
        )
        out = agent.ask(text)
        self.history = agent.history[-40:]   # bounded; the ledger is the memory
        return out

    def _preview_from_agent(
        self, description: str, field: str, op: str, value: str | None = None
    ) -> dict[str, Any]:
        leaf: dict[str, Any] = {"field": field, "op": op}
        if op not in {"falsy", "truthy", "exists", "missing"}:
            try:
                leaf["value"] = int(value) if value is not None and str(value).lstrip("-").isdigit() else value
            except Exception:
                leaf["value"] = value
        return self.preview_rule(
            {
                "raw": description, "target": "symbolic", "description": description,
                "reasoning": "proposed in chat", "inputs": [field], "predicate": leaf,
            }
        )

    def find_route(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Probe real routes until one answers, then adopt it.

        Replaces guesswork: a route can be listed, free, and still unusable
        because of an account provider allowlist or an upstream rate limit,
        and only a real call reveals which.
        """
        from dataclasses import replace as _replace
        from .providers.openai_compat import find_working_route

        payload = payload or {}
        key = self.config.key(home=self.home)
        if not key or self.config.provider not in _PROBEABLE:
            return {"error": "save a working key for openai or openrouter first"}

        # Free routes are tried first because they cost nothing, but the search
        # does not stop there: on a free tier no route satisfies every stage, so
        # restricting the probe to free routes guarantees a partial pipeline.
        free_only = bool(payload.get("free_only", False))
        self.log(f"probing {'free' if free_only else 'free-first, then paid'} routes")
        winner, attempts = find_working_route(
            key,
            provider=self.config.provider,
            free_only=free_only,
            on_try=lambda m, note: self.bus.publish(
                {"type": "chat", "kind": "tool", "tool": f"probe {m}",
                 "state": "started" if note == "trying" else
                          ("done" if note == "works" else "failed"),
                 "detail": "" if note == "trying" else note}
            ),
        )
        if winner:
            self.config = _replace(self.config, model_id=winner)
            self.config.save(self.home)
            self.log(f"route set to {winner}")
        return {
            "route": winner,
            "attempts": [{"model": m, "result": r} for m, r in attempts],
            **self.setup(),
        }

    def explain_rubric(self) -> dict[str, Any]:
        """The rubric as configured here, so explanations are read not invented."""
        return {
            "rubric_version": RUBRIC.version,
            "dimensions": list(RUBRIC.dimensions),
            "max_total": RUBRIC.max_total,
            "tracks": list(RUBRIC.tracks),
            "thresholds": [
                {"dimension": t.dimension, "minimum": t.minimum} for t in THRESHOLDS
            ],
            "rules_note": (
                "Ranking happens within a track only; two tracks are not on one "
                "scale. Adding or redefining a dimension is a breaking change: "
                "stored scores stop being comparable and cannot be backfilled."
            ),
            "pipeline": self.manifest.report(),
        }

    def list_models(self) -> dict[str, Any]:
        """Real routes for the configured provider, so nobody guesses an id."""
        from .providers.openai_compat import OpenAICompatClient

        key = self.config.key(home=self.home)
        if not key or self.config.provider not in _COMPAT:
            return {"models": []}
        client = OpenAICompatClient(self.config.provider, key, self.config.model_id or "")
        return {"models": client.models()}

    def forget_key(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        from . import secrets as secret_store

        payload = payload or {}
        account = str(payload.get("account", "")).strip() or (
            self.config.key_ref.account or self.config.provider
        )
        if account:
            secret_store.forget(self.home, account)
        return {"ok": True, **self.setup()}

    def use_key(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Switch to a key already saved, without retyping it.

        Selecting a key implies its provider: a stored OpenRouter key is not a
        Google credential, and quietly keeping the old provider would send it to
        the wrong endpoint.
        """
        from . import secrets as secret_store

        account = str(payload.get("account", "")).strip()
        match = next(
            (k for k in secret_store.saved(self.home) if k.account == account), None
        )
        if match is None:
            return {"error": f"no saved key called {account!r}", **self.setup()}

        cfg = self.config.with_provider(match.provider)
        cfg = replace(
            cfg, key_ref=replace(cfg.key_ref, source=KeySource.KEYRING, account=account)
        )
        cfg.save(self.home)
        self.config = cfg
        self.log(f"using saved key {match.label} ({match.provider})")
        return {"ok": True, "using": match.label, **self.setup()}

    def dossier(self, idea_id: str) -> dict[str, Any]:
        idea = next((i for i in self.store.load_ideas() if i.id == idea_id), None)
        if idea is None:
            return {"error": f"unknown idea {idea_id}"}
        out: dict[str, Any] = {
            "id": idea.id,
            "name": idea.name,
            "track": idea.track,
            "status": idea.status.value,
            "dimensions": [],
        }
        if idea.vector:
            out["dimensions"] = [
                {
                    "dimension": s.dimension,
                    "value": s.value,
                    "confidence": s.confidence,
                    "falsifier": s.falsifier,
                    "provenance": s.provenance.value,
                    "original": s.original_value,
                    "evidence": list(s.evidence_ids),
                }
                for s in idea.vector.scores
            ]
            out["rubric_version"] = idea.vector.rubric_version
            if not RUBRIC.validates(idea.vector):
                s = score(idea, RUBRIC, THRESHOLDS)
                out["total"] = s.total
                out["max"] = s.max_total
                out["thresholds_met"] = s.thresholds_met
                out["failed"] = list(s.failed_thresholds)

        # The archived document is meant to be read away from this app, where
        # "evidence: e3, e7" identifies nothing. Resolve the ids to the claims
        # and urls they stand for.
        seen: dict[str, Any] = {}
        for ev in getattr(idea, "evidence", ()) or ():
            seen[ev.id] = {"id": ev.id, "claim": getattr(ev, "claim", ""),
                           "url": getattr(ev, "url", ""),
                           "name": getattr(ev, "name", "")}
        out["evidence"] = list(seen.values())

        graph = self.store.load_graph()
        out["assumptions"] = [
            {"id": r.assumption_id, "statement": graph.assumptions[r.assumption_id].statement}
            for r in graph.load_bearing_unverified()
            if r.assumption_id in graph.assumptions
        ]
        return out

    # -- actions ---------------------------------------------------------
    def why_demo(self) -> str:
        """Why this run will use fictional candidates, in one sentence.

        Empty when it will not. Labelling rows "demo-" said *that* it was demo
        and never *why*, which reads as a bug when your key is valid and your
        connection test is green. The cause is almost never the key.
        """
        if self.live_evaluator() is not None:
            return ""
        provider = self.config.provider or "no provider"
        if not self.config.key(home=self.home):
            return ("No key is saved, so there is nothing to ask. The pipeline is "
                    "real; the candidates are fictional and labelled demo-.")
        if provider not in _COMPAT and provider != "anthropic":
            return (f"{provider} has no evaluator in this build, so candidates are "
                    "fictional and labelled demo-.")
        return (
            f"{provider} cannot search the web here, and the evidence stage needs "
            "citable competitors. Rather than let a model invent them, the run uses "
            "fictional candidates labelled demo-. Everything else is real: your gates "
            "ran in code, and the score was computed, not guessed. For real "
            "candidates, use a provider that can search - Anthropic can."
        )

    def live_evaluator(self) -> Any | None:
        """A model-backed evaluator, when the provider can actually do the job.

        Returns None rather than a half-working one: without server search the
        evidence stage cannot be satisfied, and an evaluator that invents
        competitors is worse than an obviously-labelled demo.
        """
        key = self.config.key(home=self.home)
        if not key:
            return None
        from .evaluator import ModelEvaluator
        from .providers.base import Capability

        backend: Any
        if self.config.provider == "anthropic":
            # Anthropic does not speak the OpenAI dialect, so it has its own
            # transport - which existed, declared every capability the pipeline
            # wants, and was wired to nothing. A valid Anthropic key therefore
            # produced demo output: the one provider that can run every stage
            # was the only one that could not reach this function.
            from .providers.anthropic_transport import AnthropicTransport

            backend = AnthropicTransport(
                api_key=key, model_id=self.config.model_id or ANTHROPIC_MODEL,
            )
        elif self.config.provider in _COMPAT:
            from .providers.compat_adapter import CompatProvider

            backend = CompatProvider(
                provider=self.config.provider,
                api_key=key,
                model=self.config.model_id or DEFAULT_ROUTE.get(self.config.provider, ""),
                on_retry=self.log,
            )
        else:
            return None

        if Capability.SERVER_SEARCH not in backend.capabilities():
            return None
        return ModelEvaluator(
            provider=backend,
            rubric=RUBRIC,
            facts=self.facts(),
            rules=self.store.load_rules(active_only=True),
            calibration=self.store.load_calibration(),
        )

    def run(self, brief: str, evaluator: Any) -> list[dict[str, Any]]:
        started = time.monotonic()
        model_stages = {s.id for s in self.manifest.model_stages}
        self.bus.publish({"type": "run_start", "brief": brief})
        pipeline = Pipeline(
            manifest=self.manifest,
            facts=self.facts(),
            rubric=RUBRIC,
            rules=self.store.load_rules(active_only=True),
            thresholds=THRESHOLDS,
            evaluator=evaluator,
            coverage=CoveragePolicy(),
            rejected_ids=self.store.rejected_ids(),
            calibration=self.store.load_calibration(),
            on_event=lambda cid, e: self.bus.publish(
                {"type": "stage", "candidate": cid, "stage": e.stage,
                 "state": e.state, "detail": e.detail,
                 "kind": "model" if e.stage in model_stages else "code",
                 "t": round(time.monotonic() - started, 2)}
            ),
        )
        results = pipeline.run(brief, count=3)
        summary = []
        for r in results:
            if r.idea is not None:
                self.store.save_idea(r.idea)
            if r.outcome is not Outcome.SCORED:
                self.store.record_rejection(r.candidate.id, r.outcome.value, r.explanation)
            summary.append(
                {"id": r.candidate.id, "name": r.candidate.name,
                 "outcome": r.outcome.value, "why": r.explanation}
            )
        self.bus.publish({"type": "run_complete", "results": summary})
        return summary

    def preview_rule(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Compile a rule from a raw A5-shaped payload and dry-run it."""
        intake = RuleIntake(manifest=self.manifest, facts=self.facts())
        proposal = RuleIntake.from_payload(str(payload.get("raw", "")), payload)
        rule, placement = intake.resolve(
            proposal, declared_fields=payload.get("declared_fields", ())
        )
        preview = intake.preview(rule, placement, self.store.load_ideas(), self.rules())
        self.pending[rule.id] = (proposal, preview)
        return {
            "rule_id": rule.id,
            "target": proposal.target,
            "reasoning": proposal.reasoning,
            "stage": placement.stage,
            "breaking": preview.breaking,
            "report": preview.report(),
        }

    def activate_rule(self, rule_id: str, *, confirmed: bool, breaking: bool) -> dict[str, Any]:
        entry = self.pending.get(rule_id)
        if entry is None:
            return {"error": f"no pending rule {rule_id}; preview it first"}
        _, preview = entry
        rule = RuleIntake.activate(preview, confirmed=confirmed, breaking_confirmed=breaking)
        self.store.save_rule(rule)
        self.pending.pop(rule_id, None)
        return {"activated": rule.id, "stage": rule.stage, "target": rule.target.value}

    def override(self, idea_id: str, dimension: str, value: int, reason: str) -> dict[str, Any]:
        ideas = self.store.load_ideas()
        idea = next((i for i in ideas if i.id == idea_id), None)
        if idea is None or idea.vector is None:
            return {"error": "no scored idea with that id"}
        scores = tuple(
            s.override(value, reason=reason) if s.dimension == dimension else s
            for s in idea.vector.scores
        )
        from dataclasses import replace

        updated = replace(idea, vector=replace(idea.vector, scores=scores))
        self.store.save_idea(updated)
        return {"ok": True, "dimension": dimension, "provenance": Provenance.OVERRIDDEN.value}

    def archive(self, idea_id: str) -> dict[str, Any]:
        """Operator-triggered snapshot: idea-scoped, timestamped."""
        from datetime import datetime, timezone

        idea = next((i for i in self.store.load_ideas() if i.id == idea_id), None)
        if idea is None:
            return {"error": f"unknown idea {idea_id}"}
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
        target = self.home / "archive" / idea.id / stamp
        (target / "evidence").mkdir(parents=True, exist_ok=True)

        snapshot = self.dossier(idea_id)
        (target / "snapshot.json").write_text(
            json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
        )
        (target / "dossier.md").write_text(_markdown(snapshot), encoding="utf-8")

        index = self.home / "archive" / "index.json"
        entries = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
        entries.append({"idea": idea.id, "at": stamp, "path": str(target.relative_to(self.home))})
        index.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        return {"archived": str(target)}


#: Provider failures an operator can act on, matched on the provider's own
#: wording. The point is not classification for its own sake: each of these has
#: a different fix, and a run that dies on one of them should say which.
_FAILURE_HINTS: tuple[tuple[str, str], ...] = (
    ("credit balance is too low",
     "Add credit to that provider's account, or switch to another saved key in Setup."),
    ("rate-limiting", "Wait and retry, or move to a paid route."),
    ("exceeded your current quota",
     "That is a free-tier quota, not your key. Wait for it to reset, or add billing."),
    ("rejected the key", "Re-check the key in Setup; the connection test will confirm it."),
    ("no route called",
     "That model id is not available to this account. Press Save & test to re-resolve it."),
    ("thought_signature",
     "A tool call was replayed without its provider state. Start a new message."),
)


def _explain_failure(exc: BaseException) -> tuple[str, str]:
    """(message, fix). Strips the exception class and adds the next action."""
    raw = str(exc).strip() or type(exc).__name__
    # Transports already phrase their errors for a person; the class name in
    # front is the only thing making them look like a crash.
    for prefix in ("TransportUnavailable: ", "ChatError: ", "ModelRefused: "):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    low = raw.lower()
    for needle, fix in _FAILURE_HINTS:
        if needle in low:
            return raw, fix
    return raw, ""


def _markdown(dossier: Mapping[str, Any]) -> str:
    """The archived document, written to be read away from this app.

    A bare score table is not a dossier. Someone opening this file weeks later
    needs to know what produced the number before they can trust or argue with
    it: which rubric, from what evidence, and whether a human moved anything.
    """
    from datetime import datetime, timezone

    name = dossier.get("name", "?")
    lines = [f"# {name}", ""]

    facts: list[str] = []
    if dossier.get("track"):
        facts.append(f"**Track** {dossier['track']}")
    if dossier.get("status"):
        facts.append(f"**Status** {str(dossier['status']).replace('_', ' ')}")
    if dossier.get("rubric_version") is not None:
        facts.append(f"**Rubric** v{dossier['rubric_version']}")
    facts.append("**Archived** " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    if facts:
        lines += [" &middot; ".join(facts), ""]

    if dossier.get("total") is not None:
        lines += [f"## Score {dossier['total']} of {dossier.get('max', '?')}", ""]
        # Scores only compare inside a track and inside a rubric version. Saying
        # so in the file stops the number being carried somewhere it means
        # nothing - which is the failure mode of every shortlist spreadsheet.
        lines += [
            "> Computed by code from the evidence below, not assigned by a model. "
            "Comparable only against other ideas on the same track and rubric version.",
            "",
        ]
    elif dossier.get("status"):
        lines += [
            f"## Not scored", "",
            "> This idea has no score. That is a result, not a gap: the pipeline "
            "refuses to score an idea whose evidence did not meet the coverage "
            "policy, rather than scoring it thinly.",
            "",
        ]

    dims = dossier.get("dimensions", [])
    if dims:
        lines += ["## Dimensions", "",
                  "| dimension | value | confidence | provenance | what would change it |",
                  "|---|---:|---:|---|---|"]
        for d in dims:
            lines.append(
                f"| {d['dimension']} | {d['value']} | {d.get('confidence', '')} | "
                f"{d.get('provenance', '')} | {d.get('falsifier', '')} |"
            )
        overridden = [d for d in dims if d.get("provenance") == "overridden"]
        if overridden:
            lines += ["", "### Overridden by hand", ""]
            for d in overridden:
                # An override stores its reason in the falsifier slot, which is
                # the right place - the reason *is* what would have to change -
                # but the summary must show the value it replaced, or the record
                # of the judgement is incomplete.
                was = d.get("original")
                lines.append(
                    f"- **{d['dimension']}** set to {d['value']}"
                    + (f", was {was}" if was is not None else "")
                    + (f" — {d['falsifier']}" if d.get("falsifier") else "")
                )
            lines += ["",
                      "> Overridden dimensions are excluded from calibration, so a "
                      "judgement call never teaches the system it was a measurement.",
                      ]
        lines.append("")

    ev = dossier.get("evidence") or []
    lines += ["## Evidence", ""]
    if ev:
        for e in ev:
            url = e.get("url") or ""
            claim = e.get("claim") or e.get("name") or e.get("id") or "?"
            lines.append(f"- {claim}" + (f" — <{url}>" if url else ""))
    else:
        lines.append("_None recorded._ A score resting on no citable evidence is "
                     "the thing this ledger exists to make visible.")
    lines.append("")

    if dossier.get("assumptions"):
        lines += ["## Load-bearing assumptions", ""]
        for a in dossier["assumptions"]:
            lines.append(f"- {a.get('statement', a)}")
        lines.append("")

    if "(demo)" in str(name):
        lines += ["---", "",
                  "_Demo data. Generated without a search-capable provider, so the "
                  "competitors and evidence are fictional and labelled as such._", ""]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
def make_handler(ws: Workspace, evaluator_factory: Callable[[], Any]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "IdeaLedger/0.1"

        def log_message(self, *args: Any) -> None:  # quieter console
            pass

        # -- helpers ---------------------------------------------------
        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=_encode).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        # -- routes ----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path)
            params = parse_qs(route.query)

            if route.path in ("/", "/index.html"):
                return self._static("index.html", "text/html; charset=utf-8")
            if route.path == "/api/state":
                return self._json(ws.state())
            if route.path == "/api/setup":
                return self._json(ws.setup())
            if route.path == "/api/idea":
                return self._json(ws.dossier(params.get("id", [""])[0]))
            if route.path == "/api/models":
                return self._json(ws.list_models())
            if route.path == "/api/logs":
                return self._json({"lines": ws.logs[-200:]})
            if route.path == "/api/events":
                return self._events()
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path)
            try:
                body = self._body()
            except json.JSONDecodeError:
                return self._json({"error": "body was not valid json"}, 400)

            try:
                if route.path == "/api/run":
                    ev = ws.live_evaluator() or evaluator_factory()
                    return self._json({"results": ws.run(body.get("brief", "an idea"), ev)})
                if route.path == "/api/setup":
                    return self._json(ws.save_setup(body))
                if route.path == "/api/chat":
                    return self._json(
                        ws.chat(body.get("message", ""), evaluator_factory())
                    )
                if route.path == "/api/test-connection":
                    return self._json(ws.test_connection())
                if route.path == "/api/find-route":
                    return self._json(ws.find_route(body))
                if route.path == "/api/use-key":
                    return self._json(ws.use_key(body))
                if route.path == "/api/forget-key":
                    return self._json(ws.forget_key(body))
                if route.path == "/api/rule/preview":
                    return self._json(ws.preview_rule(body))
                if route.path == "/api/rule/activate":
                    return self._json(
                        ws.activate_rule(
                            body.get("rule_id", ""),
                            confirmed=bool(body.get("confirmed")),
                            breaking=bool(body.get("breaking_confirmed")),
                        )
                    )
                if route.path == "/api/override":
                    return self._json(
                        ws.override(
                            body.get("idea", ""), body.get("dimension", ""),
                            int(body.get("value", 0)), body.get("reason", ""),
                        )
                    )
                if route.path == "/api/archive":
                    return self._json(ws.archive(body.get("idea", "")))
            except PermissionError as exc:
                # A refused activation is a policy decision, not a malformed
                # request: 409 so a client can tell "you must confirm" apart
                # from "you sent nonsense".
                return self._json({"error": str(exc), "needs_confirmation": True}, 409)
            except Exception as exc:  # surfaced to the operator, not swallowed
                # A provider saying "no credit" is not a malformed request, and
                # prefixing it with a Python class name buries the one sentence
                # that tells you what to do. Real example this replaces:
                #   TransportUnavailable: anthropic returned 400: Your credit
                #   balance is too low to access the Anthropic API.
                message, fix = _explain_failure(exc)
                return self._json(
                    {"error": message, "fix": fix, "kind": type(exc).__name__}, 400
                )

            self._json({"error": "not found"}, 404)

        # -- transports ------------------------------------------------
        def _static(self, name: str, content_type: str) -> None:
            path = STATIC / name
            if not path.exists():
                return self._json({"error": f"missing static asset {name}"}, 404)
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # A local tool that changes under the operator's feet must never be
            # served from cache; a stale script looks exactly like a dead app.
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _events(self) -> None:
            q = ws.bus.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    try:
                        line = q.get(timeout=15)
                        self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")  # stops proxies idling out
                    self.wfile.flush()
            except (ConnectionError, OSError):
                # A reload, a closed tab, or a sleeping laptop aborts the stream
                # mid-write. Every variant of that is normal and means exactly
                # one thing: this subscriber is gone. Listing the errno classes
                # individually missed ConnectionAbortedError, which is the one
                # Windows raises (WinError 10053), so a page reload dumped a
                # traceback. ConnectionError is the common base of all of them.
                pass
            finally:
                ws.bus.unsubscribe(q)

        def handle_one_request(self) -> None:
            # Same reasoning one level up: the client can vanish between the
            # response line and the body, and that is not a server fault.
            try:
                super().handle_one_request()
            except (ConnectionError, OSError):
                self.close_connection = True

    return Handler


def _encode(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (Status, Provenance, Outcome, RuleTarget, ChangeClass)):
        return obj.value
    return str(obj)


def serve(
    host: str = "127.0.0.1",
    port: int = 8420,
    *,
    home: Path | None = None,
    evaluator_factory: Callable[[], Any] | None = None,
    open_browser: bool = True,
) -> None:
    from .demo import demo_evaluator

    ws = Workspace(home)
    handler = make_handler(ws, evaluator_factory or demo_evaluator)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Idea Ledger  ->  {url}")
    print(f"ledger       ->  {ws.home}")
    print(f"next step    ->  {ws.config.next_step().value}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
