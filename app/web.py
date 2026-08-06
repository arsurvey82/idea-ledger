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

from .config import DEFAULT_ENV_VARS, Config, KeySource, user_dir, validate_key
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

STATIC = Path(__file__).resolve().parent / "static"
RUBRIC = Rubric(
    version=1,
    dimensions=("demand", "competition", "ease", "capital", "profit", "solo_marketing"),
    tracks=("service", "physical"),
)
THRESHOLDS = (Threshold("ease", 7), Threshold("solo_marketing", 7))

#: Used when the operator has not named a route. OpenRouter's free tier means a
#: working chat before anyone has spent anything.
DEFAULT_ROUTE = {"openrouter": "openrouter/free", "openai": "gpt-4o-mini"}


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

    def setup(self) -> dict[str, Any]:
        from . import secrets as secret_store

        provider_name = self.config.provider or "anthropic"
        try:
            backend = get_provider(provider_name)
            if self.config.model_id:
                # A named route is part of the provider's identity, so the
                # capability report must show it rather than "(route not chosen)".
                backend = with_capabilities(
                    backend, backend.capabilities(), self.config.model_id
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

        stored = ""
        key = str(payload.get("key", "")).strip()
        if key:
            # Checked before it reaches the store, so a stray paste is refused
            # while the operator still has the clipboard, rather than surfacing
            # later as a wall of "rejected the key" from the provider.
            key, complaint = validate_key(cfg.provider, key)
            if complaint:
                return {"error": complaint, **self.setup()}
            try:
                info = secret_store.store(self.home, cfg.key_ref.account or provider, key)
                cfg = replace(cfg, key_ref=replace(cfg.key_ref, source=KeySource.KEYRING))
                stored = f"{info.backend}: {info.detail}"
            except secret_store.SecretStoreUnavailable as exc:
                # Refuse rather than silently write plaintext.
                return {"error": f"not stored: {exc}", "env_var": cfg.key_ref.env_var}

        cfg.save(self.home)
        self.config = cfg
        return {"ok": True, "stored": stored, **self.setup()}

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
        if text and self.config.key(home=self.home) and self.config.provider in {
            "openai", "openrouter"
        }:
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
            has_key = self.config.key(home=self.home) is not None
            say(
                f"Evaluating candidates for **{brief}**."
                + ("" if has_key else " No key is configured, so this uses the labelled "
                   "demo evaluator - the pipeline is real, the candidates are fictional.")
            )
            tool("run_pipeline", "started", brief)
            results = self.run(brief, evaluator)
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

        say(
            f"I do not have a command for “{text}” yet, and I would rather say so "
            "than guess. Try `help`, or use the panels on the right."
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
                "run_pipeline": lambda brief="an idea": {
                    "results": self.run(brief, self.live_evaluator() or evaluator),
                    "evaluator": "model" if self.live_evaluator() else "demo (fictional)",
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

    def find_route(self) -> dict[str, Any]:
        """Probe real routes until one answers, then adopt it.

        Replaces guesswork: a route can be listed, free, and still unusable
        because of an account provider allowlist or an upstream rate limit,
        and only a real call reveals which.
        """
        from dataclasses import replace as _replace
        from .providers.openai_compat import find_working_route

        key = self.config.key(home=self.home)
        if not key or self.config.provider not in {"openai", "openrouter"}:
            return {"error": "save a working key for openai or openrouter first"}

        self.log("probing routes for one that answers")
        winner, attempts = find_working_route(
            key,
            provider=self.config.provider,
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
        if not key or self.config.provider not in {"openai", "openrouter"}:
            return {"models": []}
        client = OpenAICompatClient(self.config.provider, key, self.config.model_id or "")
        return {"models": client.models()}

    def forget_key(self) -> dict[str, Any]:
        from . import secrets as secret_store

        account = self.config.key_ref.account or self.config.provider
        if account:
            secret_store.delete(self.home, account)
        return {"ok": True, **self.setup()}

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
            if not RUBRIC.validates(idea.vector):
                s = score(idea, RUBRIC, THRESHOLDS)
                out["total"] = s.total
                out["max"] = s.max_total
                out["thresholds_met"] = s.thresholds_met
                out["failed"] = list(s.failed_thresholds)
        return out

    # -- actions ---------------------------------------------------------
    def live_evaluator(self) -> Any | None:
        """A model-backed evaluator, when the provider can actually do the job.

        Returns None rather than a half-working one: without server search the
        evidence stage cannot be satisfied, and an evaluator that invents
        competitors is worse than an obviously-labelled demo.
        """
        key = self.config.key(home=self.home)
        if not key or self.config.provider not in {"openai", "openrouter"}:
            return None
        from .evaluator import ModelEvaluator
        from .providers.base import Capability
        from .providers.compat_adapter import CompatProvider

        backend = CompatProvider(
            provider=self.config.provider,
            api_key=key,
            model=self.config.model_id or DEFAULT_ROUTE.get(self.config.provider, ""),
            on_retry=self.log,
        )
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


def _markdown(dossier: Mapping[str, Any]) -> str:
    lines = [f"# {dossier.get('name', '?')}", ""]
    if "total" in dossier:
        lines.append(f"**{dossier['total']}/{dossier['max']}** &middot; {dossier['track']} track")
        lines.append("")
    lines += ["| dimension | value | confidence | provenance | falsifier |",
              "|---|---|---|---|---|"]
    for d in dossier.get("dimensions", []):
        lines.append(
            f"| {d['dimension']} | {d['value']} | {d['confidence']} | "
            f"{d['provenance']} | {d['falsifier']} |"
        )
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
                    return self._json(ws.find_route())
                if route.path == "/api/forget-key":
                    return self._json(ws.forget_key())
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
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

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
