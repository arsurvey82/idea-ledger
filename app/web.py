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
import threading
import webbrowser
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlparse

from .config import Config, user_dir
from .core.assumptions import AssumptionGraph, summarise as summarise_assumptions
from .core.manifest import Manifest
from .core.rules import ChangeClass, Rule, RuleTarget
from .core.scoring import Threshold, score
from .core.types import FactBase, Idea, Provenance, Rubric, Status
from .pipeline import Candidate, CoveragePolicy, Outcome, Pipeline, StageEvent
from .providers import default_requirements, get as get_provider, negotiate
from .rule_intake import ImpactPreview, RuleIntake, RuleProposal
from .store import Store

STATIC = Path(__file__).resolve().parent / "static"
RUBRIC = Rubric(
    version=1,
    dimensions=("demand", "competition", "ease", "capital", "profit", "solo_marketing"),
    tracks=("service", "physical"),
)
THRESHOLDS = (Threshold("ease", 7), Threshold("solo_marketing", 7))


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
        provider_name = self.config.provider or "anthropic"
        try:
            report = negotiate(get_provider(provider_name), default_requirements()).report()
        except KeyError as exc:
            report = str(exc)
        return {
            "config": self.config.describe(),
            "negotiation": report,
            "next_step": self.config.next_step().value,
            "home": str(self.home),
        }

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
    def run(self, brief: str, evaluator: Any) -> list[dict[str, Any]]:
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
                 "state": e.state, "detail": e.detail}
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
                    return self._json(
                        {"results": ws.run(body.get("brief", "an idea"), evaluator_factory())}
                    )
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
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                ws.bus.unsubscribe(q)

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
