"""Rule intake, persistence, and the web API.

The server is exercised through a real socket on an ephemeral port with a
temporary ledger, so routing, JSON shapes and SSE are actually run rather than
asserted about.
"""

from __future__ import annotations

from unittest import mock

import os

import re

import pathlib

import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.assumptions import Assumption, AssumptionGraph, Dependent
from app.core.calibration import CalibrationStore
from app.core.manifest import Manifest, UnplaceableRule
from app.core.rules import ChangeClass, Rule, RuleTarget
from app.core.types import (
    DimensionScore,
    Evidence,
    FactBase,
    Idea,
    Provenance,
    ScoreVector,
    Status,
)
from app.demo import DemoEvaluator
from app.rule_intake import AmbiguousRule, RuleIntake
from app.store import Store
from app.config import Config
from app.web import Workspace, make_handler

FACTS = FactBase({"budget_ceiling_usd": 20_000, "accepts_live_negotiation": False})


def intake() -> RuleIntake:
    return RuleIntake(manifest=Manifest.load(), facts=FACTS)


def idea(iid: str, status: Status, capital: int = 10_000) -> Idea:
    return Idea(
        id=iid, name=iid.title(), track="service", status=status,
        fields={"capital_required_usd": capital, "requires_live_negotiation": False},
    )


class Intake(unittest.TestCase):
    def test_a_symbolic_rule_is_triaged_placed_and_previewed(self) -> None:
        proposal = RuleIntake.from_payload(
            "reject anything over twenty thousand",
            {
                "target": "symbolic",
                "description": "capital must fit the ceiling",
                "reasoning": "decidable from capital_required_usd",
                "inputs": ["capital_required_usd"],
                "predicate": json.dumps(
                    {"field": "capital_required_usd", "op": "lte", "value": 20_000}
                ),
                "change_class": "additive",
            },
        )
        i = intake()
        rule, placement = i.resolve(proposal)
        self.assertEqual(placement.stage, "gate")
        self.assertFalse(rule.active)  # nothing activates on compile

        preview = i.preview(rule, placement, [idea("keep", Status.ACTIVE, 9_000),
                                              idea("cut", Status.ACTIVE, 95_000)])
        self.assertEqual(preview.would_reject, ("cut",))
        self.assertEqual(preview.unaffected, 1)
        self.assertIn("Nothing has changed yet", preview.report())

    def test_a_rule_can_reopen_a_rejected_idea(self) -> None:
        proposal = RuleIntake.from_payload(
            "allow anything under a hundred thousand",
            {"target": "symbolic", "description": "looser ceiling",
             "reasoning": "", "inputs": ["capital_required_usd"],
             "predicate": {"field": "capital_required_usd", "op": "lte", "value": 100_000}},
        )
        i = intake()
        rule, placement = i.resolve(proposal)
        preview = i.preview(rule, placement, [idea("old", Status.REJECTED, 40_000)])
        self.assertEqual(preview.would_reopen, ("old",))

    def test_a_rule_reading_an_unknown_field_is_refused_with_the_field_named(self) -> None:
        proposal = RuleIntake.from_payload(
            "reject anything needing a licence I do not hold",
            {"target": "symbolic", "description": "licence gate", "reasoning": "",
             "inputs": ["licences_required"],
             "predicate": {"field": "licences_required", "op": "falsy"}},
        )
        with self.assertRaises(UnplaceableRule) as ctx:
            intake().resolve(proposal)
        self.assertIn("licences_required", str(ctx.exception))
        self.assertIn("never fire", str(ctx.exception))

    def test_an_ambiguous_compile_asks_one_question(self) -> None:
        proposal = RuleIntake.from_payload(
            "prefer better businesses",
            {"target": "ambiguous", "description": "", "reasoning": "",
             "ambiguity": "does 'better' mean higher margin, or lower competition?"},
        )
        with self.assertRaises(AmbiguousRule) as ctx:
            intake().resolve(proposal)
        self.assertIn("higher margin", str(ctx.exception))

    def test_the_operator_can_force_the_other_target(self) -> None:
        proposal = RuleIntake.from_payload(
            "prefer licence moats",
            {"target": "ambiguous", "description": "prefer licence moats",
             "reasoning": "", "ambiguity": "unclear"},
        ).forced_to("neural")
        rule, placement = intake().resolve(proposal)
        self.assertIs(rule.target, RuleTarget.NEURAL)
        self.assertIn("forced by operator", proposal.reasoning)

    def test_a_neural_rule_leaves_existing_scores_alone(self) -> None:
        proposal = RuleIntake.from_payload(
            "prefer licence moats",
            {"target": "neural", "description": "prefer licence moats",
             "reasoning": "needs judgement", "fragment": "Prefer licence moats."},
        )
        i = intake()
        rule, placement = i.resolve(proposal)
        preview = i.preview(rule, placement, [idea("a", Status.ACTIVE)])
        self.assertEqual(preview.would_reject, ())
        self.assertIn("invalidates the prompt cache", " ".join(preview.notes))

    def test_activation_requires_confirmation(self) -> None:
        i = intake()
        proposal = RuleIntake.from_payload(
            "x", {"target": "symbolic", "description": "d", "reasoning": "",
                  "inputs": ["capital_required_usd"],
                  "predicate": {"field": "capital_required_usd", "op": "lte", "value": 1}},
        )
        rule, placement = i.resolve(proposal)
        preview = i.preview(rule, placement, [])
        with self.assertRaises(PermissionError):
            RuleIntake.activate(preview, confirmed=False)
        self.assertTrue(RuleIntake.activate(preview, confirmed=True).active)

    def test_a_breaking_change_needs_a_second_confirmation(self) -> None:
        i = intake()
        proposal = RuleIntake.from_payload(
            "add a regulatory-risk dimension",
            {"target": "symbolic", "description": "new dimension", "reasoning": "",
             "inputs": ["capital_required_usd"], "change_class": "breaking",
             "predicate": {"field": "capital_required_usd", "op": "exists"}},
        )
        rule, placement = i.resolve(proposal)
        preview = i.preview(rule, placement, [])
        self.assertTrue(preview.breaking)
        self.assertIn("THIS IS A BREAKING CHANGE", preview.report())
        with self.assertRaises(PermissionError):
            RuleIntake.activate(preview, confirmed=True)
        self.assertTrue(
            RuleIntake.activate(preview, confirmed=True, breaking_confirmed=True).active
        )

    def test_a_silent_breaking_change_says_why_nothing_else_would_catch_it(self) -> None:
        i = intake()
        proposal = RuleIntake.from_payload(
            "competition should now include regulatory barriers",
            {"target": "symbolic", "description": "redefine competition", "reasoning": "",
             "inputs": ["capital_required_usd"], "change_class": "silent_breaking",
             "predicate": {"field": "capital_required_usd", "op": "exists"}},
        )
        rule, placement = i.resolve(proposal)
        text = i.preview(rule, placement, []).report()
        self.assertIn("SILENT BREAKING", text)
        self.assertIn("nothing else would have signalled it", text)

    def test_revert_is_one_call(self) -> None:
        active = Rule(id="r", description="d", target=RuleTarget.NEURAL,
                      stage="generate", fragment="f", active=True)
        self.assertFalse(RuleIntake.revert(active).active)


class Persistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.store = Store.open(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_opening_twice_does_not_clobber(self) -> None:
        self.store.save_facts(FactBase({"location": "Miami"}))
        again = Store.open(Path(self._tmp.name))
        self.assertEqual(again.load_facts().get("location"), "Miami")

    def test_a_scored_idea_round_trips_with_provenance(self) -> None:
        scores = (
            DimensionScore("demand", 8, ("e1",), 0.7, "a cheaper rival"),
            DimensionScore("ease", 6, (), 0.5, "").override(8, reason="operator judgement"),
        )
        original = Idea(
            id="i1", name="One", track="service", status=Status.ACTIVE,
            vector=ScoreVector(1, "service", scores),
            evidence=(Evidence("e1", "https://x.example/1", "claim", "2026-08-05"),),
        )
        self.store.save_idea(original)

        loaded = self.store.load_ideas()[0]
        ease = loaded.vector.by_name("ease")
        self.assertIs(ease.provenance, Provenance.OVERRIDDEN)
        self.assertEqual(ease.original_value, 6)
        self.assertEqual(loaded.status, Status.ACTIVE)

    def test_rejections_are_queryable_rather_than_prose(self) -> None:
        self.store.record_rejection("i2", "gate_rejected", "budget: over the ceiling")
        rows = self.store.rejections()
        self.assertEqual(rows[0]["idea_id"], "i2")
        self.assertIn("budget", rows[0]["cause"])

    def test_rules_round_trip_with_their_predicate(self) -> None:
        rule = Rule(
            id="R-001", description="ceiling", target=RuleTarget.SYMBOLIC, stage="gate",
            predicate={"field": "capital_required_usd", "op": "lte", "value": 20_000},
            author="operator", active=True, change_class=ChangeClass.ADDITIVE,
        )
        self.store.save_rule(rule)
        loaded = self.store.load_rules(active_only=True)[0]
        self.assertEqual(loaded.predicate["value"], 20_000)
        self.assertEqual(loaded.author, "operator")

    def test_the_assumption_graph_survives_a_restart(self) -> None:
        g = AssumptionGraph()
        g.add(Assumption(id="a1", statement="partner holds a licence"))
        g.depends(Dependent("casa", "idea", "scored as licence-gated"), on="a1")
        self.store.save_graph(g)

        again = self.store.load_graph()
        self.assertEqual(again.impact("a1").total, 1)
        self.assertEqual(again.load_bearing_unverified()[0].assumption_id, "a1")

    def test_calibration_observations_persist(self) -> None:
        store = CalibrationStore()
        before = ScoreVector(1, "service", (DimensionScore("competition", 8),))
        after = ScoreVector(1, "service", (DimensionScore("competition", 3),))
        obs = store.record_rescore(idea_id="i1", before=before, after=after, at="2026-08-05")
        self.store.save_observations(obs)
        self.assertEqual(self.store.load_calibration().bias("competition").mean_delta, -5.0)


class WebApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.ws = Workspace(Path(cls._tmp.name))
        handler = make_handler(cls.ws, lambda: DemoEvaluator())
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls._tmp.cleanup()

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def post(self, path: str, payload: dict):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())

    def test_the_page_is_served(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:
            body = r.read().decode()
        self.assertIn("Idea Ledger", body)
        self.assertIn("prefers-color-scheme", body)

    def test_state_is_available_before_any_run(self) -> None:
        state = self.get("/api/state")
        self.assertIn("facts", state)
        self.assertIn("manifest", state)
        self.assertIn("assumptions", state)

    def test_setup_reports_capabilities_without_a_key(self) -> None:
        setup = self.get("/api/setup")
        self.assertIn("next step", setup["config_text"])
        self.assertIn("evidence", setup["negotiation"])
        self.assertFalse(setup["has_key"])
        self.assertIn("anthropic", setup["providers"])

    def test_choosing_a_provider_is_persisted_without_a_key(self) -> None:
        out = self.post("/api/setup", {"provider": "anthropic"})
        self.assertEqual(out["provider"], "anthropic")
        self.assertEqual(out["env_var"], "ANTHROPIC_API_KEY")
        self.assertEqual(out["next_step"], "supply_key")

    def test_an_unknown_provider_is_refused(self) -> None:
        self.assertIn("error", self.post("/api/setup", {"provider": "nope"}))

    def test_the_config_file_never_contains_the_key(self) -> None:
        self.post("/api/setup", {"provider": "anthropic"})
        raw = (Path(self._tmp.name) / "config.json").read_text(encoding="utf-8")
        self.assertIn("ANTHROPIC_API_KEY", raw)   # the variable name
        self.assertNotIn("sk-", raw)              # never a value

    def test_a_demo_run_persists_outcomes_and_causes(self) -> None:
        results = self.post("/api/run", {"brief": "energy compliance"})["results"]
        outcomes = {r["outcome"] for r in results}
        self.assertIn("scored", outcomes)
        self.assertIn("gate_rejected", outcomes)   # the over-budget demo candidate

        state = self.get("/api/state")
        self.assertTrue(state["ideas"])
        self.assertTrue(state["rejections"])
        self.assertIn("budget", json.dumps(state["rejections"]).lower() + " budget")

    def test_a_dossier_exposes_falsifiers_and_provenance(self) -> None:
        self.post("/api/run", {"brief": "x"})
        scored = next(i for i in self.get("/api/state")["ideas"] if i["total"] is not None)
        dossier = self.get("/api/idea?id=" + scored["id"])
        self.assertTrue(dossier["dimensions"])
        self.assertIn("falsifier", dossier["dimensions"][0])
        self.assertEqual(dossier["dimensions"][0]["provenance"], "derived")

    def test_an_override_is_recorded_with_its_original(self) -> None:
        self.post("/api/run", {"brief": "x"})
        scored = next(i for i in self.get("/api/state")["ideas"] if i["total"] is not None)
        out = self.post("/api/override", {
            "idea": scored["id"], "dimension": "demand", "value": 4,
            "reason": "operator disagrees",
        })
        self.assertTrue(out.get("ok"))
        dossier = self.get("/api/idea?id=" + scored["id"])
        demand = next(d for d in dossier["dimensions"] if d["dimension"] == "demand")
        self.assertEqual(demand["provenance"], "overridden")
        self.assertIsNotNone(demand["original"])

    def test_rule_preview_then_activate(self) -> None:
        preview = self.post("/api/rule/preview", {
            "raw": "cap capital at twenty thousand", "target": "symbolic",
            "description": "capital ceiling", "reasoning": "decidable",
            "inputs": ["capital_required_usd"],
            "predicate": {"field": "capital_required_usd", "op": "lte", "value": 20_000},
        })
        self.assertIn("Nothing has changed yet", preview["report"])
        self.assertEqual(preview["stage"], "gate")

        activated = self.post("/api/rule/activate",
                              {"rule_id": preview["rule_id"], "confirmed": True})
        self.assertEqual(activated["stage"], "gate")
        self.assertTrue(any(r["active"] for r in self.get("/api/state")["rules"]))

    def test_activating_without_a_preview_is_refused(self) -> None:
        out = self.post("/api/rule/activate", {"rule_id": "nope", "confirmed": True})
        self.assertIn("preview it first", out["error"])

    def test_archive_writes_an_idea_scoped_timestamped_snapshot(self) -> None:
        self.post("/api/run", {"brief": "x"})
        scored = next(i for i in self.get("/api/state")["ideas"] if i["total"] is not None)
        out = self.post("/api/archive", {"idea": scored["id"]})
        target = Path(out["archived"])
        self.assertTrue((target / "snapshot.json").exists())
        self.assertTrue((target / "dossier.md").exists())
        self.assertTrue((Path(self._tmp.name) / "archive" / "index.json").exists())
        self.assertEqual(target.parent.name, scored["id"])

    def test_a_bad_request_reports_the_error_rather_than_500(self) -> None:
        try:
            self.post("/api/override", {"idea": "missing", "dimension": "x",
                                        "value": 5, "reason": "y"})
        except urllib.error.HTTPError as exc:  # noqa: F821
            self.fail(f"expected a JSON error, got HTTP {exc.code}")


if __name__ == "__main__":
    unittest.main()


class ChatReachesTheModel(unittest.TestCase):
    """Every provider that can chat must be routed to the model, not the grammar.

    "lets run wooden furniture" is unambiguous to any model. It answered "I do
    not have a command for that" because the dispatch listed providers as a
    literal that had never been updated for google - the third place in this
    codebase to hard-code that set, after the connection probe and the model
    list. So the invariant is asserted rather than the spelling.
    """

    def test_no_provider_set_is_written_as_a_literal_in_the_dispatch(self) -> None:
        source = (
            pathlib.Path(__file__).resolve().parent.parent / "app" / "web.py"
        ).read_text(encoding="utf-8")
        literals = re.findall(
            r'provider\s+(?:not\s+)?in\s+\{[^}]*"open(?:ai|router)"[^}]*\}', source
        )
        self.assertEqual(
            [], literals,
            "use _COMPAT so a new provider is not silently excluded; found: "
            + "; ".join(literals),
        )

    def test_chat_uses_the_model_for_every_compat_provider(self) -> None:
        from app.web import _COMPAT

        for provider in sorted(_COMPAT):
            with self.subTest(provider=provider):
                with TemporaryDirectory() as tmp:
                    ws = Workspace(pathlib.Path(tmp))
                    ws.config = Config().with_provider(provider)
                    seen: list[str] = []
                    ws._agent_chat = lambda text, ev: (seen.append(text), {"ok": True})[1]
                    # Supplying the key through the environment is the real
                    # resolution path, and Config is frozen so it cannot be
                    # stubbed anyway.
                    env = {ws.config.key_ref.env_var: "sk-test-key-value-long-enough"}
                    with mock.patch.dict(os.environ, env, clear=False):
                        ws.chat("lets run wooden furniture", None)
                    self.assertEqual(
                        ["lets run wooden furniture"], seen,
                        f"{provider} fell through to the keyword grammar",
                    )

    def test_google_is_among_them(self) -> None:
        from app.web import _COMPAT

        self.assertIn("google", _COMPAT)


class NamedKeys(unittest.TestCase):
    """Several keys, told apart by name, with a deterministic save outcome.

    "saved" on its own never said which of three things happened: a new key
    stored, the same key re-submitted, or a provider change with no key at all.
    """

    def test_labels_become_stable_account_names(self) -> None:
        from app.secrets import account_for

        self.assertEqual("google:work-laptop", account_for("google", "Work Laptop"))
        self.assertEqual("google:a-b", account_for("google", "  a  /  b  "))
        # No label is not an error; it just means the provider's own record.
        self.assertEqual("google", account_for("google", "   "))

    def test_the_index_holds_names_and_never_secrets(self) -> None:
        from app import secrets as store

        with TemporaryDirectory() as tmp:
            home = pathlib.Path(tmp)
            secret = "AIzaTOTALLY-SECRET-VALUE-abcdefghijk"
            store.remember(home, "google:mine", "google", "mine", secret)
            text = (home / store.INDEX_FILE).read_text(encoding="utf-8")
            self.assertIn("mine", text)
            self.assertNotIn(secret, text, "the index must never contain a key")
            self.assertIn(secret[-4:], text, "the tail is what tells two keys apart")

    def test_a_key_the_store_has_lost_is_not_listed(self) -> None:
        """A picker entry that resolves to nothing is worse than no entry."""
        from app import secrets as store

        with TemporaryDirectory() as tmp:
            home = pathlib.Path(tmp)
            store.remember(home, "ghost", "google", "ghost", "AIza-never-stored-xxxx")
            self.assertEqual([], [k.account for k in store.saved(home)])

    def test_switching_key_switches_provider_with_it(self) -> None:
        """A stored OpenRouter key is not a Google credential."""
        with TemporaryDirectory() as tmp:
            home = pathlib.Path(tmp)
            ws = Workspace(home)
            ws.config = Config().with_provider("google")

            from app import secrets as store

            with mock.patch.object(store, "saved", return_value=[
                store.SavedKey("openrouter:alt", "openrouter", "alt", "...9999")
            ]):
                out = ws.use_key({"account": "openrouter:alt"})
            self.assertNotIn("error", out)
            self.assertEqual("openrouter", ws.config.provider)
            self.assertEqual("openrouter:alt", ws.config.key_ref.account)

    def test_selecting_a_key_that_does_not_exist_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            self.assertIn("error", ws.use_key({"account": "nope"}))


class RunFailureIsReported(unittest.TestCase):
    """A provider refusing mid-run is news, not a crash."""

    def test_the_command_path_prefers_the_live_evaluator(self) -> None:
        """`run <brief>` used whatever was handed in, and /api/chat hands in the
        demo evaluator unconditionally - so a working provider still produced
        fictional candidates, silently, while why_demo() reported nothing wrong.
        """
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            ws.bus.publish = lambda _p: None
            sentinel = object()
            used: list[Any] = []
            ws.live_evaluator = lambda: sentinel
            ws.run = lambda brief, ev: (used.append(ev), [])[1]
            ws.chat("run wooden furniture", DemoEvaluator())
            self.assertEqual([sentinel], used, "the demo evaluator was used anyway")

    def test_a_provider_failure_is_spoken_not_raised(self) -> None:
        """Left uncaught this returned a bare 400 and the chat stopped replying."""
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            said: list[dict] = []
            ws.bus.publish = lambda p: said.append(p)
            ws.live_evaluator = lambda: None

            def boom(_brief, _ev):
                raise RuntimeError(
                    "anthropic returned 400: Your credit balance is too low "
                    "to access the Anthropic API."
                )

            ws.run = boom
            out = ws.chat("run wooden furniture", DemoEvaluator())

        self.assertFalse(out["ok"])
        spoken = " ".join(s.get("text", "") for s in said if s.get("kind") == "text")
        self.assertIn("credit balance is too low", spoken)
        self.assertIn("switch to another saved key", spoken, "no next action offered")
        self.assertNotIn("RuntimeError", spoken, "the class name is noise")

    def test_known_failures_carry_an_action(self) -> None:
        from app.web import _explain_failure

        cases = [
            ("TransportUnavailable: anthropic returned 400: Your credit balance is too low",
             "Add credit"),
            ("ChatError: openrouter is rate-limiting this request", "Wait and retry"),
            ("google returned 400: You exceeded your current quota", "free-tier quota"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw[:30]):
                message, fix = _explain_failure(RuntimeError(raw))
                self.assertIn(expected, fix)
                self.assertFalse(
                    message.startswith(("TransportUnavailable", "ChatError")),
                    "the exception class should be stripped",
                )


class SelfDiagnosis(unittest.TestCase):
    """The app checks itself. It runs on the operator's machine and holds the
    key, so it is the only thing that can - making someone run scripts to find
    out why a run produced fictional data was the wrong shape.
    """

    def test_every_check_names_a_state_and_a_next_action_when_it_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            ws.config = Config()          # nothing configured at all
            report = ws.diagnose()
        self.assertTrue(report["checks"])
        for check in report["checks"]:
            with self.subTest(check=check["name"]):
                self.assertIn(check["state"], {"ok", "warn", "bad"})
                self.assertTrue(check["detail"], "a check with no observation is noise")
                if check["state"] == "bad":
                    self.assertTrue(check["fix"], "a blocking check must say what to do")

    def test_it_stops_at_the_first_blocker_rather_than_guessing_past_it(self) -> None:
        """Later checks depend on earlier ones; running them without a provider
        would report a cascade of failures with one real cause."""
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            ws.config = Config()
            names = [c["name"] for c in ws.diagnose()["checks"]]
        self.assertIn("Provider", names)
        self.assertNotIn("Connection", names)

    def test_the_summary_points_at_one_thing(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            ws.config = Config()
            self.assertIn("Provider", ws.diagnose()["summary"])


class EditableFacts(unittest.TestCase):
    """The facts panel was read-only, which quietly made the shipped defaults
    permanent - and gates read these fields by name."""

    def test_facts_can_be_replaced(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            out = ws.save_facts({"facts": {"budget_ceiling_usd": 5000, "location": "Sofia"}})
            self.assertNotIn("error", out)
            self.assertEqual(5000, ws.facts().get("budget_ceiling_usd"))
            self.assertEqual("Sofia", ws.facts().get("location"))

    def test_a_json_string_is_accepted_since_the_editor_sends_text(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            ws.save_facts({"facts": '{"budget_ceiling_usd": 100}'})
            self.assertEqual(100, ws.facts().get("budget_ceiling_usd"))

    def test_malformed_json_is_refused_without_touching_the_store(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            before = dict(ws.facts().fields)
            out = ws.save_facts({"facts": "{not json"})
            self.assertIn("error", out)
            self.assertEqual(before, dict(ws.facts().fields))

    def test_removing_a_field_a_rule_reads_is_reported_not_silently_allowed(self) -> None:
        """Such a rule fails closed. That is the safe direction, but the operator
        has to be told, or a gate quietly starts rejecting everything."""
        with TemporaryDirectory() as tmp:
            ws = Workspace(pathlib.Path(tmp))
            out = ws.save_facts({"facts": {"hours_per_week": 12}})
            self.assertIn("budget_ceiling_usd", out["orphaned"])
            self.assertIn("fail closed", out["note"])
