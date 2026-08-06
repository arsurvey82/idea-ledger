"""Assumption graph, calibration, and configuration tests.

Still no key, no network, no third-party dependency.
"""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import Config, KeySource, SetupStep, validate_key
from app.core.assumptions import (
    Assumption,
    AssumptionGraph,
    CyclicDependency,
    Dependent,
    summarise,
)
from app.core.calibration import MIN_SAMPLES, CalibrationStore, seed_from_history
from app.core.types import DimensionScore, Provenance, ScoreVector


def graph_with_licence_premise() -> AssumptionGraph:
    """The real case, reconstructed: a score resting on an unchecked premise."""
    g = AssumptionGraph()
    g.add(Assumption(id="licence_fl", statement="Partner holds a Florida licence"))
    g.add(
        Assumption(
            id="licence_moat",
            statement="The licence is the idea's durable moat",
            source="unverified",
        )
    )
    g.depends(Dependent("licence_moat", "assumption", "moat rests on the licence"), on="licence_fl")
    g.depends(Dependent("casa-sello", "idea", "scored as licence-gated"), on="licence_moat")
    g.depends(
        Dependent("casa-sello-v1", "score", "competition dimension assumed the moat"),
        on="licence_moat",
    )
    g.depends(
        Dependent("bayline-advantages", "dossier", "ranked the credential first"),
        on="licence_fl",
    )
    return g


class Graph(unittest.TestCase):
    def test_invalidation_is_transitive(self) -> None:
        """One false premise must surface every artifact downstream of it,
        including those reached through another assumption."""
        g = graph_with_licence_premise()
        impact = g.impact("licence_fl")

        ids = {d.id for d in impact.direct} | {d.id for d in impact.transitive}
        self.assertEqual(
            ids, {"licence_moat", "bayline-advantages", "casa-sello", "casa-sello-v1"}
        )
        self.assertEqual(impact.total, 4)

    def test_invalidate_records_the_reason_and_returns_the_blast_radius(self) -> None:
        g = graph_with_licence_premise()
        impact = g.invalidate("licence_fl", reason="confirmed not registered in Florida")
        self.assertEqual(g.assumptions["licence_fl"].confidence, 0.0)
        self.assertIn("not registered", g.assumptions["licence_fl"].source)
        self.assertEqual(impact.total, 4)

    def test_report_names_artifacts_by_kind(self) -> None:
        report = graph_with_licence_premise().impact("licence_fl").report()
        self.assertIn("4 artifact(s) depend", report)
        self.assertIn("dossier", report)
        self.assertIn("bayline-advantages", report)

    def test_load_bearing_unverified_is_the_standing_query(self) -> None:
        """The question nobody could ask before: what is the shortlist
        quietly betting on?"""
        g = graph_with_licence_premise()
        risky = g.load_bearing_unverified()
        self.assertEqual(risky[0].assumption_id, "licence_fl")
        self.assertIn("unverified assumption(s) are load-bearing", summarise(g))

    def test_verified_assumptions_are_not_flagged(self) -> None:
        g = AssumptionGraph()
        g.add(Assumption(id="a", statement="checked", evidence_ids=("e1",)))
        g.depends(Dependent("i1", "idea", "uses it"), on="a")
        self.assertEqual(g.load_bearing_unverified(), ())
        self.assertIn("Every load-bearing assumption is verified", summarise(g))

    def test_unverified_but_unused_is_not_load_bearing(self) -> None:
        g = AssumptionGraph()
        g.add(Assumption(id="idle", statement="nothing rests on this"))
        self.assertEqual(g.load_bearing_unverified(), ())

    def test_cycles_are_refused(self) -> None:
        g = AssumptionGraph()
        g.add(Assumption(id="a", statement="a"))
        g.add(Assumption(id="b", statement="b"))
        g.depends(Dependent("b", "assumption", "b rests on a"), on="a")
        with self.assertRaises(CyclicDependency):
            g.depends(Dependent("a", "assumption", "a rests on b"), on="b")

    def test_unknown_assumption_is_refused(self) -> None:
        g = AssumptionGraph()
        with self.assertRaises(KeyError):
            g.depends(Dependent("i", "idea", "x"), on="nope")


def vec(**values: int) -> ScoreVector:
    base = dict(demand=8, competition=8, ease=8, capital=9, profit=7, solo_marketing=8)
    base.update(values)
    return ScoreVector(
        1, "service",
        tuple(DimensionScore(dimension=k, value=v) for k, v in base.items()),
    )


class Calibration(unittest.TestCase):
    def test_measures_signed_drift_per_dimension(self) -> None:
        store = CalibrationStore()
        store.record_rescore(idea_id="i1", before=vec(), after=vec(competition=4), at="2026-08-04")
        bias = store.bias("competition")
        assert bias is not None
        self.assertEqual(bias.mean_delta, -4.0)
        self.assertEqual(bias.direction, "optimistic")

    def test_below_the_sample_floor_it_refuses_to_state_a_number(self) -> None:
        """A confident prior from three observations is the overconfidence
        this loop exists to correct."""
        store = CalibrationStore()
        store.record_rescore(idea_id="i1", before=vec(), after=vec(competition=4), at="x")
        text = store.prior_text()
        self.assertIn("direction, not a number", text)
        self.assertNotIn("points on average", text)

    def test_above_the_floor_it_states_the_measured_bias(self) -> None:
        store = CalibrationStore()
        for n in range(MIN_SAMPLES):
            store.record_rescore(
                idea_id=f"i{n}", before=vec(), after=vec(competition=4), at="x"
            )
        text = store.prior_text()
        self.assertIn("points on average", text)
        self.assertIn("100% of revisions moved down", text)
        self.assertIn("url-resolvable competitors", text)

    def test_a_fresh_install_emits_no_prior_at_all(self) -> None:
        self.assertEqual(CalibrationStore().prior_text(), "")

    def test_overridden_dimensions_never_reach_the_store(self) -> None:
        before = vec()
        scores = list(vec(competition=4).scores)
        scores[2] = scores[2].override(3, reason="operator judgement")
        after = ScoreVector(1, "service", tuple(scores))

        store = CalibrationStore()
        store.record_rescore(idea_id="i1", before=before, after=after, at="x")
        self.assertIsNone(store.bias("ease"))
        self.assertIsNotNone(store.bias("competition"))

    def test_history_can_be_imported(self) -> None:
        store = seed_from_history(
            [{"dimension": "competition", "first_pass": 8, "verified": 3}] * MIN_SAMPLES
        )
        self.assertIn("points on average", store.prior_text())


class Configuration(unittest.TestCase):
    def test_setup_walks_the_operator_forward(self) -> None:
        env: dict[str, str] = {}
        cfg = Config()
        self.assertIs(cfg.next_step(env), SetupStep.CHOOSE_PROVIDER)

        cfg = cfg.with_provider("anthropic")
        self.assertIs(cfg.next_step(env), SetupStep.SUPPLY_KEY)

        env["ANTHROPIC_API_KEY"] = "sk-test-value-1234"
        self.assertIs(cfg.next_step(env), SetupStep.EDIT_FACT_BASE)

        cfg = dataclasses.replace(cfg, fact_base_edited=True)
        self.assertIs(cfg.next_step(env), SetupStep.READY)

    def test_openrouter_asks_for_the_key_before_the_route(self) -> None:
        """Order matters: a route cannot be looked up without a key.

        Listing a broker's usable routes is an authenticated call, so asking for
        the route first asks for something the operator has no way to answer.

        ``home`` is pinned to an empty directory deliberately. Left unset it
        resolves to the real user directory, and this assertion then depends on
        whether the machine running the tests happens to have a key stored.
        """
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            cfg = Config().with_provider("openrouter")
            self.assertIs(cfg.next_step({}, home), SetupStep.SUPPLY_KEY)

            env = {"OPENROUTER_API_KEY": "sk-or-test-value"}
            self.assertIs(cfg.next_step(env, home), SetupStep.NAME_ROUTE)

            routed = dataclasses.replace(cfg, model_id="a/b:free")
            self.assertIs(routed.next_step(env, home), SetupStep.EDIT_FACT_BASE)

    def test_the_key_is_never_returned_in_a_description(self) -> None:
        secret = "sk-super-secret-abcd"
        cfg = Config().with_provider("anthropic")
        text = cfg.describe({"ANTHROPIC_API_KEY": secret})
        self.assertNotIn(secret, text)
        self.assertIn("ending ...abcd", text)

    def test_a_missing_key_explains_how_to_supply_one(self) -> None:
        text = Config().with_provider("anthropic").describe({})
        self.assertIn("ANTHROPIC_API_KEY", text)
        self.assertIn("export", text)
        self.assertIn("$env:", text)

    def test_config_round_trips_without_ever_storing_the_key(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            cfg = Config().with_provider("anthropic")
            path = cfg.save(home)
            raw = path.read_text(encoding="utf-8")
            self.assertIn("ANTHROPIC_API_KEY", raw)   # the variable name
            self.assertNotIn("sk-", raw)              # never a value
            self.assertEqual(Config.load(home).provider, "anthropic")

    def test_user_dir_is_outside_the_repo_and_overridable(self) -> None:
        from app.config import user_dir

        self.assertEqual(user_dir({"IDEA_LEDGER_HOME": "/tmp/x"}), Path("/tmp/x"))
        self.assertNotIn("idea-ledger" + "/app", str(user_dir({})))

    def test_absent_key_reports_absent(self) -> None:
        source, _ = Config().with_provider("openai").key_status({})
        self.assertIs(source, KeySource.ABSENT)


if __name__ == "__main__":
    unittest.main()


class KeyValidation(unittest.TestCase):
    """A key field that accepts anything turns a paste slip into an account hunt.

    The case that motivated this: a browser console traceback was pasted into
    the key box and stored. Every route probe then answered "rejected the key",
    which reads as an OpenRouter account problem and is nowhere near the truth.
    """

    CONSOLE_ERROR = (
        "Uncaught (in promise) ReferenceError: api is not defined\n"
        "    refresh http://127.0.0.1:8420/:531\n"
        "    <anonymous> http://127.0.0.1:8420/:695\n"
        "    <anonymous> http://127.0.0.1:8420/:701\n"
        "127.0.0.1:8420:531:3"
    )

    def test_a_pasted_traceback_is_refused(self) -> None:
        key, complaint = validate_key("openrouter", self.CONSOLE_ERROR)
        self.assertEqual("", key)
        self.assertIn("spaces or line breaks", complaint)

    def test_a_key_for_the_wrong_provider_names_the_right_one(self) -> None:
        key, complaint = validate_key("openrouter", "sk-ant-" + "a" * 40)
        self.assertEqual("", key)
        self.assertIn("sk-or-", complaint)
        self.assertIn("anthropic", complaint)

    def test_a_truncated_copy_says_it_is_short(self) -> None:
        _, complaint = validate_key("openai", "sk-abc")
        self.assertIn("cut short", complaint)

    def test_an_overlong_blob_is_refused_even_with_no_whitespace(self) -> None:
        _, complaint = validate_key("openrouter", "sk-or-" + "x" * 400)
        self.assertIn("longer than any key", complaint)

    def test_an_empty_field_says_so_plainly(self) -> None:
        _, complaint = validate_key("openai", "   ")
        self.assertIn("empty", complaint)

    def test_real_looking_keys_pass_and_come_back_trimmed(self) -> None:
        for provider, raw in [
            ("anthropic", "  sk-ant-api03-" + "A" * 40 + "  "),
            ("openai", "sk-proj-" + "B" * 40),
            ("openrouter", "sk-or-v1-" + "c" * 40),
        ]:
            with self.subTest(provider=provider):
                key, complaint = validate_key(provider, raw)
                self.assertEqual("", complaint)
                self.assertEqual(raw.strip(), key)

    def test_an_unknown_provider_skips_the_prefix_check(self) -> None:
        """Structural checks still apply; a prefix we do not know cannot be asserted."""
        key, complaint = validate_key("someone-else", "xyz-" + "d" * 40)
        self.assertEqual("", complaint)
        self.assertTrue(key)
