"""Pipeline and manifest tests.

The fake evaluator counts its own calls, which lets these tests assert the
thesis directly: gates and scoring consult no model, and a candidate that fails
a constraint costs nothing.
"""

from __future__ import annotations

import unittest
from collections import Counter

from app.core.assumptions import Assumption, AssumptionGraph
from app.core.manifest import Manifest, UnplaceableRule, validate
from app.core.rules import Rule, RuleTarget
from app.core.scoring import Threshold
from app.core.types import DimensionScore, Evidence, FactBase, Idea, Rubric, Status
from app.pipeline import (
    Candidate,
    CoveragePolicy,
    EvidenceBatch,
    Outcome,
    Pipeline,
    StageEvent,
    Verdict,
    summarise,
)

RUBRIC = Rubric(
    version=1,
    dimensions=("demand", "competition", "ease", "capital", "profit", "solo_marketing"),
    tracks=("service", "physical"),
)
THRESHOLDS = (Threshold("ease", 7), Threshold("solo_marketing", 7))

FACTS = FactBase(
    {
        "budget_ceiling_usd": 20_000,
        "accepts_live_negotiation": False,
        "licences_held": [],
    }
)

BUDGET_RULE = Rule(
    id="budget",
    description="capital must fit the budget ceiling",
    target=RuleTarget.SYMBOLIC,
    stage="gate",
    predicate={"field": "capital_required_usd", "op": "lte", "value": 20_000},
)
NEGOTIATION_RULE = Rule(
    id="no_live_negotiation",
    description="delivery must not require live negotiation",
    target=RuleTarget.SYMBOLIC,
    stage="gate",
    predicate={"field": "requires_live_negotiation", "op": "falsy"},
)


def candidate(cid: str = "c1", **fields: object) -> Candidate:
    base = {"capital_required_usd": 12_000, "requires_live_negotiation": False}
    base.update(fields)
    return Candidate(id=cid, name=cid.upper(), track="service", fields=base)


def ev(n: int) -> Evidence:
    return Evidence(
        id=f"e{n}",
        url=f"https://example.com/{n}",
        claim=f"competitor {n} exists",
        retrieved_at="2026-08-04",
    )


class FakeEvaluator:
    """Counts every call, so tests can assert what was never asked."""

    def __init__(
        self,
        *,
        competitors_per_attempt: int = 3,
        refute: bool = False,
        ease: int = 8,
        exhaust_after: int | None = None,
    ) -> None:
        self.calls: Counter[str] = Counter()
        self._per_attempt = competitors_per_attempt
        self._refute = refute
        self._ease = ease
        self._exhaust_after = exhaust_after

    def propose(self, brief, count, prior):
        self.calls["propose"] += 1
        self.prior_seen = prior
        return [candidate(f"c{i}") for i in range(count)]

    def gather(self, cand, attempt):
        self.calls["gather"] += 1
        start = (attempt - 1) * self._per_attempt
        items = tuple(ev(start + i) for i in range(self._per_attempt))
        return EvidenceBatch(
            evidence=items,
            competitors=tuple(f"rival-{start + i}" for i in range(self._per_attempt)),
            exhausted=self._exhaust_after is not None and attempt >= self._exhaust_after,
        )

    def judge(self, cand, evidence):
        self.calls["judge"] += 1
        values = dict(
            demand=8, competition=5, ease=self._ease, capital=9, profit=7, solo_marketing=8
        )
        return tuple(
            DimensionScore(
                dimension=k,
                value=v,
                evidence_ids=tuple(e.id for e in evidence[:2]),
                confidence=0.6,
                falsifier="a cheaper named incumbent would move this",
            )
            for k, v in values.items()
        )

    def refute(self, cand, evidence):
        self.calls["refute"] += 1
        return Verdict(
            refuted=self._refute,
            basis="a near-exact incumbent was found" if self._refute else "no refutation stood up",
        )

    def render(self, idea, scored):
        self.calls["render"] += 1
        return "narrative"


def build(evaluator, **kwargs) -> Pipeline:
    return Pipeline(
        manifest=Manifest.load(),
        facts=FACTS,
        rubric=RUBRIC,
        rules=[BUDGET_RULE, NEGOTIATION_RULE],
        thresholds=THRESHOLDS,
        evaluator=evaluator,
        **kwargs,
    )


class GateShortCircuit(unittest.TestCase):
    def test_a_gate_failure_costs_no_model_call(self) -> None:
        """Two of three termination paths never ask a model anything."""
        fake = FakeEvaluator()
        result = build(fake).run_one(candidate(capital_required_usd=95_000))

        self.assertIs(result.outcome, Outcome.GATE_REJECTED)
        self.assertEqual(fake.calls["gather"], 0)
        self.assertEqual(fake.calls["judge"], 0)
        self.assertEqual(fake.calls["refute"], 0)

    def test_rejection_names_the_failing_rule(self) -> None:
        result = build(FakeEvaluator()).run_one(candidate(capital_required_usd=95_000))
        self.assertIn("budget", result.explanation)

    def test_an_unresolved_field_rejects_rather_than_passes(self) -> None:
        pipe = build(FakeEvaluator())
        result = pipe.run_one(Candidate(id="c9", name="C9", track="service", fields={}))
        self.assertIs(result.outcome, Outcome.GATE_REJECTED)
        self.assertIn("unresolved input", result.explanation)

    def test_already_rejected_ideas_never_reach_the_gate(self) -> None:
        fake = FakeEvaluator()
        pipe = build(fake, rejected_ids=frozenset({"c1"}))
        result = pipe.run_one(candidate("c1"))
        self.assertIs(result.outcome, Outcome.DUPLICATE)
        self.assertEqual(sum(fake.calls.values()), 0)


class CoverageCage(unittest.TestCase):
    def test_thin_evidence_yields_no_score_at_all(self) -> None:
        """Under-researched is a terminal state, not a low score."""
        fake = FakeEvaluator(competitors_per_attempt=1, exhaust_after=1)
        result = build(fake, coverage=CoveragePolicy(min_competitors=3)).run_one(candidate())

        self.assertIs(result.outcome, Outcome.UNDER_RESEARCHED)
        self.assertIsNone(result.scored)
        self.assertEqual(result.idea.status, Status.UNDER_RESEARCHED)
        self.assertEqual(fake.calls["judge"], 0)
        self.assertIn("not scored rather than scored on thin evidence", result.explanation)

    def test_it_searches_again_until_the_cage_is_satisfied(self) -> None:
        fake = FakeEvaluator(competitors_per_attempt=1)
        result = build(fake, coverage=CoveragePolicy(min_competitors=3, max_attempts=5)).run_one(
            candidate()
        )
        self.assertIs(result.outcome, Outcome.SCORED)
        self.assertEqual(fake.calls["gather"], 3)

    def test_attempts_are_bounded(self) -> None:
        fake = FakeEvaluator(competitors_per_attempt=0)
        result = build(fake, coverage=CoveragePolicy(min_competitors=3, max_attempts=2)).run_one(
            candidate()
        )
        self.assertEqual(fake.calls["gather"], 2)
        self.assertIs(result.outcome, Outcome.UNDER_RESEARCHED)


class Adversarial(unittest.TestCase):
    def test_a_standing_refutation_ends_the_run(self) -> None:
        fake = FakeEvaluator(refute=True)
        result = build(fake).run_one(candidate())
        self.assertIs(result.outcome, Outcome.REFUTED)
        self.assertIsNone(result.scored)
        self.assertEqual(result.idea.status, Status.REJECTED)
        self.assertIn("near-exact incumbent", result.explanation)


class Scoring(unittest.TestCase):
    def test_a_survivor_is_scored_by_code_not_by_the_model(self) -> None:
        fake = FakeEvaluator()
        result = build(fake).run_one(candidate())
        self.assertIs(result.outcome, Outcome.SCORED)
        self.assertEqual(result.scored.total, 45)
        self.assertEqual(result.scored.display, "45/60")
        self.assertEqual(result.idea.status, Status.ACTIVE)

    def test_failing_a_threshold_blocks_rather_than_rejects(self) -> None:
        result = build(FakeEvaluator(ease=5)).run_one(candidate())
        self.assertIs(result.outcome, Outcome.SCORED)
        self.assertFalse(result.scored.thresholds_met)
        self.assertEqual(result.idea.status, Status.BLOCKED)

    def test_events_are_emitted_in_order_for_the_interface(self) -> None:
        seen: list[StageEvent] = []
        pipe = build(FakeEvaluator(), on_event=lambda _cid, e: seen.append(e))
        pipe.run_one(candidate())
        stages = [e.stage for e in seen]
        self.assertEqual(stages[0], "hydrate")
        self.assertEqual(stages[-1], "score")
        self.assertIn("evidence", stages)

    def test_the_calibration_prior_reaches_the_generate_stage(self) -> None:
        from app.core.calibration import MIN_SAMPLES, seed_from_history

        store = seed_from_history(
            [{"dimension": "competition", "first_pass": 8, "verified": 3}] * MIN_SAMPLES
        )
        fake = FakeEvaluator()
        build(fake, calibration=store).run("a brief", count=1)
        self.assertIn("optimistic", fake.prior_seen)

    def test_assumptions_are_bound_when_evidence_is_used(self) -> None:
        graph = AssumptionGraph()
        graph.add(Assumption(id="e0", statement="competitor 0 exists"))
        build(FakeEvaluator(), assumptions=graph).run_one(candidate("c1"))
        self.assertEqual(graph.direct_dependents("e0")[0].id, "c1")


class RunSummary(unittest.TestCase):
    def test_every_rejection_states_its_cause(self) -> None:
        pipe = build(FakeEvaluator())
        results = [
            pipe.run_one(candidate("good")),
            pipe.run_one(candidate("pricey", capital_required_usd=99_000)),
        ]
        text = summarise(results)
        self.assertIn("scored (1)", text)
        self.assertIn("gate_rejected (1)", text)
        self.assertIn("budget", text)


class ManifestTopology(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Manifest.load()

    def test_it_is_healthy_as_shipped(self) -> None:
        self.assertEqual(validate(self.m, dimensions=RUBRIC.dimensions), [])

    def test_it_knows_which_stages_call_a_model(self) -> None:
        self.assertEqual(
            [s.id for s in self.m.symbolic_stages], ["hydrate", "gate", "score"]
        )
        self.assertIn("evidence", [s.id for s in self.m.model_stages])

    def test_upstream_and_downstream_are_derived_from_the_manifest(self) -> None:
        self.assertEqual(self.m.upstream_of("score")[0], "hydrate")
        self.assertIn("render", self.m.downstream_of("score"))

    def test_a_symbolic_rule_is_placed_at_a_stage_that_accepts_predicates(self) -> None:
        placement = self.m.place(
            rule_id="r1", target="symbolic", inputs={"capital_required_usd"}
        )
        self.assertEqual(placement.stage, "gate")

    def test_a_neural_rule_is_placed_at_a_prompt_stage(self) -> None:
        placement = self.m.place(rule_id="r2", target="neural")
        self.assertIn(placement.stage, ("generate", "evidence", "refute", "render"))

    def test_a_rule_reading_an_undefined_field_is_refused_at_intake(self) -> None:
        """Upstream resolution: a rule that could never fire must not activate."""
        with self.assertRaises(UnplaceableRule) as ctx:
            self.m.place(rule_id="r3", target="symbolic", inputs={"licences_required"})
        self.assertIn("licences_required", str(ctx.exception))
        self.assertIn("never fire", str(ctx.exception))

    def test_a_declared_new_field_unblocks_placement(self) -> None:
        placement = self.m.place(
            rule_id="r3",
            target="symbolic",
            inputs={"licences_required"},
            extra_fields={"licences_required"},
        )
        self.assertEqual(placement.stage, "gate")

    def test_a_stage_that_rejects_the_target_is_refused(self) -> None:
        with self.assertRaises(UnplaceableRule):
            self.m.place(rule_id="r4", target="symbolic", preferred_stage="generate")

    def test_placing_a_rule_reports_what_it_invalidates_downstream(self) -> None:
        self.assertIn("score", self.m.impact_of_placing("gate"))

    def test_a_symbolic_stage_cannot_require_a_model_capability(self) -> None:
        with self.assertRaises(ValueError):
            Manifest.from_dict(
                {
                    "version": 1,
                    "stages": [
                        {"id": "gate", "kind": "symbolic", "requires": ["server_search"]}
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
