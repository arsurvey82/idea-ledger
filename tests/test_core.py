"""Core tests. These must pass with no API key, no network, no third-party deps.

Each test pins a property the prior system failed to hold. The docstrings name
the failure, so a regression reads as "we reintroduced X" rather than
"assertion false".
"""

from __future__ import annotations

import unittest

from app.core.rules import (
    ChangeClass,
    GateResult,
    MalformedPredicate,
    Rule,
    RuleTarget,
    UnresolvedInput,
    apply_gates,
    evaluate,
    referenced_fields,
    unresolved_inputs,
)
from app.core.scoring import (
    NotComparable,
    Threshold,
    calibration_pairs,
    rank_within_track,
    score,
)
from app.core.types import (
    DimensionScore,
    Evidence,
    FactBase,
    Idea,
    IllegalTransition,
    Provenance,
    Rubric,
    ScoreVector,
    Status,
)

RUBRIC = Rubric(
    version=1,
    dimensions=("demand", "competition", "ease", "capital", "profit", "solo_marketing"),
    tracks=("service", "physical"),
)
THRESHOLDS = (Threshold("ease", 7), Threshold("solo_marketing", 7))


def vector(track: str = "service", version: int = 1, **overrides: int) -> ScoreVector:
    base = dict(demand=8, competition=5, ease=7, capital=9, profit=7, solo_marketing=8)
    base.update(overrides)
    return ScoreVector(
        rubric_version=version,
        track=track,
        scores=tuple(
            DimensionScore(dimension=k, value=v, evidence_ids=("e1",), confidence=0.6)
            for k, v in base.items()
        ),
    )


class Predicates(unittest.TestCase):
    def test_leaf_and_combinators(self) -> None:
        scope = {"capital_required_usd": 12_000, "track": "service", "licences": ["2-15"]}
        self.assertTrue(evaluate({"field": "capital_required_usd", "op": "lte", "value": 20_000}, scope))
        self.assertFalse(evaluate({"field": "capital_required_usd", "op": "gt", "value": 20_000}, scope))
        self.assertTrue(evaluate({"field": "licences", "op": "contains", "value": "2-15"}, scope))
        self.assertTrue(
            evaluate(
                {"all": [
                    {"field": "track", "op": "eq", "value": "service"},
                    {"not": {"field": "capital_required_usd", "op": "gt", "value": 20_000}},
                ]},
                scope,
            )
        )

    def test_unresolved_field_raises_rather_than_defaulting(self) -> None:
        """A rule reading a field nobody defined must not quietly pass."""
        with self.assertRaises(UnresolvedInput):
            evaluate({"field": "licences_held", "op": "exists"}, {"track": "service"})

    def test_malformed_predicate_is_rejected_at_construction(self) -> None:
        with self.assertRaises(MalformedPredicate):
            Rule(id="r", description="", target=RuleTarget.SYMBOLIC, stage="gate",
                 predicate={"nonsense": 1})

    def test_referenced_fields_walks_the_whole_tree(self) -> None:
        pred = {"any": [
            {"field": "a", "op": "exists"},
            {"all": [{"field": "b", "op": "eq", "value": 1}, {"not": {"field": "c", "op": "truthy"}}]},
        ]}
        self.assertEqual(referenced_fields(pred), frozenset({"a", "b", "c"}))

    def test_no_python_is_executed(self) -> None:
        """Operator rules are data. A payload that would be dangerous under
        eval must be inert here."""
        scope = {"x": 1}
        with self.assertRaises(MalformedPredicate):
            evaluate({"field": "x", "op": "__import__('os').system"}, scope)


class Gates(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = FactBase({"budget_ceiling_usd": 20_000, "licences_held": []})
        self.idea = Idea(id="i1", name="Test", track="service",
                         fields={"capital_required_usd": 12_000})

    def test_passing_gate(self) -> None:
        rules = [Rule(id="budget", description="within budget", target=RuleTarget.SYMBOLIC,
                      stage="gate",
                      predicate={"field": "capital_required_usd", "op": "lte", "value": 20_000})]
        self.assertTrue(apply_gates(self.idea, self.facts, rules))

    def test_failing_gate_names_the_predicate(self) -> None:
        rules = [Rule(id="budget", description="within budget", target=RuleTarget.SYMBOLIC,
                      stage="gate",
                      predicate={"field": "capital_required_usd", "op": "lte", "value": 5_000})]
        result = apply_gates(self.idea, self.facts, rules)
        self.assertFalse(result.passed)
        self.assertEqual(result.failures[0].rule_id, "budget")

    def test_unresolved_input_fails_closed(self) -> None:
        """The prior system loaded its constraints file but never applied it.
        A rule that cannot be evaluated must fail, never pass."""
        rules = [Rule(id="lic", description="needs a licence he holds",
                      target=RuleTarget.SYMBOLIC, stage="gate",
                      predicate={"field": "requires_licence", "op": "falsy"})]
        result = apply_gates(self.idea, self.facts, rules)
        self.assertFalse(result.passed)
        self.assertEqual(result.failures[0].missing_input, "requires_licence")

    def test_neural_rules_are_skipped_by_the_gate(self) -> None:
        rules = [Rule(id="moat", description="prefer licence moats", target=RuleTarget.NEURAL,
                      stage="generate", fragment="Prefer moats that are a licence.")]
        self.assertTrue(apply_gates(self.idea, self.facts, rules))

    def test_upstream_resolution_reports_the_missing_field(self) -> None:
        rules = [Rule(id="lic", description="", target=RuleTarget.SYMBOLIC, stage="gate",
                      predicate={"field": "licences_required", "op": "exists"})]
        gaps = unresolved_inputs(rules, known=self.facts.field_names | {"capital_required_usd"})
        self.assertEqual(gaps, {"lic": frozenset({"licences_required"})})


class Transitions(unittest.TestCase):
    def test_rejected_cannot_go_straight_back_to_active(self) -> None:
        """Reviving a rejected idea is an explicit act, not a run side effect."""
        idea = Idea(id="i", name="n", track="service", status=Status.REJECTED)
        with self.assertRaises(IllegalTransition):
            idea.with_status(Status.ACTIVE)
        self.assertEqual(idea.with_status(Status.ON_HOLD).status, Status.ON_HOLD)


class Overrides(unittest.TestCase):
    def test_override_retains_the_original_and_demands_a_reason(self) -> None:
        s = DimensionScore(dimension="ease", value=6, confidence=0.5)
        edited = s.override(8, reason="operator disagrees; has run this playbook before")
        self.assertEqual(edited.value, 8)
        self.assertEqual(edited.original_value, 6)
        self.assertIs(edited.provenance, Provenance.OVERRIDDEN)
        with self.assertRaises(ValueError):
            s.override(8, reason="   ")

    def test_calibration_excludes_overridden_dimensions(self) -> None:
        """Counting a human correction as model error would drift the prior
        toward measuring the operator instead of the model."""
        before = vector()
        after_scores = list(vector(competition=3).scores)
        after_scores[2] = after_scores[2].override(9, reason="manual")
        after = ScoreVector(1, "service", tuple(after_scores))

        pairs = dict((d, (a, b)) for d, a, b in calibration_pairs(before, after))
        self.assertIn("competition", pairs)
        self.assertEqual(pairs["competition"], (5, 3))
        self.assertNotIn("ease", pairs)


class Scoring(unittest.TestCase):
    def test_total_and_threshold_reporting(self) -> None:
        idea = Idea(id="i", name="n", track="service", vector=vector(ease=6))
        s = score(idea, RUBRIC, THRESHOLDS)
        self.assertEqual(s.total, 43)
        self.assertEqual(s.max_total, 60)
        self.assertFalse(s.thresholds_met)
        self.assertEqual(s.failed_thresholds, ("ease",))

    def test_cross_rubric_version_ranking_is_refused(self) -> None:
        """Adding a dimension is a breaking change: old and new scores are not
        on one scale, and the system must refuse rather than sort them."""
        a = Idea(id="a", name="A", track="service", vector=vector())
        b = Idea(id="b", name="B", track="service", vector=vector(version=2))
        scored = [
            score(a, RUBRIC, THRESHOLDS),
            score(b, Rubric(2, RUBRIC.dimensions, RUBRIC.tracks), THRESHOLDS),
        ]
        with self.assertRaises(NotComparable):
            rank_within_track(scored, "service")

    def test_ranking_is_per_track(self) -> None:
        ideas = [
            Idea(id="a", name="A", track="service", vector=vector(demand=9)),
            Idea(id="b", name="B", track="service", vector=vector(demand=6)),
            Idea(id="c", name="C", track="physical", vector=vector(track="physical", demand=10)),
        ]
        scored = [score(i, RUBRIC, THRESHOLDS) for i in ideas]
        ranked = rank_within_track(scored, "service")
        self.assertEqual([s.idea_id for s in ranked], ["a", "b"])

    def test_vector_must_match_the_rubric(self) -> None:
        partial = ScoreVector(1, "service", (DimensionScore("demand", 8),))
        idea = Idea(id="i", name="n", track="service", vector=partial)
        with self.assertRaises(NotComparable):
            score(idea, RUBRIC, THRESHOLDS)


class EvidenceRules(unittest.TestCase):
    def test_evidence_requires_a_resolvable_url(self) -> None:
        with self.assertRaises(ValueError):
            Evidence(id="e", url="probably a competitor", claim="x", retrieved_at="2026-08-04")


if __name__ == "__main__":
    unittest.main()
