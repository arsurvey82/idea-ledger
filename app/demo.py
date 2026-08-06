"""A deterministic evaluator that needs no key.

Two reasons this ships rather than living in the tests: the interface has to be
explorable before anyone has entered a credential, and a demo run is the
honest way to show what a rejection looks like without spending tokens to
manufacture one.

Everything it returns is clearly fictional and labelled as such in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .core.types import DimensionScore, Evidence, Idea
from .pipeline import Candidate, EvidenceBatch, Verdict

_CANDIDATES = [
    (
        "demo-benchmark",
        "BenchmarkBridge (demo)",
        "annual compliance filing for building energy benchmarking",
        {"capital_required_usd": 9_000, "requires_live_negotiation": False,
         "requires_in_person": False, "requires_licence": False,
         "recurring_revenue": True, "moat_kind": "logistics"},
    ),
    (
        "demo-overpriced",
        "HeavyImport (demo)",
        "container-scale import requiring certification",
        {"capital_required_usd": 140_000, "requires_live_negotiation": False,
         "requires_in_person": True, "requires_licence": True,
         "recurring_revenue": False, "moat_kind": "physical_supply"},
    ),
    (
        "demo-negotiation",
        "ClaimFight (demo)",
        "contingency-fee bill negotiation with insurers",
        {"capital_required_usd": 6_000, "requires_live_negotiation": True,
         "requires_in_person": False, "requires_licence": False,
         "recurring_revenue": False, "moat_kind": "none"},
    ),
]

_SCORES = {
    "demo-benchmark": dict(
        demand=8, competition=6, ease=8, capital=9, profit=7, solo_marketing=8
    ),
}


@dataclass(slots=True)
class DemoEvaluator:
    """Returns fixed, obviously-fictional results. No network, no key.

    Candidate ids carry a per-run suffix. Without it the second demo run trips
    the reject-list guard and every candidate returns "already rejected", which
    is correct behaviour that reads exactly like a broken demo.
    """

    thin_evidence_for: str = "demo-thin"
    run: str = ""

    def __post_init__(self) -> None:
        if not self.run:
            self.run = f"{_counter():02d}"

    def propose(self, brief: str, count: int, prior: str) -> Sequence[Candidate]:
        return [
            Candidate(id=f"{cid}-{self.run}", name=name, track="service",
                      rationale=why, fields=fields)
            for cid, name, why, fields in _CANDIDATES[:count]
        ]

    def gather(self, candidate: Candidate, attempt: int) -> EvidenceBatch:
        if candidate.id == self.thin_evidence_for:
            return EvidenceBatch(evidence=(), competitors=(), exhausted=True)
        base = (attempt - 1) * 3
        evidence = tuple(
            Evidence(
                id=f"{candidate.id}-e{base + n}",
                url=f"https://example.invalid/{candidate.id}/{base + n}",
                claim=f"Demo competitor {base + n} serving the same customer",
                retrieved_at="2026-08-05T00:00:00+00:00",
                verbatim="(demo evidence - not a real source)",
            )
            for n in range(3)
        )
        return EvidenceBatch(
            evidence=evidence,
            competitors=tuple(f"DemoRival{base + n}" for n in range(3)),
        )

    def judge(
        self, candidate: Candidate, evidence: Sequence[Evidence]
    ) -> Sequence[DimensionScore]:
        values = _SCORES.get(
            candidate.id.rsplit("-", 1)[0],
            dict(demand=7, competition=4, ease=6, capital=7, profit=6, solo_marketing=7),
        )
        return tuple(
            DimensionScore(
                dimension=name,
                value=value,
                evidence_ids=tuple(e.id for e in evidence[:2]),
                confidence=0.55,
                falsifier="a named incumbent at a lower price would move this",
            )
            for name, value in values.items()
        )

    def refute(self, candidate: Candidate, evidence: Sequence[Evidence]) -> Verdict:
        return Verdict(refuted=False, basis="demo mode does not refute")

    def render(self, idea: Idea, scored: object) -> str:
        return f"{idea.name} - demo narrative, no model was called."


_RUNS = [0]


def _counter() -> int:
    _RUNS[0] += 1
    return _RUNS[0]


def demo_evaluator() -> DemoEvaluator:
    return DemoEvaluator()
