"""The pipeline orchestrator.

Control flow lives here, in ordinary code. The model is consulted at four
points and decides nothing about what happens next — it cannot skip a gate,
cannot decide it has searched enough, and never sees a score.

Two of the three termination paths are reached without asking a model whether
an idea is any good:

    gate rejection      a constraint predicate failed
    coverage exhausted  not enough cited competitors were found
    refuted             the adversarial pass stood up (this one does use a model)

The orchestrator talks to an ``Evaluator`` port rather than a provider, so the
whole flow is exercisable with no key, no network, and no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from .core.assumptions import AssumptionGraph, Dependent
from .core.calibration import CalibrationStore
from .core.manifest import Manifest
from .core.rules import GateResult, Rule, RuleTarget, apply_gates
from .core.scoring import Scored, Threshold, score
from .core.types import (
    DimensionScore,
    Evidence,
    FactBase,
    Idea,
    Rubric,
    ScoreVector,
    Status,
)


class Outcome(str, Enum):
    SCORED = "scored"
    GATE_REJECTED = "gate_rejected"
    UNDER_RESEARCHED = "under_researched"
    REFUTED = "refuted"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class CoveragePolicy:
    """M3, the coverage cage. Search until satisfied or out of attempts."""

    min_competitors: int = 3
    max_attempts: int = 4

    def satisfied(self, found: int) -> bool:
        return found >= self.min_competitors


@dataclass(frozen=True, slots=True)
class Candidate:
    """What the generate stage returns. Not yet an idea, not yet scored."""

    id: str
    name: str
    track: str
    fields: Mapping[str, object] = field(default_factory=dict)
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceBatch:
    evidence: tuple[Evidence, ...]
    competitors: tuple[str, ...]
    exhausted: bool = False


@dataclass(frozen=True, slots=True)
class Verdict:
    refuted: bool
    basis: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageEvent:
    stage: str
    state: str        # "started" | "passed" | "rejected" | "exhausted" | "skipped"
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.stage:<10} {self.state:<10} {self.detail}"


@dataclass(frozen=True, slots=True)
class RunResult:
    candidate: Candidate
    outcome: Outcome
    events: tuple[StageEvent, ...]
    idea: Idea | None = None
    gate: GateResult | None = None
    scored: Scored | None = None
    verdict: Verdict | None = None
    competitors_found: int = 0

    @property
    def explanation(self) -> str:
        """Why this idea ended where it did, in one line, for the reject log."""
        # `self.gate is not None`, never `self.gate`: GateResult is falsy exactly
        # when it failed, which is the case whose reason we need to report.
        if self.outcome is Outcome.GATE_REJECTED and self.gate is not None:
            return "; ".join(str(f) for f in self.gate.failures)
        if self.outcome is Outcome.UNDER_RESEARCHED:
            return (
                f"only {self.competitors_found} cited competitor(s) found; "
                "not scored rather than scored on thin evidence"
            )
        if self.outcome is Outcome.REFUTED and self.verdict:
            return self.verdict.basis
        if self.outcome is Outcome.DUPLICATE:
            return "already on the reject list; reviving is an explicit act"
        return f"scored {self.scored.display}" if self.scored else ""


class Evaluator(Protocol):
    """The model-facing port. Four methods, matching the four call sites."""

    def propose(self, brief: str, count: int, prior: str) -> Sequence[Candidate]:
        ...

    def gather(self, candidate: Candidate, attempt: int) -> EvidenceBatch:
        ...

    def judge(
        self, candidate: Candidate, evidence: Sequence[Evidence]
    ) -> Sequence[DimensionScore]:
        ...

    def refute(self, candidate: Candidate, evidence: Sequence[Evidence]) -> Verdict:
        ...

    def render(self, idea: Idea, scored: Scored) -> str:
        ...


@dataclass(slots=True)
class Pipeline:
    manifest: Manifest
    facts: FactBase
    rubric: Rubric
    rules: Sequence[Rule]
    thresholds: Sequence[Threshold]
    evaluator: Evaluator
    coverage: CoveragePolicy = field(default_factory=CoveragePolicy)
    rejected_ids: frozenset[str] = frozenset()
    calibration: CalibrationStore | None = None
    assumptions: AssumptionGraph | None = None
    on_event: Callable[[str, StageEvent], None] | None = None

    # -- entry points ----------------------------------------------------
    def run(self, brief: str, count: int = 3) -> list[RunResult]:
        prior = self.calibration.prior_text() if self.calibration else ""
        candidates = self.evaluator.propose(brief, count, prior)
        return [self.run_one(c) for c in candidates]

    def run_one(self, candidate: Candidate) -> RunResult:
        events: list[StageEvent] = []

        def emit(stage: str, state: str, detail: str = "") -> None:
            event = StageEvent(stage, state, detail)
            events.append(event)
            if self.on_event:
                self.on_event(candidate.id, event)

        # ST0 hydrate - symbolic. Dedup happens before anything is spent.
        emit("hydrate", "started")
        if candidate.id in self.rejected_ids:
            emit("hydrate", "rejected", "already rejected")
            return RunResult(candidate, Outcome.DUPLICATE, tuple(events))
        emit("hydrate", "passed")

        idea = Idea(
            id=candidate.id,
            name=candidate.name,
            track=candidate.track,
            fields=dict(candidate.fields),
        )

        # ST2 gate - symbolic. No model has been consulted about this idea yet.
        emit("gate", "started")
        gate = apply_gates(idea, self.facts, self.rules)
        if not gate.passed:
            emit("gate", "rejected", "; ".join(str(f) for f in gate.failures))
            return RunResult(
                candidate,
                Outcome.GATE_REJECTED,
                tuple(events),
                idea=idea.with_status(Status.REJECTED),
                gate=gate,
            )
        emit("gate", "passed")

        # ST3 evidence - neural, under the coverage cage.
        emit("evidence", "started")
        found: list[Evidence] = []
        competitors: set[str] = set()
        for attempt in range(1, self.coverage.max_attempts + 1):
            batch = self.evaluator.gather(candidate, attempt)
            found.extend(batch.evidence)
            competitors.update(batch.competitors)
            emit(
                "evidence",
                "started",
                f"attempt {attempt}: {len(competitors)} cited competitor(s)",
            )
            if self.coverage.satisfied(len(competitors)) or batch.exhausted:
                break

        if not self.coverage.satisfied(len(competitors)):
            emit(
                "evidence",
                "exhausted",
                f"{len(competitors)} of {self.coverage.min_competitors} required",
            )
            return RunResult(
                candidate,
                Outcome.UNDER_RESEARCHED,
                tuple(events),
                idea=idea.with_status(Status.UNDER_RESEARCHED),
                competitors_found=len(competitors),
            )
        emit("evidence", "passed", f"{len(competitors)} cited competitor(s)")

        idea = replace(idea, evidence=tuple(found))
        dimensions = tuple(self.evaluator.judge(candidate, found))

        # ST4 refute - neural, fresh context, defaults to refuted under doubt.
        emit("refute", "started")
        verdict = self.evaluator.refute(candidate, found)
        if verdict.refuted:
            emit("refute", "rejected", verdict.basis)
            return RunResult(
                candidate,
                Outcome.REFUTED,
                tuple(events),
                idea=idea.with_status(Status.REJECTED),
                verdict=verdict,
                competitors_found=len(competitors),
            )
        emit("refute", "passed", verdict.basis)

        # ST5 score - symbolic. The model has never seen a number.
        emit("score", "started")
        vector = ScoreVector(
            rubric_version=self.rubric.version, track=candidate.track, scores=dimensions
        )
        idea = replace(idea, vector=vector)
        scored = score(idea, self.rubric, self.thresholds)
        idea = idea.with_status(
            Status.ACTIVE if scored.thresholds_met else Status.BLOCKED
        )
        emit(
            "score",
            "passed",
            f"{scored.display}"
            + ("" if scored.thresholds_met else f" (below {', '.join(scored.failed_thresholds)})"),
        )

        self._record_assumptions(idea, found)

        return RunResult(
            candidate,
            Outcome.SCORED,
            tuple(events),
            idea=idea,
            gate=gate,
            scored=scored,
            verdict=verdict,
            competitors_found=len(competitors),
        )

    # -- helpers ---------------------------------------------------------
    def _record_assumptions(self, idea: Idea, evidence: Sequence[Evidence]) -> None:
        """Bind the idea to whatever it rests on, at the moment it is used.

        Recording the dependency later, from memory, is exactly how a dependent
        gets missed.
        """
        if self.assumptions is None:
            return
        for item in evidence:
            if item.id in self.assumptions.assumptions:
                self.assumptions.depends(
                    Dependent(idea.id, "idea", f"scored using {item.claim[:48]}"),
                    on=item.id,
                )

    def neural_stages(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.manifest.model_stages)

    def symbolic_stages(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.manifest.symbolic_stages)


def summarise(results: Iterable[RunResult]) -> str:
    """The run report. Every rejection names its cause; nothing is silent."""
    rows = list(results)
    if not rows:
        return "No candidates."
    by_outcome: dict[Outcome, list[RunResult]] = {}
    for r in rows:
        by_outcome.setdefault(r.outcome, []).append(r)

    lines = [f"{len(rows)} candidate(s):"]
    for outcome in Outcome:
        group = by_outcome.get(outcome)
        if not group:
            continue
        lines.append(f"  {outcome.value} ({len(group)})")
        for r in group:
            lines.append(f"    {r.candidate.name:<26} {r.explanation}")
    return "\n".join(lines)
