"""The assumption graph and invalidation.

This module exists because of one concrete failure. In the system this
replaces, a business idea scored 47/60 and sat at the top of a shortlist for
days on a premise nobody had checked — that a person held a professional
licence. When the premise turned out to be false, finding everything that
depended on it was manual archaeology, and one dependent was missed entirely:
a competitive analysis still ranked that same premise as its strongest
advantage weeks later.

So: assumptions are first-class, dependents are recorded at the moment a claim
is used, and "what breaks if this is wrong" is a query rather than a memory.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping


class CyclicDependency(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Assumption:
    """Something believed, with an honest record of how well it is known."""

    id: str
    statement: str
    confidence: float = 0.5
    evidence_ids: tuple[str, ...] = ()
    source: str = "unverified"        # evidence id, "operator", or "unverified"
    verified_at: str | None = None    # ISO 8601, supplied by the caller

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"assumption {self.id!r}: confidence outside 0..1")

    @property
    def verified(self) -> bool:
        return bool(self.evidence_ids) or self.verified_at is not None

    def invalidated(self, *, reason: str) -> "Assumption":
        if not reason.strip():
            raise ValueError("invalidating an assumption requires a reason")
        return replace(self, confidence=0.0, source=f"invalidated: {reason}", verified_at=None)


@dataclass(frozen=True, slots=True)
class Dependent:
    """An artifact that would be wrong if the assumption were wrong."""

    id: str
    kind: str    # "idea" | "score" | "rule" | "assumption" | "dossier"
    role: str    # why it depends, in the operator's language


@dataclass(frozen=True, slots=True)
class Impact:
    assumption_id: str
    statement: str
    direct: tuple[Dependent, ...]
    transitive: tuple[Dependent, ...]

    @property
    def total(self) -> int:
        return len(self.direct) + len(self.transitive)

    def by_kind(self) -> Mapping[str, tuple[Dependent, ...]]:
        grouped: dict[str, list[Dependent]] = defaultdict(list)
        for dep in self.direct + self.transitive:
            grouped[dep.kind].append(dep)
        return {k: tuple(v) for k, v in sorted(grouped.items())}

    def report(self) -> str:
        """Operator-facing. Leads with the count, then names what is affected."""
        if not self.total:
            return f"Nothing depends on {self.assumption_id}."
        lines = [
            f"{self.total} artifact(s) depend on this assumption:",
            f'  "{self.statement}"',
            "",
        ]
        for kind, deps in self.by_kind().items():
            lines.append(f"  {kind} ({len(deps)})")
            for dep in deps:
                lines.append(f"    {dep.id:<28} {dep.role}")
        return "\n".join(lines)


@dataclass(slots=True)
class AssumptionGraph:
    """Assumptions plus the edges from each to whatever it supports.

    An assumption may support another assumption, so invalidation is a
    transitive walk rather than a single lookup.
    """

    assumptions: dict[str, Assumption] = field(default_factory=dict)
    _edges: dict[str, dict[str, Dependent]] = field(default_factory=lambda: defaultdict(dict))

    # -- construction ----------------------------------------------------
    def add(self, assumption: Assumption) -> None:
        self.assumptions[assumption.id] = assumption

    def depends(self, dependent: Dependent, on: str) -> None:
        """Record that ``dependent`` rests on assumption ``on``."""
        if on not in self.assumptions:
            raise KeyError(f"unknown assumption {on!r}; add it before recording dependents")
        if dependent.id == on:
            raise CyclicDependency(f"{on!r} cannot depend on itself")
        self._edges[on][dependent.id] = dependent
        self._guard_cycles(on)

    # -- queries ---------------------------------------------------------
    def direct_dependents(self, assumption_id: str) -> tuple[Dependent, ...]:
        return tuple(self._edges.get(assumption_id, {}).values())

    def impact(self, assumption_id: str) -> Impact:
        """Everything that becomes stale if this assumption changes."""
        assumption = self.assumptions.get(assumption_id)
        if assumption is None:
            raise KeyError(f"unknown assumption {assumption_id!r}")

        direct = self.direct_dependents(assumption_id)
        seen = {assumption_id} | {d.id for d in direct}
        transitive: list[Dependent] = []

        frontier = [d for d in direct if d.kind == "assumption"]
        while frontier:
            nxt = frontier.pop()
            for dep in self.direct_dependents(nxt.id):
                if dep.id in seen:
                    continue
                seen.add(dep.id)
                transitive.append(dep)
                if dep.kind == "assumption":
                    frontier.append(dep)

        return Impact(assumption_id, assumption.statement, direct, tuple(transitive))

    def invalidate(self, assumption_id: str, *, reason: str) -> Impact:
        """Mark an assumption false and return everything it takes with it."""
        current = self.assumptions[assumption_id]
        self.assumptions[assumption_id] = current.invalidated(reason=reason)
        return self.impact(assumption_id)

    def load_bearing_unverified(self) -> tuple[Impact, ...]:
        """Unverified assumptions that something actually rests on.

        This is the standing answer to "what is the shortlist quietly betting
        on?" — the query nobody could run before, sorted worst first.
        """
        found = [
            self.impact(a.id)
            for a in self.assumptions.values()
            if not a.verified and self.direct_dependents(a.id)
        ]
        return tuple(sorted(found, key=lambda i: (-i.total, i.assumption_id)))

    # -- internals -------------------------------------------------------
    def _guard_cycles(self, start: str) -> None:
        stack = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            for dep in self.direct_dependents(node):
                if dep.kind != "assumption":
                    continue
                if dep.id in path:
                    raise CyclicDependency(" -> ".join(path + (dep.id,)))
                stack.append((dep.id, path + (dep.id,)))


def summarise(graph: AssumptionGraph) -> str:
    """A short standing report for the interface's facts pane."""
    risky = graph.load_bearing_unverified()
    if not risky:
        return "Every load-bearing assumption is verified."
    lines = [f"{len(risky)} unverified assumption(s) are load-bearing:"]
    for impact in risky:
        lines.append(f"  {impact.assumption_id:<24} {impact.total} dependent(s)  {impact.statement}")
    return "\n".join(lines)
