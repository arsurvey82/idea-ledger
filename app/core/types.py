"""Domain types for the pure core.

Nothing in this module performs I/O, calls a model, or reads a clock. Every
value that can reach a human carries its provenance, because a number whose
origin is unknown is the defect this whole system exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence


class Provenance(str, Enum):
    """Where a value came from. Calibration reads DERIVED transitions only."""

    DERIVED = "derived"        # the rule engine computed it from evidence
    OVERRIDDEN = "overridden"  # a human edited it; the original is retained
    SEEDED = "seeded"          # imported from prior work, no evidence behind it


class Status(str, Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    ON_HOLD = "on_hold"
    REJECTED = "rejected"
    UNDER_RESEARCHED = "under_researched"


#: Legal status transitions. A rejected idea cannot be silently re-proposed;
#: reviving one is an explicit operator act, not a side effect of a new run.
TRANSITIONS: Mapping[Status, frozenset[Status]] = {
    Status.UNDER_RESEARCHED: frozenset({Status.ACTIVE, Status.REJECTED, Status.BLOCKED}),
    Status.ACTIVE: frozenset({Status.BLOCKED, Status.ON_HOLD, Status.REJECTED}),
    Status.BLOCKED: frozenset({Status.ACTIVE, Status.REJECTED, Status.ON_HOLD}),
    Status.ON_HOLD: frozenset({Status.ACTIVE, Status.REJECTED}),
    Status.REJECTED: frozenset({Status.ON_HOLD}),  # revive to review, never straight to active
}


class IllegalTransition(ValueError):
    pass


def transition(current: Status, target: Status) -> Status:
    if target is current:
        return current  # setting a status to what it already is is a no-op
    if target not in TRANSITIONS[current]:
        raise IllegalTransition(
            f"{current.value} -> {target.value} is not a legal transition; "
            f"allowed: {sorted(s.value for s in TRANSITIONS[current])}"
        )
    return target


@dataclass(frozen=True, slots=True)
class Evidence:
    """A single sourced claim. No url, no claim."""

    id: str
    url: str
    claim: str
    retrieved_at: str          # ISO 8601, supplied by the caller — the core has no clock
    verbatim: str = ""

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"evidence {self.id!r}: url must be resolvable, got {self.url!r}")


@dataclass(frozen=True, slots=True)
class DimensionScore:
    """One scored dimension, with the metacognitive contract attached.

    ``falsifier`` is what makes rescoring cheap: it records, at scoring time,
    what would have to change for this number to move.
    """

    dimension: str
    value: int
    evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.5
    falsifier: str = ""
    provenance: Provenance = Provenance.DERIVED
    original_value: int | None = None   # retained when provenance is OVERRIDDEN

    def __post_init__(self) -> None:
        if not 1 <= self.value <= 10:
            raise ValueError(f"{self.dimension}: value {self.value} outside 1..10")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.dimension}: confidence {self.confidence} outside 0..1")
        if self.provenance is Provenance.OVERRIDDEN and self.original_value is None:
            raise ValueError(f"{self.dimension}: an override must retain the original value")

    def override(self, new_value: int, *, reason: str) -> "DimensionScore":
        if not reason.strip():
            raise ValueError("an override requires a reason")
        return replace(
            self,
            value=new_value,
            original_value=self.value if self.original_value is None else self.original_value,
            provenance=Provenance.OVERRIDDEN,
            falsifier=reason,
        )


@dataclass(frozen=True, slots=True)
class ScoreVector:
    """A full scorecard, pinned to the rubric version that produced it.

    Two vectors on different rubric versions are never comparable. That is
    enforced here rather than left to the caller's discipline.
    """

    rubric_version: int
    track: str
    scores: tuple[DimensionScore, ...]

    def __post_init__(self) -> None:
        names = [s.dimension for s in self.scores]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate dimensions in vector: {names}")

    def by_name(self, dimension: str) -> DimensionScore | None:
        return next((s for s in self.scores if s.dimension == dimension), None)

    @property
    def has_override(self) -> bool:
        return any(s.provenance is Provenance.OVERRIDDEN for s in self.scores)

    @property
    def all_derived(self) -> bool:
        return all(s.provenance is Provenance.DERIVED for s in self.scores)


@dataclass(frozen=True, slots=True)
class Idea:
    id: str
    name: str
    track: str
    status: Status = Status.UNDER_RESEARCHED
    vector: ScoreVector | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()

    def with_status(self, target: Status) -> "Idea":
        return replace(self, status=transition(self.status, target))


@dataclass(frozen=True, slots=True)
class FactBase:
    """The operator's own profile. Never shipped, never committed."""

    fields: Mapping[str, Any]

    def get(self, path: str, default: Any = None) -> Any:
        return self.fields.get(path, default)

    def has(self, path: str) -> bool:
        return path in self.fields

    @property
    def field_names(self) -> frozenset[str]:
        return frozenset(self.fields)


@dataclass(frozen=True, slots=True)
class Rubric:
    version: int
    dimensions: tuple[str, ...]
    tracks: tuple[str, ...]

    @property
    def max_total(self) -> int:
        return len(self.dimensions) * 10

    def validates(self, vector: ScoreVector) -> Sequence[str]:
        """Return a list of problems; empty means the vector fits this rubric."""
        problems: list[str] = []
        if vector.rubric_version != self.version:
            problems.append(
                f"vector is rubric v{vector.rubric_version}, rubric is v{self.version}"
            )
        present = {s.dimension for s in vector.scores}
        for missing in sorted(set(self.dimensions) - present):
            problems.append(f"missing dimension {missing!r}")
        for extra in sorted(present - set(self.dimensions)):
            problems.append(f"unknown dimension {extra!r}")
        if vector.track not in self.tracks:
            problems.append(f"unknown track {vector.track!r}")
        return problems
