"""Score assembly and ranking.

The model never emits a total. It emits evidence and per-dimension judgements;
this module produces every number a human sees. That is what stops a model
scoring to a threshold it can see — the measured failure in the prior system,
where the two gated dimensions collapsed to near-constants while the rubric
claimed a 1..10 range.

Ranking is always within a track. A physical-goods idea and a service idea are
not on one scale, and cross-track comparison is refused rather than discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .types import Idea, Provenance, Rubric, ScoreVector


class NotComparable(ValueError):
    """Raised when a caller tries to rank across rubric versions or tracks."""


@dataclass(frozen=True, slots=True)
class Threshold:
    dimension: str
    minimum: int

    def holds_for(self, vector: ScoreVector) -> bool:
        score = vector.by_name(self.dimension)
        return score is not None and score.value >= self.minimum


@dataclass(frozen=True, slots=True)
class Scored:
    idea_id: str
    name: str
    track: str
    rubric_version: int
    total: int
    max_total: int
    thresholds_met: bool
    failed_thresholds: tuple[str, ...]
    has_override: bool
    mean_confidence: float

    @property
    def display(self) -> str:
        return f"{self.total}/{self.max_total}"


def total(vector: ScoreVector) -> int:
    return sum(s.value for s in vector.scores)


def mean_confidence(vector: ScoreVector) -> float:
    if not vector.scores:
        return 0.0
    return sum(s.confidence for s in vector.scores) / len(vector.scores)


def comparable(a: ScoreVector, b: ScoreVector) -> bool:
    """Two vectors are comparable only on the same rubric version and track."""
    return a.rubric_version == b.rubric_version and a.track == b.track


def score(idea: Idea, rubric: Rubric, thresholds: Sequence[Threshold]) -> Scored:
    if idea.vector is None:
        raise ValueError(f"idea {idea.id!r} has no vector to score")
    problems = rubric.validates(idea.vector)
    if problems:
        raise NotComparable(f"idea {idea.id!r}: " + "; ".join(problems))

    failed = tuple(t.dimension for t in thresholds if not t.holds_for(idea.vector))
    return Scored(
        idea_id=idea.id,
        name=idea.name,
        track=idea.track,
        rubric_version=idea.vector.rubric_version,
        total=total(idea.vector),
        max_total=rubric.max_total,
        thresholds_met=not failed,
        failed_thresholds=failed,
        has_override=idea.vector.has_override,
        mean_confidence=mean_confidence(idea.vector),
    )


def rank_within_track(scored: Iterable[Scored], track: str) -> list[Scored]:
    """Rank one track's ideas. Mixed rubric versions are refused, not sorted."""
    cohort = [s for s in scored if s.track == track]
    versions = {s.rubric_version for s in cohort}
    if len(versions) > 1:
        raise NotComparable(
            f"track {track!r} spans rubric versions {sorted(versions)}; "
            "re-run the evidence pass or view them separately"
        )
    return sorted(cohort, key=lambda s: (-s.total, s.name))


def rank_all(scored: Iterable[Scored]) -> Mapping[str, list[Scored]]:
    """Ranked cohorts, keyed by track. There is deliberately no global ranking."""
    items = list(scored)
    return {track: rank_within_track(items, track) for track in sorted({s.track for s in items})}


def calibration_pairs(
    before: ScoreVector, after: ScoreVector
) -> list[tuple[str, int, int]]:
    """Per-dimension (name, first_pass, verified) for DERIVED -> DERIVED only.

    Human overrides are excluded. If a correction a person made were counted as
    model error, the calibration prior would drift toward measuring the operator
    rather than the model.
    """
    if not comparable(before, after):
        raise NotComparable("calibration needs two vectors on the same rubric and track")

    pairs: list[tuple[str, int, int]] = []
    for old in before.scores:
        new = after.by_name(old.dimension)
        if new is None:
            continue
        if old.provenance is Provenance.DERIVED and new.provenance is Provenance.DERIVED:
            pairs.append((old.dimension, old.value, new.value))
    return pairs
