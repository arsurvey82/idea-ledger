"""M4 — the calibration loop, and M1 — the prior it emits.

Metacognition here is not something the model does. The model self-reports a
confidence; this module measures whether that confidence was warranted and
writes the measurement back into the next prompt.

The measurement comes from real history: the prior system rescored sixteen
ideas after a fuller competitive sweep and every single one moved down, with
one dimension routinely overstated by four or five points. That document was
written by hand, once. Here it is a table the system maintains continuously.

Two disciplines this module keeps:

  * Only DERIVED -> DERIVED transitions are observed. A human correction is
    not model error, and counting it would drift the prior toward measuring
    the operator.
  * Below a sample floor it emits no numeric claim. A confident prior computed
    from three observations is exactly the overconfidence it exists to fix.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .scoring import calibration_pairs
from .types import ScoreVector

#: Below this many observations for a dimension, report the direction only.
MIN_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class Observation:
    dimension: str
    first_pass: int
    verified: int
    idea_id: str
    at: str  # ISO 8601, supplied by the caller

    @property
    def delta(self) -> int:
        """Signed. Negative means the first pass was optimistic."""
        return self.verified - self.first_pass


@dataclass(frozen=True, slots=True)
class Bias:
    dimension: str
    samples: int
    mean_delta: float
    median_delta: float
    worst_delta: int
    downward_share: float  # fraction of observations that moved down

    @property
    def confident(self) -> bool:
        return self.samples >= MIN_SAMPLES

    @property
    def direction(self) -> str:
        if self.mean_delta < -0.5:
            return "optimistic"
        if self.mean_delta > 0.5:
            return "pessimistic"
        return "roughly calibrated"


@dataclass(slots=True)
class CalibrationStore:
    observations: list[Observation] = field(default_factory=list)

    def record_rescore(
        self,
        *,
        idea_id: str,
        before: ScoreVector,
        after: ScoreVector,
        at: str,
    ) -> list[Observation]:
        """Observe one rescore. Overridden dimensions are silently skipped."""
        new = [
            Observation(dimension=dim, first_pass=old, verified=fresh, idea_id=idea_id, at=at)
            for dim, old, fresh in calibration_pairs(before, after)
        ]
        self.observations.extend(new)
        return new

    def bias(self, dimension: str) -> Bias | None:
        rows = [o for o in self.observations if o.dimension == dimension]
        if not rows:
            return None
        deltas = [o.delta for o in rows]
        return Bias(
            dimension=dimension,
            samples=len(rows),
            mean_delta=statistics.fmean(deltas),
            median_delta=statistics.median(deltas),
            worst_delta=min(deltas),
            downward_share=sum(1 for d in deltas if d < 0) / len(deltas),
        )

    def biases(self) -> tuple[Bias, ...]:
        dims = sorted({o.dimension for o in self.observations})
        found = (self.bias(d) for d in dims)
        return tuple(sorted((b for b in found if b), key=lambda b: b.mean_delta))

    # -- M1: the prompt fragment -----------------------------------------
    def prior_text(self, *, dimensions: Sequence[str] | None = None) -> str:
        """The calibration prior injected into the generate and evidence prompts.

        Returns an empty string when there is nothing honest to say, which is
        the correct output on a fresh install — a fabricated prior would be a
        worse failure than no prior.
        """
        biases = [b for b in self.biases() if b.direction == "optimistic"]
        if dimensions:
            wanted = set(dimensions)
            biases = [b for b in biases if b.dimension in wanted]
        if not biases:
            return ""

        total = sum(b.samples for b in biases)
        lines = [
            f"Calibration prior, measured over {total} audited rescoring(s) in this ledger:",
        ]
        for b in biases:
            if b.confident:
                lines.append(
                    f"  - {b.dimension}: first-pass estimates ran optimistic by "
                    f"{abs(b.mean_delta):.1f} points on average "
                    f"(median {abs(b.median_delta):.0f}, worst {abs(b.worst_delta)}, "
                    f"{b.downward_share:.0%} of revisions moved down, n={b.samples})."
                )
            else:
                lines.append(
                    f"  - {b.dimension}: early signal that first-pass estimates run "
                    f"optimistic, but only {b.samples} observation(s) so far - treat as "
                    f"a direction, not a number."
                )
        lines.append(
            "Assume your first estimate is optimistic on these dimensions. Before "
            "assigning a value, state how many named, url-resolvable competitors you "
            "actually found."
        )
        return "\n".join(lines)

    def summary(self) -> str:
        biases = self.biases()
        if not biases:
            return "No rescorings observed yet; no calibration prior available."
        lines = [f"{len(self.observations)} observation(s) across {len(biases)} dimension(s):"]
        for b in biases:
            flag = "" if b.confident else "  (below sample floor)"
            lines.append(
                f"  {b.dimension:<18} mean {b.mean_delta:+.2f}  median {b.median_delta:+.1f}  "
                f"n={b.samples}  {b.direction}{flag}"
            )
        return "\n".join(lines)


def seed_from_history(rows: Iterable[Mapping[str, object]]) -> CalibrationStore:
    """Build a store from an imported audit table.

    Lets an operator carry forward a hand-written revision log — the exact
    artifact the prior system produced once — instead of starting blind.
    """
    store = CalibrationStore()
    for row in rows:
        store.observations.append(
            Observation(
                dimension=str(row["dimension"]),
                first_pass=int(row["first_pass"]),  # type: ignore[arg-type]
                verified=int(row["verified"]),      # type: ignore[arg-type]
                idea_id=str(row.get("idea_id", "imported")),
                at=str(row.get("at", "")),
            )
        )
    return store
