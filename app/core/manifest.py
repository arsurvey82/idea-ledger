"""The pipeline's description of itself.

"Self-aware" in this system means one specific, unglamorous thing: there is a
declarative file describing every stage, what each one needs, and which kinds of
rule each will accept. Rule placement, upstream resolution, and capability
negotiation all read it. Nothing is inferred and nothing emerges — the system
knows its own shape because its shape is written down.

Core stays pure: this module names capabilities as strings and never imports
the provider layer. Providers depend on core, not the other way round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULTS_DIR = Path(__file__).resolve().parents[2] / "defaults"
MANIFEST_FILE = DEFAULTS_DIR / "manifest.json"


class UnplaceableRule(ValueError):
    """A rule that no stage will accept, or that reads fields nobody defines."""


@dataclass(frozen=True, slots=True)
class Stage:
    id: str
    kind: str                       # "symbolic" | "neural"
    summary: str
    requires: tuple[str, ...] = ()  # capability names; empty for symbolic stages
    reason: Mapping[str, str] = None  # type: ignore[assignment]
    accepts: tuple[str, ...] = ()   # rule targets this stage will host

    def __post_init__(self) -> None:
        if self.kind not in {"symbolic", "neural"}:
            raise ValueError(f"stage {self.id!r}: kind must be symbolic or neural")
        if self.kind == "symbolic" and self.requires:
            raise ValueError(
                f"stage {self.id!r}: a symbolic stage cannot require a model capability"
            )
        object.__setattr__(self, "reason", dict(self.reason or {}))

    @property
    def calls_model(self) -> bool:
        return self.kind == "neural"


@dataclass(frozen=True, slots=True)
class Placement:
    rule_id: str
    stage: str
    target: str

    def __str__(self) -> str:
        return f"{self.rule_id} -> {self.stage} ({self.target})"


@dataclass(frozen=True, slots=True)
class Manifest:
    version: int
    stages: tuple[Stage, ...]
    fact_fields: tuple[str, ...]
    idea_fields: tuple[str, ...]

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> "Manifest":
        raw = json.loads((path or MANIFEST_FILE).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Manifest":
        stages = tuple(
            Stage(
                id=str(s["id"]),
                kind=str(s["kind"]),
                summary=str(s.get("summary", "")),
                requires=tuple(s.get("requires", ())),      # type: ignore[arg-type]
                reason=dict(s.get("reason", {})),           # type: ignore[arg-type]
                accepts=tuple(s.get("accepts", ())),        # type: ignore[arg-type]
            )
            for s in raw["stages"]  # type: ignore[union-attr]
        )
        ids = [s.id for s in stages]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate stage ids in manifest: {ids}")
        return cls(
            version=int(raw["version"]),                    # type: ignore[arg-type]
            stages=stages,
            fact_fields=tuple(raw.get("fact_fields", ())),  # type: ignore[arg-type]
            idea_fields=tuple(raw.get("idea_fields", ())),  # type: ignore[arg-type]
        )

    # -- topology --------------------------------------------------------
    def stage(self, stage_id: str) -> Stage:
        found = next((s for s in self.stages if s.id == stage_id), None)
        if found is None:
            raise KeyError(f"unknown stage {stage_id!r}; known: {[s.id for s in self.stages]}")
        return found

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.stages)

    @property
    def model_stages(self) -> tuple[Stage, ...]:
        return tuple(s for s in self.stages if s.calls_model)

    @property
    def symbolic_stages(self) -> tuple[Stage, ...]:
        return tuple(s for s in self.stages if not s.calls_model)

    def upstream_of(self, stage_id: str) -> tuple[str, ...]:
        idx = self.order.index(self.stage(stage_id).id)
        return self.order[:idx]

    def downstream_of(self, stage_id: str) -> tuple[str, ...]:
        idx = self.order.index(self.stage(stage_id).id)
        return self.order[idx + 1:]

    # -- capability handoff to the provider layer ------------------------
    def capability_names(self) -> Mapping[str, tuple[str, ...]]:
        return {s.id: s.requires for s in self.model_stages}

    def capability_reasons(self) -> Mapping[str, Mapping[str, str]]:
        return {s.id: dict(s.reason) for s in self.model_stages}

    # -- rule placement --------------------------------------------------
    @property
    def known_fields(self) -> frozenset[str]:
        return frozenset(self.fact_fields) | frozenset(self.idea_fields)

    def stages_accepting(self, target: str) -> tuple[str, ...]:
        return tuple(s.id for s in self.stages if target in s.accepts)

    def place(
        self,
        *,
        rule_id: str,
        target: str,
        inputs: Iterable[str] = (),
        preferred_stage: str | None = None,
        extra_fields: Iterable[str] = (),
    ) -> Placement:
        """Decide where a compiled rule belongs, or refuse and say why.

        Upstream resolution happens here: a rule referencing a field nothing
        defines is refused at intake, rather than activating and never firing.
        """
        candidates = self.stages_accepting(target)
        if not candidates:
            raise UnplaceableRule(
                f"{rule_id}: no stage accepts a {target!r} rule; "
                f"stages accept {sorted({t for s in self.stages for t in s.accepts})}"
            )

        if preferred_stage is not None:
            if preferred_stage not in candidates:
                raise UnplaceableRule(
                    f"{rule_id}: stage {preferred_stage!r} does not accept {target!r} rules; "
                    f"candidates are {list(candidates)}"
                )
            chosen = preferred_stage
        else:
            chosen = candidates[0]

        missing = frozenset(inputs) - (self.known_fields | frozenset(extra_fields))
        if missing:
            raise UnplaceableRule(
                f"{rule_id}: reads field(s) nothing defines: {sorted(missing)}. "
                f"Add them to the fact base first, or the rule would never fire."
            )

        return Placement(rule_id=rule_id, stage=chosen, target=target)

    def impact_of_placing(self, stage_id: str) -> tuple[str, ...]:
        """Downstream stages whose output a new rule at this stage invalidates."""
        return self.downstream_of(stage_id)

    # -- reporting -------------------------------------------------------
    def report(self) -> str:
        lines = [f"manifest v{self.version} - {len(self.stages)} stages"]
        for s in self.stages:
            kind = "model" if s.calls_model else "code "
            caps = ", ".join(s.requires) or "-"
            accepts = ", ".join(s.accepts) or "-"
            lines.append(f"  {s.id:<13} {kind}  needs: {caps:<34} accepts: {accepts}")
        model = len(self.model_stages)
        lines.append(
            f"  {model} of {len(self.stages)} stages call a model; "
            f"{len(self.symbolic_stages)} are plain code."
        )
        return "\n".join(lines)


def validate(manifest: Manifest, *, dimensions: Sequence[str] = ()) -> list[str]:
    """Problems that would make the pipeline unrunnable. Empty means healthy."""
    problems: list[str] = []
    if not manifest.stages:
        problems.append("manifest declares no stages")
    if not manifest.stages_accepting("symbolic"):
        problems.append("no stage accepts symbolic rules; gates could never be added")
    if not manifest.stages_accepting("neural"):
        problems.append("no stage accepts neural rules; prompt fragments could never be added")
    for stage in manifest.model_stages:
        for cap in stage.requires:
            if cap not in stage.reason:
                problems.append(
                    f"stage {stage.id!r} requires {cap!r} with no operator-facing reason; "
                    "a refusal must explain the consequence"
                )
    if dimensions and len(dimensions) > 7:
        problems.append(
            f"{len(dimensions)} scoring dimensions exceeds what a reader holds at once"
        )
    return problems
