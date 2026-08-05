"""The model-backed Evaluator: the four call sites, wired to a provider.

This is the seam between the deterministic pipeline and a probabilistic
backend. Everything defensive about the model lives here, so the pipeline can
stay a plain sequence of steps:

  * The frozen prefix is built once and reused, so the cache breakpoint has
    something stable to sit behind.
  * The calibration prior is injected into the two stages where first-pass
    optimism was actually measured.
  * A competitor claim without a resolvable url is discarded before it can
    influence anything. That single rule is the fix for the largest error in
    the system this replaces.
  * The refute stage runs on a fresh prefix carrying only the claim and its
    evidence, never the generation transcript, so the generator's framing
    cannot anchor its own referee.
  * A refusal is surfaced, never silently treated as an empty result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.calibration import CalibrationStore
from .core.rules import Rule, RuleTarget
from .core.types import DimensionScore, Evidence, FactBase, Idea, Rubric
from .pipeline import Candidate, EvidenceBatch, Verdict
from .providers.base import Completion, Provider

DEFAULTS_DIR = Path(__file__).resolve().parents[1] / "defaults"
SCHEMA_FILE = DEFAULTS_DIR / "schemas.json"


class ModelRefused(RuntimeError):
    """The provider declined the request. Surfaced, never swallowed."""


class MalformedModelOutput(RuntimeError):
    """The response did not match its schema. Treated as failure, not salvaged."""


def load_schemas(path: Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    raw = json.loads((path or SCHEMA_FILE).read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@dataclass(slots=True)
class ModelEvaluator:
    provider: Provider
    rubric: Rubric
    facts: FactBase
    rules: Sequence[Rule] = ()
    calibration: CalibrationStore | None = None
    schemas: Mapping[str, Mapping[str, Any]] = field(default_factory=load_schemas)
    _evidence_seq: int = 0

    # -- prompt construction ---------------------------------------------
    def frozen_prefix(self, stage: str) -> str:
        """Stable across every candidate in a run. Cached at the provider.

        Order matters: the parts that never change come first, so a longer
        prefix stays cacheable as the volatile tail grows.
        """
        parts = [
            "You evaluate business ideas for one specific operator.",
            "",
            "You never assign a total score. You supply evidence and per-dimension "
            "judgements; the total is computed outside this conversation, by code.",
            "",
            f"Rubric v{self.rubric.version}. Dimensions, each scored 1-10:",
            "  " + ", ".join(self.rubric.dimensions),
            f"Tracks: {', '.join(self.rubric.tracks)}. Ideas are only ever compared "
            "within a track, so do not reason about cross-track ranking.",
            "",
            "Operator facts:",
        ]
        for key in sorted(self.facts.field_names):
            parts.append(f"  {key}: {self.facts.get(key)!r}")

        fragments = [
            r.fragment for r in self.rules
            if r.active and r.target is RuleTarget.NEURAL and r.stage == stage
        ]
        if fragments:
            parts += ["", "Standing instructions for this stage:"]
            parts += [f"  - {f}" for f in fragments]

        prior = self.calibration.prior_text() if self.calibration else ""
        if prior and stage in {"generate", "evidence"}:
            parts += ["", prior]

        return "\n".join(parts)

    # -- the four call sites ---------------------------------------------
    def propose(self, brief: str, count: int, prior: str) -> Sequence[Candidate]:
        payload = self._call(
            stage="generate",
            schema="propose",
            prompt=(
                f"Propose {count} distinct candidate ideas for: {brief}\n\n"
                "Each must be a proven model already operating in the market, paired "
                "with a specific edge this operator actually has. Do not propose "
                "anything whose pitch is that nobody else does it."
            ),
            search=False,
        )
        out: list[Candidate] = []
        for row in payload.get("candidates", []):
            out.append(
                Candidate(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    track=str(row["track"]),
                    fields=dict(row.get("fields", {})),
                    rationale=str(row.get("rationale", "")),
                )
            )
        return out

    def gather(self, candidate: Candidate, attempt: int) -> EvidenceBatch:
        payload = self._call(
            stage="evidence",
            schema="gather",
            prompt=(
                f"Find named competitors for: {candidate.name} ({candidate.rationale}).\n"
                f"This is search attempt {attempt}. Report only competitors you have "
                "actually read a page for, each with the url you read. A competitor "
                "you cannot cite does not count and must be omitted."
            ),
            search=True,
        )

        kept: list[Evidence] = []
        names: list[str] = []
        for row in payload.get("competitors", []):
            url = str(row.get("url", ""))
            if not url.startswith(("http://", "https://")):
                continue  # uncited claims are discarded, not downweighted
            self._evidence_seq += 1
            kept.append(
                Evidence(
                    id=f"e{self._evidence_seq}",
                    url=url,
                    claim=f"{row.get('name', '?')}: {row.get('positioning', '')}",
                    retrieved_at=_now(),
                    verbatim=str(row.get("verbatim", "")),
                )
            )
            names.append(str(row.get("name", "")))

        return EvidenceBatch(
            evidence=tuple(kept),
            competitors=tuple(n for n in names if n),
            exhausted=bool(payload.get("exhausted", False)),
        )

    def judge(
        self, candidate: Candidate, evidence: Sequence[Evidence]
    ) -> Sequence[DimensionScore]:
        payload = self._call(
            stage="evidence",
            schema="judge",
            prompt=(
                f"Score {candidate.name} on each rubric dimension, using only the "
                "evidence below. For each dimension give a value, your confidence, "
                "the evidence ids you relied on, and a falsifier: what would have to "
                "be true for this value to move.\n\n" + _render_evidence(evidence)
            ),
            search=False,
        )

        scores: list[DimensionScore] = []
        for row in payload.get("dimensions", []):
            scores.append(
                DimensionScore(
                    dimension=str(row["dimension"]),
                    value=int(row["value"]),
                    evidence_ids=tuple(str(x) for x in row.get("evidence_ids", ())),
                    confidence=float(row.get("confidence", 0.5)),
                    falsifier=str(row.get("falsifier", "")),
                )
            )
        missing = set(self.rubric.dimensions) - {s.dimension for s in scores}
        if missing:
            raise MalformedModelOutput(
                f"{candidate.id}: no judgement for {sorted(missing)}"
            )
        return tuple(scores)

    def refute(self, candidate: Candidate, evidence: Sequence[Evidence]) -> Verdict:
        payload = self._call(
            stage="refute",
            schema="refute",
            prompt=(
                f"Try to refute this idea: {candidate.name}. {candidate.rationale}\n\n"
                "Your job is to find the reason it does not work: a named incumbent "
                "already serving this customer, a regulatory bar, or economics that "
                "have already failed for someone else. Search for it.\n\n"
                "If you are uncertain, answer refuted=true. A wrong rejection costs "
                "one idea; a wrong acceptance costs months.\n\n"
                + _render_evidence(evidence)
            ),
            search=True,
            fresh=True,
        )
        return Verdict(
            refuted=bool(payload["refuted"]),
            basis=str(payload.get("basis", "")),
            evidence_ids=tuple(str(x) for x in payload.get("evidence_ids", ())),
        )

    def render(self, idea: Idea, scored: Any) -> str:
        completion = self._raw(
            stage="render",
            prompt=(
                f"Write a short dossier narrative for {idea.name}. Do not state any "
                "score, total, or rank; those are shown separately and restating them "
                "here risks contradicting the computed value."
            ),
            schema=None,
            search=False,
        )
        return completion.text

    # -- plumbing ---------------------------------------------------------
    def _call(
        self, *, stage: str, schema: str, prompt: str, search: bool, fresh: bool = False
    ) -> Mapping[str, Any]:
        completion = self._raw(
            stage=stage, prompt=prompt, schema=self.schemas[schema], search=search, fresh=fresh
        )
        if completion.parsed is None:
            raise MalformedModelOutput(
                f"{stage}: response did not parse against the {schema!r} schema"
            )
        return completion.parsed

    def _raw(
        self,
        *,
        stage: str,
        prompt: str,
        schema: Mapping[str, Any] | None,
        search: bool,
        fresh: bool = False,
    ) -> Completion:
        system = (
            "You are checking one specific claim. You have not seen how it was "
            "generated, and should not assume it is sound."
            if fresh
            else self.frozen_prefix(stage)
        )
        completion = self.provider.complete(
            system=system,
            prompt=prompt,
            schema=schema,
            search=search,
            cache_prefix=not fresh,
        )
        if completion.refused:
            raise ModelRefused(f"{stage}: {completion.refusal_reason}")
        return completion


def _render_evidence(evidence: Sequence[Evidence]) -> str:
    if not evidence:
        return "(no evidence gathered)"
    return "Evidence:\n" + "\n".join(
        f"  [{e.id}] {e.claim} <{e.url}>" for e in evidence
    )


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
