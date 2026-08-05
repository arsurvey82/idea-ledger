"""Rule intake: plain language in, orchestration artifact out.

The operator types "reject anything needing a licence I don't hold" and it
becomes a live part of the pipeline. Five things happen between those points,
and only the first involves a model:

    compile   [A5] parse the sentence into a typed proposal
    triage    decidable from fields -> predicate; needs judgement -> fragment
    resolve   place it against the manifest; refuse if it reads unknown fields
    preview   dry-run against the existing ledger, and show the blast radius
    activate  versioned, attributed, reversible - and never automatic

Nothing activates without a human seeing the preview. A natural-language rule
compiled to a predicate can simply be wrong, and the mitigation for that is not
a cleverer compiler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .core.manifest import Manifest, Placement, UnplaceableRule
from .core.rules import (
    ChangeClass,
    MalformedPredicate,
    Rule,
    RuleTarget,
    apply_gates,
    referenced_fields,
)
from .core.types import FactBase, Idea, Status


class AmbiguousRule(ValueError):
    """The compiler could not decide; the operator must answer one question."""


@dataclass(frozen=True, slots=True)
class RuleProposal:
    """What the compiler returned, before anyone has agreed to it."""

    raw: str
    target: str                    # "symbolic" | "neural" | "ambiguous"
    description: str
    reasoning: str
    inputs: tuple[str, ...] = ()
    predicate: Mapping[str, Any] | None = None
    fragment: str = ""
    change_class: ChangeClass = ChangeClass.ADDITIVE
    ambiguity: str = ""

    @property
    def decided(self) -> bool:
        return self.target in {"symbolic", "neural"}

    def forced_to(self, target: str) -> "RuleProposal":
        """The operator overrides the compiler's triage. Always allowed.

        Forcing to ``neural`` on a proposal the compiler never wrote a fragment
        for falls back to the operator's own sentence, which is what they meant
        the instruction to be.
        """
        if target not in {"symbolic", "neural"}:
            raise ValueError(f"cannot force target {target!r}")
        fragment = self.fragment
        if target == "neural" and not fragment.strip():
            fragment = self.description.strip() or self.raw.strip()
        return replace(
            self,
            target=target,
            fragment=fragment,
            reasoning=f"{self.reasoning} (forced by operator)".strip(),
        )

    def to_rule(self, rule_id: str, *, stage: str, author: str) -> Rule:
        if not self.decided:
            raise AmbiguousRule(self.ambiguity or "compiler could not choose a target")
        return Rule(
            id=rule_id,
            description=self.description,
            target=RuleTarget(self.target),
            stage=stage,
            predicate=self.predicate,
            fragment=self.fragment,
            author=author,
            change_class=self.change_class,
            active=False,  # activation is a separate, explicit act
        )


@dataclass(frozen=True, slots=True)
class ImpactPreview:
    """The blast radius, computed before anything changes."""

    rule: Rule
    placement: Placement
    would_reject: tuple[str, ...] = ()
    would_reopen: tuple[str, ...] = ()
    unaffected: int = 0
    invalidates_stages: tuple[str, ...] = ()
    change_class: ChangeClass = ChangeClass.ADDITIVE
    notes: tuple[str, ...] = ()

    @property
    def breaking(self) -> bool:
        return self.change_class is not ChangeClass.ADDITIVE

    def report(self) -> str:
        lines = [
            f"{self.rule.id}  ->  {self.placement.stage} ({self.placement.target})",
            f"  {self.rule.description}",
            "",
        ]
        if self.rule.target is RuleTarget.NEURAL:
            lines.append(
                "  A prompt-stage rule changes future judgements only. Existing "
                "scores are untouched until you re-run."
            )
        else:
            lines.append(
                f"  would reject {len(self.would_reject)} currently-active idea(s)"
                + (f": {', '.join(self.would_reject)}" if self.would_reject else "")
            )
            lines.append(
                f"  would re-open {len(self.would_reopen)} rejected idea(s)"
                + (f": {', '.join(self.would_reopen)}" if self.would_reopen else "")
            )
            lines.append(f"  leaves {self.unaffected} idea(s) unchanged")

        if self.invalidates_stages:
            lines.append(f"  invalidates downstream: {', '.join(self.invalidates_stages)}")

        if self.breaking:
            lines += [
                "",
                f"  THIS IS A {self.change_class.value.upper().replace('_', ' ')} CHANGE.",
                "  Every stored score stops being comparable to every future score, and",
                "  the missing evidence cannot honestly be backfilled. Confirming will",
                "  bump the rubric version and reset the calibration store.",
            ]
        for note in self.notes:
            lines.append(f"  note: {note}")
        lines += ["", "  Nothing has changed yet."]
        return "\n".join(lines)


@dataclass(slots=True)
class RuleIntake:
    manifest: Manifest
    facts: FactBase
    compiler: Any = None          # anything with .compile_rule(text) -> Mapping
    _seq: int = 0

    # -- 1. compile (the only model call) --------------------------------
    def compile(self, text: str) -> RuleProposal:
        if self.compiler is None:
            raise RuntimeError("no rule compiler configured")
        payload = self.compiler.compile_rule(text)
        return self.from_payload(text, payload)

    @staticmethod
    def from_payload(text: str, payload: Mapping[str, Any]) -> RuleProposal:
        """Turn the A5 response into a proposal. Kept pure so it is testable."""
        predicate: Mapping[str, Any] | None = None
        raw_predicate = payload.get("predicate")
        if raw_predicate:
            predicate = (
                json.loads(raw_predicate) if isinstance(raw_predicate, str) else dict(raw_predicate)
            )
            referenced_fields(predicate)  # raises MalformedPredicate early

        return RuleProposal(
            raw=text,
            target=str(payload.get("target", "ambiguous")),
            description=str(payload.get("description", text)),
            reasoning=str(payload.get("reasoning", "")),
            inputs=tuple(str(i) for i in payload.get("inputs", ())),
            predicate=predicate,
            fragment=str(payload.get("fragment", "")),
            change_class=ChangeClass(payload.get("change_class", "additive")),
            ambiguity=str(payload.get("ambiguity", "")),
        )

    # -- 2/3. triage and placement ---------------------------------------
    def resolve(
        self,
        proposal: RuleProposal,
        *,
        author: str = "operator",
        rule_id: str | None = None,
        preferred_stage: str | None = None,
        declared_fields: Sequence[str] = (),
    ) -> tuple[Rule, Placement]:
        """Place the rule, or refuse with the specific reason."""
        if not proposal.decided:
            raise AmbiguousRule(
                proposal.ambiguity or "the compiler could not choose between a "
                "predicate and a prompt fragment"
            )

        self._seq += 1
        rid = rule_id or f"R-{self._seq:03d}"
        placement = self.manifest.place(
            rule_id=rid,
            target=proposal.target,
            inputs=proposal.inputs,
            preferred_stage=preferred_stage,
            extra_fields=declared_fields,
        )
        rule = proposal.to_rule(rid, stage=placement.stage, author=author)
        return rule, placement

    # -- 4. preview ------------------------------------------------------
    def preview(
        self,
        rule: Rule,
        placement: Placement,
        ideas: Sequence[Idea],
        existing: Sequence[Rule] = (),
    ) -> ImpactPreview:
        """Dry-run against the ledger. Nothing is written."""
        notes: list[str] = []
        would_reject: list[str] = []
        would_reopen: list[str] = []
        unaffected = 0

        if rule.target is RuleTarget.SYMBOLIC:
            candidate_rules = [*existing, replace(rule, active=True)]
            for idea in ideas:
                passes = apply_gates(idea, self.facts, candidate_rules).passed
                if idea.status in {Status.ACTIVE, Status.BLOCKED, Status.ON_HOLD} and not passes:
                    would_reject.append(idea.id)
                elif idea.status is Status.REJECTED and passes:
                    would_reopen.append(idea.id)
                else:
                    unaffected += 1
        else:
            unaffected = len(ideas)
            notes.append(
                "activating a prompt-stage rule invalidates the prompt cache; "
                "batch rule changes rather than trickling them"
            )

        if rule.change_class is ChangeClass.SILENT_BREAKING:
            notes.append(
                "this redefines what an existing dimension means; the score shape "
                "does not change, so nothing else would have signalled it"
            )

        return ImpactPreview(
            rule=rule,
            placement=placement,
            would_reject=tuple(would_reject),
            would_reopen=tuple(would_reopen),
            unaffected=unaffected,
            invalidates_stages=self.manifest.impact_of_placing(placement.stage),
            change_class=rule.change_class,
            notes=tuple(notes),
        )

    # -- 5. activate -----------------------------------------------------
    @staticmethod
    def activate(preview: ImpactPreview, *, confirmed: bool, breaking_confirmed: bool = False) -> Rule:
        """Turn a previewed rule on. Requires an explicit yes, twice if breaking."""
        if not confirmed:
            raise PermissionError("activation requires explicit confirmation")
        if preview.breaking and not breaking_confirmed:
            raise PermissionError(
                f"{preview.change_class.value} change requires a second confirmation: "
                "stored scores stop being comparable and calibration resets"
            )
        return replace(preview.rule, active=True)

    @staticmethod
    def revert(rule: Rule) -> Rule:
        """One-click undo. The rule stays in the ledger, inactive and attributed."""
        return replace(rule, active=False)


def describe_gap(exc: UnplaceableRule) -> str:
    """Turn a placement refusal into something an operator can act on."""
    return str(exc)
