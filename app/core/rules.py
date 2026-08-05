"""Constraint predicates and gate evaluation.

Operator-authored rules arrive as natural language, get parsed by a model, and
land here. They therefore must never become executable Python: a compiled rule
is data, evaluated by the small interpreter below. There is no ``eval``, no
``exec``, and no attribute traversal in this module, deliberately.

The predicate language is intentionally tiny:

    {"field": "capital_required_usd", "op": "lte", "value": 20000}
    {"field": "capital_required_usd", "op": "lte", "other": "budget_ceiling_usd"}
    {"all": [ ...predicates... ]}
    {"any": [ ...predicates... ]}
    {"not": {...predicate...}}

``other`` compares against another field in scope rather than a literal, so a
rule can say "capital must not exceed my ceiling" without copying the ceiling
into the rule. Copying it is how a constraint ends up stated three different
ways in three different places, with nothing recording which one is real.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .types import FactBase, Idea


class RuleTarget(str, Enum):
    """Where a compiled rule executes.

    SYMBOLIC rules are decidable from structured fields and run every time at
    zero token cost. NEURAL rules require judgement and become a fragment of a
    stage's frozen prompt prefix.
    """

    SYMBOLIC = "symbolic"
    NEURAL = "neural"


class ChangeClass(str, Enum):
    """How disruptive activating a rule is. See the HLD, section 7."""

    ADDITIVE = "additive"                # re-gates; stored numbers keep their meaning
    BREAKING = "breaking"                # vector shape or scale changes
    SILENT_BREAKING = "silent_breaking"  # a dimension's meaning changes; shape does not


class MalformedPredicate(ValueError):
    pass


class UnresolvedInput(ValueError):
    """A rule referenced a field that does not exist in the fact base or idea."""


_COMPARISONS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}
_MEMBERSHIP = {
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "contains": lambda a, b: b in a,
}
_UNARY = {
    "exists": lambda a: a is not None,
    "missing": lambda a: a is None,
    "truthy": lambda a: bool(a),
    "falsy": lambda a: not bool(a),
}

OPS: frozenset[str] = frozenset(_COMPARISONS) | frozenset(_MEMBERSHIP) | frozenset(_UNARY)


def referenced_fields(predicate: Mapping[str, Any]) -> frozenset[str]:
    """Every field a predicate reads. This is the upstream half of resolution."""
    if "field" in predicate:
        names = {str(predicate["field"])}
        if "other" in predicate:
            names.add(str(predicate["other"]))
        return frozenset(names)
    for combinator in ("all", "any"):
        if combinator in predicate:
            branches = predicate[combinator]
            if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
                raise MalformedPredicate(f"{combinator!r} takes a list of predicates")
            out: frozenset[str] = frozenset()
            for branch in branches:
                out |= referenced_fields(branch)
            return out
    if "not" in predicate:
        return referenced_fields(predicate["not"])
    raise MalformedPredicate(f"predicate has no field, all, any or not: {predicate!r}")


def evaluate(predicate: Mapping[str, Any], scope: Mapping[str, Any]) -> bool:
    """Evaluate a predicate against a flat name -> value scope."""
    if "all" in predicate:
        return all(evaluate(p, scope) for p in predicate["all"])
    if "any" in predicate:
        return any(evaluate(p, scope) for p in predicate["any"])
    if "not" in predicate:
        return not evaluate(predicate["not"], scope)

    if "field" not in predicate or "op" not in predicate:
        raise MalformedPredicate(f"leaf predicate needs field and op: {predicate!r}")

    name = str(predicate["field"])
    op = str(predicate["op"])
    if op not in OPS:
        raise MalformedPredicate(f"unknown op {op!r}; known: {sorted(OPS)}")
    if name not in scope:
        raise UnresolvedInput(name)

    left = scope[name]
    if op in _UNARY:
        return _UNARY[op](left)

    if "other" in predicate:
        other = str(predicate["other"])
        if other not in scope:
            raise UnresolvedInput(other)
        right = scope[other]
    elif "value" in predicate:
        right = predicate["value"]
    else:
        raise MalformedPredicate(
            f"op {op!r} requires a value or an other field: {predicate!r}"
        )

    if op in _MEMBERSHIP:
        return _MEMBERSHIP[op](left, right)
    if left is None:
        return False  # a comparison against a missing value is false, never a crash
    try:
        return _COMPARISONS[op](left, right)
    except TypeError as exc:
        raise MalformedPredicate(
            f"cannot compare {left!r} {op} {right!r} for field {name!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class Rule:
    """A compiled rule. Shipped gates and operator rules share this shape."""

    id: str
    description: str
    target: RuleTarget
    stage: str
    predicate: Mapping[str, Any] | None = None  # SYMBOLIC only
    fragment: str = ""                          # NEURAL only
    author: str = "shipped"
    version: int = 1
    active: bool = True
    change_class: ChangeClass = ChangeClass.ADDITIVE

    def __post_init__(self) -> None:
        if self.target is RuleTarget.SYMBOLIC:
            if not self.predicate:
                raise MalformedPredicate(f"rule {self.id!r}: symbolic rule needs a predicate")
            referenced_fields(self.predicate)  # raises early on a malformed tree
        elif not self.fragment.strip():
            raise MalformedPredicate(f"rule {self.id!r}: neural rule needs a fragment")

    @property
    def inputs(self) -> frozenset[str]:
        return referenced_fields(self.predicate) if self.predicate else frozenset()


@dataclass(frozen=True, slots=True)
class GateFailure:
    rule_id: str
    description: str
    missing_input: str | None = None

    def __str__(self) -> str:
        if self.missing_input:
            return f"{self.rule_id}: unresolved input {self.missing_input!r}"
        return f"{self.rule_id}: {self.description}"


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    failures: tuple[GateFailure, ...] = ()

    def __bool__(self) -> bool:
        return self.passed


def build_scope(idea: Idea, facts: FactBase) -> dict[str, Any]:
    """Flatten an idea and the fact base into one predicate scope.

    Idea fields win over fact-base fields on collision, and the collision is
    surfaced rather than silent: a rule that reads ``budget`` should not
    silently switch meaning depending on which object happens to define it.
    """
    scope: dict[str, Any] = dict(facts.fields)
    scope.update(idea.fields)
    scope.setdefault("track", idea.track)
    scope.setdefault("status", idea.status.value)
    return scope


def apply_gates(idea: Idea, facts: FactBase, rules: Iterable[Rule]) -> GateResult:
    """Run every active symbolic rule bound to the gate stage.

    An unresolved input is a failure, not a pass. A rule that references a
    field nobody defined must never quietly evaluate to true — that is how a
    constraint ends up documented but never enforced.
    """
    scope = build_scope(idea, facts)
    failures: list[GateFailure] = []

    for rule in rules:
        if not rule.active or rule.target is not RuleTarget.SYMBOLIC:
            continue
        try:
            if not evaluate(rule.predicate or {}, scope):
                failures.append(GateFailure(rule.id, rule.description))
        except UnresolvedInput as exc:
            failures.append(
                GateFailure(rule.id, rule.description, missing_input=str(exc))
            )

    return GateResult(passed=not failures, failures=tuple(failures))


def unresolved_inputs(rules: Iterable[Rule], known: Iterable[str]) -> dict[str, frozenset[str]]:
    """Upstream resolution: which rules reference fields nothing defines.

    Called at rule-intake time so a new rule blocks with a specific request
    ("needs field ``licences_held``") instead of activating and never firing.
    """
    known_set = frozenset(known)
    gaps: dict[str, frozenset[str]] = {}
    for rule in rules:
        missing = rule.inputs - known_set
        if missing:
            gaps[rule.id] = missing
    return gaps
