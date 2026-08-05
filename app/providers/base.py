"""Provider port and capability negotiation.

The system is provider-agnostic, but not by pretending every backend is the
same. The safety properties this architecture rests on — a schema enforced at
decode, searched evidence that arrives with citations — are capabilities, and
capabilities differ:

    Anthropic     structured output, strict tools, server search, cache, batch
    OpenAI        structured output, strict tools, server search, cache
    OpenRouter    depends entirely on the routed model; often no server search

So a provider *declares* what it has, the manifest declares what each stage
*requires*, and negotiation happens once at startup as a deterministic check.
A missing capability produces one of three outcomes, in order of preference:

    1. NATIVE       the provider has it
    2. COMPENSATED  another adapter supplies it (e.g. a standalone search API)
    3. REFUSED      the stage cannot run, and says exactly why

Never a silent fallback. A model asked for competitors with no search bound
will happily invent them — that is the single largest measured error in the
system this replaces, and it must fail loudly rather than quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable


class Capability(str, Enum):
    STRUCTURED_OUTPUT = "structured_output"  # response constrained to a JSON schema
    STRICT_TOOLS = "strict_tools"            # tool arguments validated at decode
    SERVER_SEARCH = "server_search"          # provider-side web search with citations
    PROMPT_CACHE = "prompt_cache"            # cacheable frozen prefix
    BATCH = "batch"                          # asynchronous batch submission
    THINKING = "thinking"                    # extended/adaptive reasoning


#: Capabilities without which a stage produces wrong answers rather than slow
#: ones. Missing any of these is REFUSED; the rest degrade to cost or latency.
CORRECTNESS_CRITICAL: frozenset[Capability] = frozenset(
    {Capability.STRUCTURED_OUTPUT, Capability.SERVER_SEARCH}
)


class Resolution(str, Enum):
    NATIVE = "native"
    COMPENSATED = "compensated"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A concrete model on a concrete provider."""

    provider: str
    model_id: str
    context_tokens: int
    input_cost_per_mtok: float
    output_cost_per_mtok: float


@dataclass(frozen=True, slots=True)
class Completion:
    """Normalised result. Callers never see provider-shaped payloads."""

    text: str
    parsed: Mapping[str, object] | None
    citations: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    model_id: str
    refused: bool = False
    refusal_reason: str = ""


@runtime_checkable
class Provider(Protocol):
    """Every backend implements exactly this. Nothing above it is provider-aware."""

    name: str

    def capabilities(self) -> frozenset[Capability]:
        ...

    def default_model(self) -> ModelSpec:
        ...

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: Mapping[str, object] | None = None,
        search: bool = False,
        cache_prefix: bool = True,
    ) -> Completion:
        ...


@dataclass(frozen=True, slots=True)
class StageRequirement:
    stage: str
    requires: frozenset[Capability]
    reason: Mapping[Capability, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageResolution:
    stage: str
    resolution: Resolution
    native: frozenset[Capability] = frozenset()
    compensated: Mapping[Capability, str] = field(default_factory=dict)
    missing: frozenset[Capability] = frozenset()
    explanation: str = ""

    @property
    def runnable(self) -> bool:
        return self.resolution is not Resolution.REFUSED


@dataclass(frozen=True, slots=True)
class Negotiation:
    provider: str
    model_id: str
    stages: tuple[StageResolution, ...]

    @property
    def ok(self) -> bool:
        return all(s.runnable for s in self.stages)

    @property
    def refused(self) -> tuple[StageResolution, ...]:
        return tuple(s for s in self.stages if not s.runnable)

    def report(self) -> str:
        """A setup-screen summary. Written for the operator, not for a log.

        Deliberately ASCII-only: this is printed to a console on whatever
        machine the operator has, and a legacy Windows code page turns a
        typographic dash into a replacement character.
        """
        width = max((len(s.stage) for s in self.stages), default=8) + 2
        lines = [f"{self.provider} / {self.model_id or '(route not chosen)'}"]
        for s in self.stages:
            if s.resolution is Resolution.NATIVE:
                mark = "ok"
                detail = "provider supports every capability this stage needs"
            elif s.resolution is Resolution.COMPENSATED:
                mark = "ok"
                detail = "; ".join(
                    f"{cap.value} supplied by {who}" for cap, who in s.compensated.items()
                )
            else:
                mark = "cannot run"
                detail = s.explanation
            lines.append(f"  {s.stage:<{width}} {mark:<11} {detail}")
        return "\n".join(lines)


def negotiate(
    provider: Provider,
    requirements: Sequence[StageRequirement],
    compensators: Mapping[Capability, str] | None = None,
) -> Negotiation:
    """Resolve every stage against one provider, once, at startup.

    ``compensators`` maps a capability to the name of an adapter that can supply
    it independently of the model provider — a standalone search API being the
    case that matters in practice.
    """
    available = provider.capabilities()
    supplied = dict(compensators or {})
    resolutions: list[StageResolution] = []

    for req in requirements:
        native = req.requires & available
        gap = req.requires - available
        covered = {cap: supplied[cap] for cap in gap if cap in supplied}
        missing = gap - set(covered)

        if not gap:
            resolution = Resolution.NATIVE
            explanation = ""
        elif not missing:
            resolution = Resolution.COMPENSATED
            explanation = ""
        else:
            resolution = Resolution.REFUSED
            explanation = _explain(req, missing)

        resolutions.append(
            StageResolution(
                stage=req.stage,
                resolution=resolution,
                native=native,
                compensated=covered,
                missing=missing,
                explanation=explanation,
            )
        )

    spec = provider.default_model()
    return Negotiation(provider.name, spec.model_id, tuple(resolutions))


def _explain(req: StageRequirement, missing: Iterable[Capability]) -> str:
    parts: list[str] = []
    for cap in sorted(missing, key=lambda c: c.value):
        why = req.reason.get(cap)
        parts.append(f"{cap.value} - {why}" if why else cap.value)
    return "missing " + "; ".join(parts)


class UnknownCapability(ValueError):
    """The manifest named a capability this build does not implement."""


def requirements_from(manifest: object) -> tuple[StageRequirement, ...]:
    """Build stage requirements from the pipeline manifest.

    The manifest is the single source for what each stage needs and why. This
    function is the only place the two layers meet, and it deliberately takes a
    duck-typed object so the provider layer never imports core's concrete type.
    """
    names: Mapping[str, Sequence[str]] = manifest.capability_names()   # type: ignore[attr-defined]
    reasons: Mapping[str, Mapping[str, str]] = manifest.capability_reasons()  # type: ignore[attr-defined]

    built: list[StageRequirement] = []
    for stage, caps in names.items():
        try:
            required = frozenset(Capability(c) for c in caps)
        except ValueError as exc:
            raise UnknownCapability(f"stage {stage!r}: {exc}") from None
        built.append(
            StageRequirement(
                stage=stage,
                requires=required,
                reason={Capability(c): why for c, why in reasons.get(stage, {}).items()},
            )
        )
    return tuple(built)


_cached: tuple[StageRequirement, ...] | None = None


def default_requirements() -> tuple[StageRequirement, ...]:
    """Requirements from the shipped manifest, loaded once."""
    global _cached
    if _cached is None:
        from ..core.manifest import Manifest

        _cached = requirements_from(Manifest.load())
    return _cached
