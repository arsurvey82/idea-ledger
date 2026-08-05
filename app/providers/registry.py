"""Concrete provider declarations.

Only capability declarations and model specs live here. The transport for each
provider is a separate module so this one stays importable — and testable —
with no network, no SDK installed, and no key present.

Capability claims are conservative on purpose. Declaring a capability the
backend does not really have converts a loud startup refusal into a silent
wrong answer, which is the failure mode this whole design exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .base import Capability, Completion, ModelSpec


class NotConfigured(RuntimeError):
    """Raised when a provider is used before its transport is wired or keyed."""


@dataclass(frozen=True, slots=True)
class _Declared:
    """A provider known by capability but not yet able to transmit."""

    name: str
    _caps: frozenset[Capability]
    _model: ModelSpec
    _note: str = ""

    def capabilities(self) -> frozenset[Capability]:
        return self._caps

    def default_model(self) -> ModelSpec:
        return self._model

    def complete(self, **_: object) -> Completion:
        raise NotConfigured(
            f"{self.name}: transport not wired yet. "
            f"Capabilities are declared so setup can negotiate before any key is entered."
        )


ANTHROPIC = _Declared(
    name="anthropic",
    _caps=frozenset(
        {
            Capability.STRUCTURED_OUTPUT,
            Capability.STRICT_TOOLS,
            Capability.SERVER_SEARCH,
            Capability.PROMPT_CACHE,
            Capability.BATCH,
            Capability.THINKING,
        }
    ),
    _model=ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        context_tokens=1_000_000,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=25.0,
    ),
)

OPENAI = _Declared(
    name="openai",
    _caps=frozenset(
        {
            Capability.STRUCTURED_OUTPUT,
            Capability.STRICT_TOOLS,
            Capability.SERVER_SEARCH,
            Capability.PROMPT_CACHE,
            Capability.BATCH,
            Capability.THINKING,
        }
    ),
    _model=ModelSpec(
        provider="openai",
        model_id="gpt-5",
        context_tokens=400_000,
        input_cost_per_mtok=0.0,   # filled from the operator's own pricing config
        output_cost_per_mtok=0.0,
    ),
    _note="server search is available on the Responses API surface only",
)

#: OpenRouter routes to whatever model the operator names, so capabilities are
#: a property of the route, not the gateway. The conservative default assumes
#: no server-side search: most routed models have none, and assuming otherwise
#: is precisely the silent failure this design refuses.
OPENROUTER = _Declared(
    name="openrouter",
    _caps=frozenset({Capability.STRUCTURED_OUTPUT}),
    _model=ModelSpec(
        provider="openrouter",
        model_id="",  # operator supplies the route
        context_tokens=0,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    _note="capabilities depend on the routed model; declare them per route",
)


REGISTRY: Mapping[str, _Declared] = {
    "anthropic": ANTHROPIC,
    "openai": OPENAI,
    "openrouter": OPENROUTER,
}


def get(name: str) -> _Declared:
    try:
        return REGISTRY[name.strip().lower()]
    except KeyError:
        raise KeyError(
            f"unknown provider {name!r}; known: {', '.join(sorted(REGISTRY))}"
        ) from None


def with_capabilities(base: _Declared, caps: frozenset[Capability], model_id: str) -> _Declared:
    """Refine a gateway provider once the operator names a route.

    Used by the setup screen: choosing OpenRouter plus a specific model lets the
    operator declare that route's real capabilities, and negotiation re-runs
    against them.
    """
    from dataclasses import replace

    return replace(base, _caps=caps, _model=replace(base._model, model_id=model_id))
