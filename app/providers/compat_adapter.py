"""Make an OpenAI-compatible chat endpoint satisfy the Provider port.

The evaluation pipeline talks to ``Provider.complete()``. OpenAI and OpenRouter
speak chat-completions. This is the joint, and it is deliberately thin: the
schema goes in as ``response_format``, and web search is requested the way each
provider actually offers it rather than assumed.

Search is the honest part. OpenRouter enables it by suffixing the route with
``:online``; OpenAI exposes it only on a different endpoint this build does not
use. So the capability set is computed per provider instead of claimed, and the
negotiation layer refuses the evidence stage where it genuinely cannot run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .base import Capability, Completion, ModelSpec
from .openai_compat import ChatError, OpenAICompatClient, route_parameters

#: Suffix OpenRouter uses to turn on web search for any route.
ONLINE_SUFFIX = ":online"


@dataclass(slots=True)
class CompatProvider:
    """Adapts a chat-completions endpoint to the pipeline's Provider port."""

    provider: str
    api_key: str
    model: str
    on_retry: Any = None

    @property
    def name(self) -> str:
        return self.provider

    def capabilities(self) -> frozenset[Capability]:
        """What this route can actually do, not what the gateway can do.

        This used to declare STRUCTURED_OUTPUT unconditionally. That was the one
        assumption in the whole negotiation layer that was never checked, and it
        is the assumption that failed: cohere/north-mini-code:free accepts
        response_format, returns 200, and answers in markdown, because
        response_format is absent from its supported_parameters. Negotiation
        reported the generate stage as ready and the stage then crashed on the
        schema parse - precisely the pretending this layer exists to prevent.
        """
        if self.provider != "openrouter":
            return frozenset({Capability.STRUCTURED_OUTPUT, Capability.STRICT_TOOLS})

        params = route_parameters(self.api_key, self.model)
        if not params:
            # The manifest was unreachable. Assume the optimistic set rather
            # than blocking every stage on a transient network failure; a wrong
            # guess surfaces as a stage error, not as silent bad data.
            return frozenset({Capability.STRUCTURED_OUTPUT, Capability.STRICT_TOOLS,
                              Capability.SERVER_SEARCH, Capability.THINKING})

        caps: set[Capability] = set()
        if {"response_format", "structured_outputs"} & params:
            caps.add(Capability.STRUCTURED_OUTPUT)
        if "tools" in params:
            caps.add(Capability.STRICT_TOOLS)
        if {"reasoning", "include_reasoning"} & params:
            caps.add(Capability.THINKING)
        # Search is a route suffix rather than a parameter, so it is available
        # on any route the gateway will put online.
        caps.add(Capability.SERVER_SEARCH)
        return frozenset(caps)

    def default_model(self) -> ModelSpec:
        return ModelSpec(
            provider=self.provider,
            model_id=self.model,
            context_tokens=0,
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
        )

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: Mapping[str, object] | None = None,
        search: bool = False,
        cache_prefix: bool = True,
    ) -> Completion:
        model = self.model
        if search and self.provider == "openrouter" and not model.endswith(ONLINE_SUFFIX):
            model += ONLINE_SUFFIX

        client = OpenAICompatClient(
            provider=self.provider,
            api_key=self.api_key,
            model=model,
            on_retry=self.on_retry,
        )
        payload_schema = None
        if schema is not None:
            payload_schema = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": dict(schema)},
            }

        try:
            reply = client.chat_raw(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                response_format=payload_schema,
            )
        except ChatError as exc:
            return Completion(
                text="", parsed=None, citations=(), input_tokens=0, output_tokens=0,
                model_id=model, refused=True, refusal_reason=str(exc),
            )

        parsed: Mapping[str, object] | None = None
        if schema is not None and reply.text.strip():
            try:
                parsed = json.loads(reply.text)
            except json.JSONDecodeError:
                parsed = None

        return Completion(
            text=reply.text,
            parsed=parsed,
            citations=tuple(reply.citations),
            input_tokens=0,
            output_tokens=0,
            model_id=reply.model or model,
        )
