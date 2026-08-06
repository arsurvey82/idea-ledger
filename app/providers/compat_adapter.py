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
from .openai_compat import ChatError, OpenAICompatClient

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
        caps = {Capability.STRUCTURED_OUTPUT, Capability.STRICT_TOOLS}
        if self.provider == "openrouter":
            # Any route can be put online with the suffix, so search is real here.
            caps.add(Capability.SERVER_SEARCH)
            caps.add(Capability.THINKING)
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
