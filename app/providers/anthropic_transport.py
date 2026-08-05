"""Anthropic transport.

The only place in the system that speaks a provider's wire format. Everything
above it sees ``Completion``.

Three details here are load-bearing rather than incidental:

  * ``stop_reason`` is checked before ``content`` is read. A safety refusal
    returns HTTP 200 with an empty or partial content list, so indexing
    ``content[0]`` unconditionally raises on exactly the requests that were
    declined.
  * Search results are mined for their source urls. The evidence stage discards
    any competitor claim without one, so the transport must surface them.
  * The system prompt is sent as a cacheable block. The frozen prefix (rubric,
    gates, fact base, calibration prior) is identical across a batch of
    candidates, and re-sending it uncached is the largest avoidable cost here.

The SDK import is deliberately lazy: the whole test suite, capability
negotiation included, must run with nothing installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .base import Capability, Completion, ModelSpec

MODEL_ID = "claude-opus-5"
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}

#: Non-streaming ceiling. Above roughly this, the SDK refuses the request
#: rather than risk an HTTP timeout, so larger jobs must stream.
MAX_TOKENS_UNSTREAMED = 16_000


class TransportUnavailable(RuntimeError):
    """The SDK is not installed, or no key was supplied."""


@dataclass(slots=True)
class AnthropicTransport:
    api_key: str
    model_id: str = MODEL_ID
    effort: str = "high"
    max_tokens: int = MAX_TOKENS_UNSTREAMED
    _client: Any = None

    name: str = "anthropic"

    # -- port ------------------------------------------------------------
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.STRUCTURED_OUTPUT,
                Capability.STRICT_TOOLS,
                Capability.SERVER_SEARCH,
                Capability.PROMPT_CACHE,
                Capability.BATCH,
                Capability.THINKING,
            }
        )

    def default_model(self) -> ModelSpec:
        return ModelSpec(
            provider=self.name,
            model_id=self.model_id,
            context_tokens=1_000_000,
            input_cost_per_mtok=5.0,
            output_cost_per_mtok=25.0,
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
        request = self.build_request(
            system=system, prompt=prompt, schema=schema, search=search,
            cache_prefix=cache_prefix,
        )
        response = self._messages().create(**request)
        return self.parse(response, expects_json=schema is not None)

    # -- request construction (pure; tested without a client) -------------
    def build_request(
        self,
        *,
        system: str,
        prompt: str,
        schema: Mapping[str, object] | None,
        search: bool,
        cache_prefix: bool,
    ) -> dict[str, Any]:
        system_block: dict[str, Any] = {"type": "text", "text": system}
        if cache_prefix:
            # The breakpoint goes on the last stable block. Volatile content
            # (this candidate, this question) lives in the user turn, after it.
            system_block["cache_control"] = {"type": "ephemeral"}

        request: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": [system_block],
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if schema is not None:
            request["output_config"]["format"] = {
                "type": "json_schema",
                "schema": dict(schema),
            }
        if search:
            request["tools"] = [dict(WEB_SEARCH_TOOL)]
        return request

    # -- response handling (pure; tested with recorded payloads) ----------
    @staticmethod
    def parse(response: Any, *, expects_json: bool) -> Completion:
        stop_reason = getattr(response, "stop_reason", None)
        model_id = getattr(response, "model", "")
        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", 0) or 0
        out_tokens = getattr(usage, "output_tokens", 0) or 0

        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            return Completion(
                text="",
                parsed=None,
                citations=(),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                model_id=model_id,
                refused=True,
                refusal_reason=f"declined by safety classifier ({category})",
            )

        text_parts: list[str] = []
        citations: list[str] = []
        for block in getattr(response, "content", []) or []:
            kind = getattr(block, "type", "")
            if kind == "text":
                text_parts.append(getattr(block, "text", ""))
            elif kind == "web_search_tool_result":
                citations.extend(_urls_from_search(block))

        text = "".join(text_parts)
        parsed: Mapping[str, object] | None = None
        if expects_json and text.strip():
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Should not happen with a schema enforced at decode; treated as
                # a hard failure rather than salvaged, so it is visible.
                parsed = None

        return Completion(
            text=text,
            parsed=parsed,
            citations=tuple(dict.fromkeys(citations)),  # de-duped, order kept
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model_id=model_id,
        )

    # -- internals -------------------------------------------------------
    def _messages(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise TransportUnavailable(
                    "no API key supplied; set the environment variable named in your config"
                )
            try:
                import anthropic  # imported lazily so the core needs nothing installed
            except ModuleNotFoundError as exc:
                raise TransportUnavailable(
                    'the anthropic SDK is not installed. Run: pip install -e ".[anthropic]"'
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client.messages


def _urls_from_search(block: Any) -> list[str]:
    """Pull source urls out of a search result block.

    An error block carries a single object rather than a list of results, so the
    shape is checked before iterating.
    """
    content = getattr(block, "content", None)
    if content is None or isinstance(content, (str, bytes)) or not hasattr(content, "__iter__"):
        return []
    found: list[str] = []
    for item in content:
        url = getattr(item, "url", None)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            found.append(url)
    return found
