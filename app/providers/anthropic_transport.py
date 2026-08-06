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
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .base import Capability, Completion, ModelSpec

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
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
    timeout: int = 180
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
        response = self._send(request)
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
    def _send(self, request: Mapping[str, Any]) -> Any:
        """Post the request over urllib, like every other provider here.

        This deliberately does not use the SDK even when one is installed.
        Preferring it failed immediately on this machine - ``Messages.create()
        got an unexpected keyword argument 'output_config'`` - because the
        installed version predates the field. The wire format is stable and
        versioned by a header; an SDK pinned by whatever happens to be in the
        environment is not. Going straight to HTTP also keeps the promise that
        this app needs nothing installed.
        """
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(dict(request)).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                detail = (json.loads(detail).get("error") or {}).get("message", detail)
            except json.JSONDecodeError:
                pass
            raise TransportUnavailable(
                f"anthropic returned {exc.code}: {detail[:300]}"
            ) from None
        except urllib.error.URLError as exc:
            raise TransportUnavailable(f"could not reach anthropic: {exc.reason}") from None
        # parse() reads the response with getattr, because it was written
        # against the SDK's objects. Wrapping keeps that one tested function
        # serving both paths instead of growing a second copy.
        return _Attr(payload)

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


def _sdk_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _Attr:
    """Attribute access over a decoded JSON body.

    ``parse`` was written against the SDK's response objects and reads
    everything through ``getattr``. Rather than fork it for the urllib path,
    the body is wrapped so both paths hand it the same shape.
    """

    __slots__ = ("_d",)

    def __init__(self, data: Any) -> None:
        self._d = data if isinstance(data, dict) else {}

    def __getattr__(self, name: str) -> Any:
        value = self._d.get(name)
        if isinstance(value, dict):
            return _Attr(value)
        if isinstance(value, list):
            return [_Attr(v) if isinstance(v, dict) else v for v in value]
        return value

    def __repr__(self) -> str:   # pragma: no cover - debugging aid
        return f"_Attr({sorted(self._d)})"
