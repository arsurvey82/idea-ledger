"""OpenAI-compatible chat transport, standard library only.

OpenAI and OpenRouter speak the same chat-completions dialect, so one client
covers both. It is written against urllib rather than an SDK for the same
reason as everything else here: the operator should not have to install a
package to talk to a provider they have already paid for.

Streaming is real token streaming, not a progress bar. Tool calls arrive as
fragments spread across many chunks and are reassembled by index, which is the
one genuinely fiddly part of this wire format.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

ENDPOINTS: Mapping[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
MODEL_LISTS: Mapping[str, str] = {
    "openai": "https://api.openai.com/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
}


class ChatError(RuntimeError):
    """A provider-level failure, already phrased for a person."""


@dataclass(slots=True)
class ToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""

    def parsed(self) -> dict[str, Any]:
        try:
            return json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}


@dataclass(slots=True)
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    model: str = ""
    reasoning: str = ""
    #: OpenRouter's opaque reasoning record. It must be echoed back on the
    #: assistant turn **unmodified** for the model to continue its own chain of
    #: thought on the next call. We never parse it, edit it, or render it as
    #: content - only carry it.
    reasoning_details: list[Any] = field(default_factory=list)
    #: Urls the provider's own search returned, when the route is online.
    citations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OpenAICompatClient:
    provider: str
    api_key: str
    model: str
    timeout: int = 180
    reasoning: bool = True
    retries: int = 3
    on_retry: Callable[[str], None] | None = None

    def _post(self, payload: Mapping[str, Any]):
        url = ENDPOINTS.get(self.provider)
        if not url:
            raise ChatError(f"{self.provider} has no chat endpoint in this build")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        }
        if self.provider == "openrouter":
            # OpenRouter asks callers to identify themselves; it also improves
            # routing and shows up in your usage dashboard.
            headers["HTTP-Referer"] = "https://github.com/arsurvey82/idea-ledger"
            headers["X-Title"] = "Idea Ledger"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        # Free routes rate-limit aggressively, and a 429 on the first token of a
        # conversation is not a real failure - it is a queue. Back off and retry
        # rather than making the operator retype the message.
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < self.retries:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    delay = float(retry_after) if (retry_after or "").isdigit() else 2 ** attempt
                    if self.on_retry:
                        self.on_retry(
                            f"rate-limited by {self.provider}; retrying in {delay:.0f}s "
                            f"({attempt + 1} of {self.retries})"
                        )
                    time.sleep(min(delay, 20))
                    last = exc
                    continue
                raise ChatError(_explain(exc, self.provider, self.model)) from None
            except urllib.error.URLError as exc:
                raise ChatError(f"could not reach {self.provider}: {exc.reason}") from None
        raise ChatError(_explain(last, self.provider, self.model))  # type: ignore[arg-type]

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Reply:
        """One turn. Streams text and reasoning as they arrive."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        if self.reasoning and self.provider == "openrouter":
            payload["reasoning"] = {"enabled": True}

        reply = Reply(model=self.model)
        calls: dict[int, ToolCall] = {}

        with self._post(payload) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                piece = delta.get("content")
                if piece:
                    reply.text += piece
                    if on_text:
                        on_text(piece)

                # Visible reasoning, for the operator to read as it happens.
                think = delta.get("reasoning")
                if think:
                    reply.reasoning += think
                    if on_reasoning:
                        on_reasoning(think)

                # The opaque record, kept in arrival order and never touched.
                # This is what lets the model resume its own chain of thought
                # on the next call.
                details = delta.get("reasoning_details")
                if details:
                    reply.reasoning_details.extend(
                        details if isinstance(details, list) else [details]
                    )

                for frag in delta.get("tool_calls") or []:
                    idx = frag.get("index", 0)
                    call = calls.setdefault(idx, ToolCall())
                    if frag.get("id"):
                        call.id = frag["id"]
                    fn = frag.get("function") or {}
                    if fn.get("name"):
                        call.name = fn["name"]
                    if fn.get("arguments"):
                        call.arguments += fn["arguments"]

                for ann in (delta.get("annotations") or []):
                    url = ((ann.get("url_citation") or {}).get("url")
                           if isinstance(ann, dict) else None)
                    if url and url not in reply.citations:
                        reply.citations.append(url)

                message = choice.get("message") or {}
                if message.get("reasoning_details") and not reply.reasoning_details:
                    reply.reasoning_details = list(message["reasoning_details"])
                if message.get("reasoning") and not reply.reasoning:
                    reply.reasoning = str(message["reasoning"])

                if choice.get("finish_reason"):
                    reply.finish_reason = choice["finish_reason"]

        reply.tool_calls = [calls[i] for i in sorted(calls) if calls[i].name]
        return reply

    def chat_raw(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        response_format: Mapping[str, Any] | None = None,
    ) -> Reply:
        """One non-streaming turn. Used where the caller wants a whole object.

        The pipeline consumes a complete JSON document, so streaming it would
        only add a reassembly step with nothing to show for it.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
        }
        if response_format:
            payload["response_format"] = dict(response_format)

        with self._post(payload) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        reply = Reply(
            text=message.get("content") or "",
            finish_reason=choice.get("finish_reason") or "",
            model=data.get("model") or self.model,
            reasoning=str(message.get("reasoning") or ""),
            reasoning_details=list(message.get("reasoning_details") or []),
        )
        for ann in message.get("annotations") or []:
            url = (ann.get("url_citation") or {}).get("url") if isinstance(ann, dict) else None
            if url and url not in reply.citations:
                reply.citations.append(url)
        return reply

    def models(self, limit: int = 400) -> list[str]:
        url = MODEL_LISTS.get(self.provider)
        if not url:
            return []
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.api_key}"}, method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:
            return []
        rows = payload.get("data") or []
        return sorted({str(r.get("id", "")) for r in rows if r.get("id")})[:limit]


def _routes_supporting(
    api_key: str, provider: str, wanted: set[str]
) -> frozenset[str]:
    """Route ids whose published parameter list includes any of ``wanted``.

    Empty when the manifest is unreachable, which callers read as "no filter"
    rather than "nothing qualifies".
    """
    url = MODEL_LISTS.get(provider)
    if not url:
        return frozenset()
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            rows = json.loads(resp.read().decode("utf-8", "replace")).get("data") or []
    except Exception:
        return frozenset()
    return frozenset(
        str(r.get("id", "")) for r in rows
        if wanted & set(r.get("supported_parameters") or ())
    )


def route_parameters(api_key: str, model: str, provider: str = "openrouter") -> frozenset[str]:
    """Which request parameters this specific route actually honours.

    OpenRouter publishes ``supported_parameters`` per route, and it is the only
    honest source for this: a route can accept ``response_format`` in the
    request, return 200, and reply in markdown anyway. Reading the manifest
    beats probing, because a probe cannot distinguish "ignored the schema" from
    "answered a question that happened to need no schema".

    Returns an empty set when the list cannot be fetched, so callers fall back
    to whatever they assumed rather than hard-failing on a network hiccup.
    """
    url = MODEL_LISTS.get(provider)
    if not url:
        return frozenset()
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            rows = json.loads(resp.read().decode("utf-8", "replace")).get("data") or []
    except Exception:
        return frozenset()
    base = model.replace(":online", "")   # search is a suffix, not a route
    for row in rows:
        if str(row.get("id", "")) == base:
            return frozenset(row.get("supported_parameters") or ())
    return frozenset()


def find_working_route(
    api_key: str,
    *,
    provider: str = "openrouter",
    free_only: bool = True,
    limit: int = 8,
    on_try: Callable[[str, str], None] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Probe candidate routes and return the first that actually answers.

    Needed because a route can be listed, be free, and still be unusable: an
    account-level provider allowlist rejects it, or its upstream is rate-limited.
    Neither is visible from the model list, and a meta-route like
    ``openrouter/free`` picks a different upstream every request, so it can pass
    one probe and fail the next. Only a real call settles it.

    Returns ``(winner, attempts)``; winner is "" when nothing worked.
    """
    client = OpenAICompatClient(provider, api_key, "", retries=0)
    # Structured output is what the pipeline needs and what a probe cannot see:
    # a route without it answers the probe happily and then returns markdown at
    # the generate stage. Filter on the published parameter list first, so the
    # probe only ever chooses between routes that could actually do the work.
    capable = _routes_supporting(api_key, provider, {"response_format", "structured_outputs"})
    candidates = [
        m for m in client.models()
        if (not free_only or m.endswith(":free") or m == f"{provider}/free")
        and (not capable or m in capable)
    ]
    # Deterministic single-upstream routes first; meta-routers last, because
    # they may work once and fail the next call.
    candidates.sort(key=lambda m: (m.endswith("/free"), m))

    attempts: list[tuple[str, str]] = []
    ping = [{
        "type": "function",
        "function": {"name": "ping", "description": "connectivity probe",
                     "parameters": {"type": "object", "properties": {}}},
    }]

    for model in candidates[:limit]:
        if on_try:
            on_try(model, "trying")
        probe = OpenAICompatClient(provider, api_key, model, retries=0, timeout=45)
        try:
            probe.chat([{"role": "user", "content": "ok"}], tools=ping)
            attempts.append((model, "works"))
            if on_try:
                on_try(model, "works")
            return model, attempts
        except ChatError as exc:
            attempts.append((model, str(exc)))
            if on_try:
                on_try(model, str(exc)[:90])
    return "", attempts


def _meta(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        meta = (payload.get("error") or {}).get("metadata")
        if isinstance(meta, dict):
            return meta
    return {}


def _explain(exc: urllib.error.HTTPError, provider: str, model: str) -> str:
    payload: Any = {}
    try:
        payload = json.loads(exc.read().decode("utf-8", "replace"))
        message = (payload.get("error") or {}).get("message") or json.dumps(payload)[:200]
    except Exception:
        message = exc.reason or ""

    if exc.code in (401, 403):
        return f"{provider} rejected the key. Re-check it in Setup."
    if exc.code == 402:
        return f"{provider} accepted the key but the account has no credit."
    if exc.code == 404:
        # OpenRouter reuses 404 for a provider-allowlist mismatch, which reads
        # as "no such model" but means something quite different and fixable.
        meta = _meta(payload)
        avail, want = meta.get("available_providers"), meta.get("requested_providers")
        if avail and want:
            return (
                f"Your {provider} account only allows upstream providers "
                f"{', '.join(want)}, and this route is served by "
                f"{', '.join(avail)}. Either widen the allowed providers in your "
                f"{provider} account settings, or choose a route one of your "
                "allowed providers serves."
            )
        return f"{provider} has no route called '{model}'. Pick a different one in Setup."
    if exc.code == 429:
        raw = str(_meta(payload).get("raw") or "")
        # The upstream note ends in a URL; cutting it mid-string produced
        # "accumulate your rate limits: h", which reads as corruption.
        note = raw.split(". Please retry")[0].split(", or add")[0].strip()
        return (
            f"{provider} is rate-limiting this request"
            + (f" - {note}" if note else "")
            + ". Free routes queue behind every other user of that route. Retry, "
            "widen your account's allowed providers, or use a paid route."
        )
    if exc.code >= 500:
        return f"{provider} returned {exc.code}. Nothing is wrong on your side."
    return f"{provider} returned {exc.code}: {message}"
