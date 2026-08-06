"""Anthropic chat with tools, presented as the OpenAI-shaped client.

``ChatAgent`` was written against one client interface: ``chat(messages, tools,
on_text, on_reasoning) -> Reply``. Anthropic does not speak that dialect, so
selecting Anthropic left the agent with no transport at all and every message
fell through to the keyword grammar - the same class of failure as the
evaluator being wired only to OpenAI-compatible providers.

Rather than teach the agent two shapes, the translation lives here. The
conversation is kept in OpenAI form and converted per request, which keeps the
history one format and the difference in one place.

The two formats disagree in exactly three ways:

  * the system prompt is a message in one and a top-level field in the other;
  * a tool call is an assistant field in one and a content block in the other;
  * a tool result is a ``tool`` role in one and a *user* content block in the
    other, and consecutive results must be merged into a single user turn.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .anthropic_transport import API_URL, API_VERSION
from .openai_compat import ChatError, Reply, ToolCall

MAX_TOKENS = 8_000


@dataclass(slots=True)
class AnthropicChatClient:
    """Duck-typed replacement for OpenAICompatClient, Anthropic underneath."""

    provider: str
    api_key: str
    model: str
    timeout: int = 180
    on_retry: Callable[[str], None] | None = None

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Reply:
        system, turns = split_system(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": turns,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [to_anthropic_tool(t) for t in tools]

        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            raise ChatError(_explain(exc)) from None
        except urllib.error.URLError as exc:
            raise ChatError(f"could not reach anthropic: {exc.reason}") from None

        reply = Reply(model=str(body.get("model") or self.model))
        reply.finish_reason = str(body.get("stop_reason") or "")
        for i, block in enumerate(body.get("content") or []):
            kind = block.get("type")
            if kind == "text":
                reply.text += block.get("text", "")
            elif kind == "thinking" and on_reasoning:
                on_reasoning(block.get("thinking", ""))
            elif kind == "tool_use":
                reply.tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or f"call_{i}"),
                        name=str(block.get("name") or ""),
                        arguments=json.dumps(block.get("input") or {}),
                    )
                )
        # No streaming here, so the whole answer arrives at once. It is still
        # handed to on_text so the page renders it the same way either way.
        if reply.text and on_text:
            on_text(reply.text)
        return reply


def split_system(
    messages: Sequence[Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Pull system messages out, and convert the rest to Anthropic turns."""
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_parts.append(str(msg.get("content") or ""))
            continue

        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": str(msg.get("tool_call_id") or ""),
                "content": str(msg.get("content") or ""),
            }
            # Anthropic rejects two user turns in a row, and a round of parallel
            # tool calls produces one `tool` message per call. They have to
            # arrive as several blocks inside a single user turn.
            if turns and turns[-1]["role"] == "user" and isinstance(turns[-1]["content"], list):
                turns[-1]["content"].append(block)
            else:
                turns.append({"role": "user", "content": [block]})
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": str(msg["content"])})
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": str(call.get("id") or ""),
                    "name": str(fn.get("name") or ""),
                    "input": args,
                })
            # An assistant turn with no content at all is rejected. It happens
            # when a model answers with tool calls and no prose.
            turns.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})
            continue

        turns.append({"role": "user", "content": str(msg.get("content") or "")})

    return "\n\n".join(p for p in system_parts if p), turns


def to_anthropic_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """OpenAI's nested function envelope, flattened."""
    fn = tool.get("function") or tool
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _explain(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8", "replace"))
        message = (payload.get("error") or {}).get("message") or str(payload)[:200]
    except Exception:
        message = exc.reason or ""
    if exc.code in (401, 403):
        return "anthropic rejected the key. Re-check it in Setup."
    if exc.code == 429:
        return "anthropic is rate-limiting this request. Wait a moment and retry."
    if exc.code >= 500:
        return f"anthropic returned {exc.code}. Nothing is wrong on your side."
    return f"anthropic returned {exc.code}: {message}"
