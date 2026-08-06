"""The conversational layer.

The model orchestrates the *conversation*. It does not orchestrate the
pipeline. It can decide to call ``run_pipeline``, and once it does, control
flow inside that pipeline stays exactly as deterministic as it was: the model
cannot skip a gate, cannot decide it has searched enough, and never sees a
score. The thesis holds — this just puts a person's own words on the front of
it instead of a command grammar.

Every tool here is an existing, tested operation. Nothing new is reachable
through chat that was not already reachable through the panels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

MAX_ROUNDS = 6   # bounded so a confused model cannot loop forever


SYSTEM = """You are the assistant inside Idea Ledger, a local tool that evaluates \
business ideas for one operator and keeps their scores honest as facts change.

How this system works, so you describe it accurately:
- Ideas run through a fixed pipeline: load, gate, gather evidence, refute, score.
- Three of those stages are plain code. Gates and scoring never consult a model.
- You never assign or guess a score. Scores are computed from evidence by code.
- An idea with too few citable competitors is marked under-researched and is not \
scored at all, rather than scored on thin evidence.
- Rules are either symbolic (a predicate over the operator's fields) or neural (a \
prompt fragment). Rules are data; they are never executed as code.

How to behave:
- Use the tools for anything factual about this ledger. Never invent a rule, an \
idea, a score, or a fact — read it.
- Be brief and concrete. Lead with the answer.
- If the operator asks for something this build cannot do, say so plainly and say \
what it can do instead. Do not pretend.
- When a run rejects candidates, say which rule rejected them and note that it \
happened in code, before any model judged the idea.
- If run_pipeline returns a demo_reason, lead with it. Those candidates are \
fictional; reporting them as findings would be the worst thing you could do here. \
Say plainly that they are demo data, give the reason verbatim, and only then \
describe what the run showed about the machinery."""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": (
                "Evaluate candidate ideas against the operator's rules. Runs the "
                "full gate/evidence/refute/score pipeline and returns each "
                "candidate's outcome and the reason for it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "brief": {
                        "type": "string",
                        "description": "What kind of business to evaluate, in a few words.",
                    }
                },
                "required": ["brief"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_rules",
            "description": "Every rule loaded in this ledger, with its stage, target, author and whether it is active.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_facts",
            "description": "The operator's fact base: budget, constraints, licences, preferences.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_ideas",
            "description": "Ideas already in the ledger with status and score.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_idea",
            "description": "One idea in full: every dimension with its value, confidence, falsifier and provenance.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_connection",
            "description": "Verify the stored key actually connects to the provider.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_rule",
            "description": (
                "Dry-run a proposed symbolic rule against the ledger and report its "
                "blast radius. Nothing is activated; activation is a separate, "
                "explicit act by the operator."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "field": {"type": "string", "description": "a fact-base or idea field"},
                    "op": {
                        "type": "string",
                        "enum": ["lte", "gte", "lt", "gt", "eq", "ne", "falsy", "truthy"],
                    },
                    "value": {"type": "string", "description": "omit for falsy/truthy"},
                },
                "required": ["description", "field", "op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_rubric",
            "description": (
                "The scoring rubric as configured: dimensions, thresholds, tracks, "
                "and the pipeline manifest. Read this before explaining how scoring "
                "works, so the explanation matches this ledger rather than a general "
                "description."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_rule",
            "description": (
                "Activate a rule the operator has just previewed and explicitly "
                "agreed to. Only call this after preview_rule, and only when the "
                "operator has said yes in their own words. Never call it "
                "speculatively."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string"},
                    "breaking_confirmed": {"type": "boolean"},
                },
                "required": ["rule_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_status",
            "description": "Provider, key state, next setup step and where the ledger lives.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass(slots=True)
class ChatAgent:
    client: Any                                   # OpenAICompatClient
    tools: Mapping[str, Callable[..., Any]]
    emit_text: Callable[[str], None]
    emit_delta: Callable[[str], None]
    emit_tool: Callable[[str, str, str], None]    # name, state, detail
    emit_think: Callable[[str], None] = lambda _t: None
    history: list[dict[str, Any]] = field(default_factory=list)

    def ask(self, message: str) -> dict[str, Any]:
        if not self.history:
            self.history.append({"role": "system", "content": SYSTEM})
        self.history.append({"role": "user", "content": message})

        for _ in range(MAX_ROUNDS):
            reply = self.client.chat(
                self.history,
                tools=TOOLS,
                on_text=self.emit_delta,
                on_reasoning=self.emit_think,
            )

            assistant: dict[str, Any] = {"role": "assistant", "content": reply.text or None}
            if reply.reasoning_details:
                # Echoed back exactly as received. The provider treats this as
                # opaque state; editing or summarising it breaks the model's
                # ability to continue its own reasoning on the next call.
                assistant["reasoning_details"] = reply.reasoning_details
            if reply.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": c.id or f"call_{i}",
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments or "{}"},
                        # Echoed back exactly as received, for the same reason as
                        # reasoning_details above. Gemini 3.x attaches a
                        # thought_signature here and rejects the next turn
                        # outright without it, so dropping this does not degrade
                        # the reply - it ends the conversation with a 400.
                        **({"extra_content": c.extra} if c.extra else {}),
                    }
                    for i, c in enumerate(reply.tool_calls)
                ]
            self.history.append(assistant)

            if not reply.tool_calls:
                if reply.text:
                    self.emit_text(reply.text)
                return {"ok": True, "text": reply.text}

            for i, call in enumerate(reply.tool_calls):
                fn = self.tools.get(call.name)
                args = call.parsed()
                detail = ", ".join(f"{k}={v}" for k, v in args.items())[:80]
                self.emit_tool(call.name, "started", detail)
                if fn is None:
                    result: Any = {"error": f"no tool named {call.name}"}
                    self.emit_tool(call.name, "failed", "unknown tool")
                else:
                    try:
                        result = fn(**args)
                        self.emit_tool(call.name, "done", _summarise(call.name, result))
                    except Exception as exc:
                        # The model reads this and relays it, so it has to be a
                        # sentence rather than a class name. "no credit" needs
                        # to reach the operator as "no credit".
                        from .web import _explain_failure

                        message, fix = _explain_failure(exc)
                        result = {"error": message, "fix": fix}
                        self.emit_tool(call.name, "failed", message[:90])
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or f"call_{i}",
                        "name": call.name,
                        "content": json.dumps(result, default=str)[:12000],
                    }
                )

        self.emit_text(
            "I kept reaching for tools without settling on an answer, so I stopped "
            "rather than loop. Try asking for one thing at a time."
        )
        return {"ok": False, "text": "round limit reached"}


def _summarise(name: str, result: Any) -> str:
    if isinstance(result, dict):
        if "results" in result:
            rows = result["results"]
            scored = sum(1 for r in rows if r.get("outcome") == "scored")
            return f"{len(rows)} candidate(s), {scored} scored"
        if "rules" in result:
            return f"{len(result['rules'])} rule(s)"
        if "ideas" in result:
            return f"{len(result['ideas'])} idea(s)"
        if "headline" in result:
            return str(result["headline"])
        if "error" in result:
            return str(result["error"])[:90]
    if isinstance(result, list):
        return f"{len(result)} row(s)"
    return ""
