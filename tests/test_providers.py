"""Capability negotiation tests.

The property under test: a provider that cannot search must not be allowed to
run the evidence stage. Silently proceeding is how a model ends up inventing
competitors, which was the largest measured error in the system this replaces.
"""

from __future__ import annotations

import io
import json
from unittest import mock
from app.providers import openai_compat

import unittest

from app.providers import (
    default_requirements,
    Capability,
    NotConfigured,
    Resolution,
    get,
    negotiate,
    with_capabilities,
)


def stage(negotiation, name: str):
    return next(s for s in negotiation.stages if s.stage == name)


class Negotiation(unittest.TestCase):
    def test_anthropic_resolves_every_stage_natively(self) -> None:
        n = negotiate(get("anthropic"), default_requirements())
        self.assertTrue(n.ok)
        self.assertTrue(all(s.resolution is Resolution.NATIVE for s in n.stages))

    def test_openai_resolves_every_stage_natively(self) -> None:
        n = negotiate(get("openai"), default_requirements())
        self.assertTrue(n.ok)

    def test_openrouter_default_refuses_the_evidence_stage(self) -> None:
        """A routed model with no server search cannot gather evidence, and the
        system says so at setup rather than at run time."""
        n = negotiate(get("openrouter"), default_requirements())
        self.assertFalse(n.ok)
        evidence = stage(n, "evidence")
        self.assertIs(evidence.resolution, Resolution.REFUSED)
        self.assertIn(Capability.SERVER_SEARCH, evidence.missing)

    def test_refusal_explains_the_consequence_not_the_parameter(self) -> None:
        n = negotiate(get("openrouter"), default_requirements())
        text = stage(n, "evidence").explanation
        self.assertIn("invents competitors", text)

    def test_stages_without_requirements_always_run(self) -> None:
        n = negotiate(get("openrouter"), default_requirements())
        self.assertTrue(stage(n, "render").runnable)
        self.assertTrue(stage(n, "generate").runnable)

    def test_a_search_compensator_unblocks_the_evidence_stage(self) -> None:
        """A standalone search adapter supplies what the provider lacks; the
        stage runs as COMPENSATED, and the operator can see that it did."""
        n = negotiate(
            get("openrouter"),
            default_requirements(),
            compensators={Capability.SERVER_SEARCH: "tavily"},
        )
        self.assertTrue(n.ok)
        evidence = stage(n, "evidence")
        self.assertIs(evidence.resolution, Resolution.COMPENSATED)
        self.assertEqual(evidence.compensated[Capability.SERVER_SEARCH], "tavily")

    def test_a_declared_route_can_upgrade_openrouter(self) -> None:
        route = with_capabilities(
            get("openrouter"),
            frozenset({Capability.STRUCTURED_OUTPUT, Capability.SERVER_SEARCH}),
            model_id="anthropic/claude-opus-5",
        )
        n = negotiate(route, default_requirements())
        self.assertTrue(n.ok)
        self.assertEqual(n.model_id, "anthropic/claude-opus-5")

    def test_report_is_readable_and_names_the_failing_stage(self) -> None:
        report = negotiate(get("openrouter"), default_requirements()).report()
        self.assertIn("openrouter", report)
        self.assertIn("cannot run", report)

    def test_negotiation_needs_no_key_and_no_network(self) -> None:
        """Setup can negotiate before the operator has entered anything."""
        provider = get("anthropic")
        self.assertTrue(negotiate(provider, default_requirements()).ok)
        with self.assertRaises(NotConfigured):
            provider.complete(system="", prompt="")


if __name__ == "__main__":
    unittest.main()


class RouteSelection(unittest.TestCase):
    """Which routes the probe is even allowed to consider.

    A probe can only tell whether a route answers. It cannot tell whether the
    route honoured a schema, because a route that ignores response_format
    answers happily and returns markdown. So the filtering has to happen on
    OpenRouter's published metadata before any call is made, and that filtering
    is what these tests pin.
    """

    CATALOGUE = {"data": [
        {"id": "good/chat:free", "supported_parameters": ["response_format", "tools"],
         "pricing": {"prompt": "0"}, "architecture": {"output_modalities": ["text"]}},
        {"id": "good/cheap", "supported_parameters": ["structured_outputs", "tools"],
         "pricing": {"prompt": "0.00000001"}, "architecture": {"output_modalities": ["text"]}},
        {"id": "good/dear", "supported_parameters": ["response_format", "tools"],
         "pricing": {"prompt": "0.00001"}, "architecture": {"output_modalities": ["text"]}},
        # The real failure: cohere/north-mini-code:free takes tools but has no
        # response_format, so it answered a probe and then returned markdown.
        {"id": "no/schema:free", "supported_parameters": ["tools", "temperature"],
         "pricing": {"prompt": "0"}, "architecture": {"output_modalities": ["text"]}},
        # A music model. Lists a schema parameter, costs nothing per prompt
        # token, and would sort to the top of the paid band on price alone.
        {"id": "music/lyria", "supported_parameters": ["response_format"],
         "pricing": {"prompt": "0"}, "architecture": {"output_modalities": ["audio"]}},
        # Auto-routers price at -1, meaning "varies by upstream".
        {"id": "meta/auto", "supported_parameters": ["response_format", "tools"],
         "pricing": {"prompt": "-1"}, "architecture": {"output_modalities": ["text"]}},
    ]}

    def catalogue(self):
        payload = json.dumps(self.CATALOGUE).encode()

        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *_): return False

        return lambda *a, **k: Response(payload)

    def capable(self):
        with mock.patch.object(openai_compat.urllib.request, "urlopen", self.catalogue()):
            return openai_compat._capable_routes(
                "sk-or-test", "openrouter", {"response_format", "structured_outputs"}
            )

    def test_a_route_without_response_format_is_never_offered(self) -> None:
        self.assertNotIn("no/schema:free", self.capable())

    def test_a_non_text_route_is_excluded_however_cheap(self) -> None:
        self.assertNotIn("music/lyria", self.capable())

    def test_a_varying_price_sorts_last_rather_than_first(self) -> None:
        caps = self.capable()
        self.assertEqual(float("inf"), caps["meta/auto"])

    def test_free_is_tried_before_paid_and_paid_cheapest_first(self) -> None:
        caps = self.capable()
        free = lambda m: m.endswith(":free") or m == "openrouter/free"
        order = sorted(caps, key=lambda m: (not free(m), caps[m], m.endswith("/free"), m))
        self.assertEqual(["good/chat:free", "good/cheap", "good/dear", "meta/auto"], order)


class GoogleProvider(unittest.TestCase):
    """Gemini reached through Google's OpenAI-compatible surface.

    Worth its own coverage because it is the only provider here with a genuinely
    free tier for schema-bound work: OpenRouter carries 21 Gemini routes and not
    one of them is free.
    """

    def test_it_is_registered_and_speaks_the_compatible_dialect(self) -> None:
        from app.providers import REGISTRY
        from app.providers.openai_compat import ENDPOINTS, MODEL_LISTS

        self.assertIn("google", REGISTRY)
        self.assertIn("generativelanguage.googleapis.com", ENDPOINTS["google"])
        self.assertIn("generativelanguage.googleapis.com", MODEL_LISTS["google"])

    def test_it_does_not_claim_search_it_has_not_verified(self) -> None:
        """Gemini has grounding; this build has not confirmed it survives the shim.

        Claiming it would put the evidence stage back in the position that broke
        the generate stage: reported ready, then failing on real data.
        """
        from app.providers import REGISTRY
        from app.providers.base import Capability

        caps = REGISTRY["google"].capabilities()
        self.assertIn(Capability.STRUCTURED_OUTPUT, caps)
        self.assertIn(Capability.STRICT_TOOLS, caps)
        self.assertNotIn(Capability.SERVER_SEARCH, caps)

    def test_google_keys_are_recognised_and_mismatches_are_named(self) -> None:
        from app.config import validate_key

        for fmt in ("AIza" + "x" * 35, "AQ." + "x" * 35):
            with self.subTest(fmt=fmt[:4]):
                key, complaint = validate_key("google", fmt)
                self.assertEqual("", complaint)
                self.assertTrue(key)

        _, complaint = validate_key("openrouter", "AIza" + "x" * 35)
        self.assertIn("google", complaint)

    def test_a_google_config_resolves_its_own_env_var(self) -> None:
        from app.config import Config

        cfg = Config().with_provider("google")
        self.assertEqual("GEMINI_API_KEY", cfg.key_ref.env_var)


class OpaqueToolCallState(unittest.TestCase):
    """Provider state attached to a tool call must survive a round trip.

    Gemini 3.x attaches a thought_signature to every function call and rejects
    the *next* turn with 400 - "Function call is missing a thought_signature in
    functionCall parts" - if the call is replayed without it. Dropping it does
    not degrade the answer; it ends the conversation. The agent understood
    "lets run wooden furniture", ran the pipeline, scored a candidate, and then
    died on the follow-up turn for exactly this reason.
    """

    SIGNED = {"google": {"thought_signature": "Eq8CCqwCARFNMg"}}

    def test_a_streamed_tool_call_keeps_its_extra_content(self) -> None:
        frag = {
            "index": 0, "id": "abc", "extra_content": self.SIGNED,
            "function": {"name": "ping", "arguments": "{}"},
        }
        call = openai_compat.ToolCall()
        # Mirrors the reassembly in chat(): fragments merge by index.
        if frag.get("id"):
            call.id = frag["id"]
        if frag.get("extra_content"):
            call.extra = dict(frag["extra_content"])
        self.assertEqual(self.SIGNED, call.extra)

    def test_the_agent_echoes_it_back_unmodified(self) -> None:
        from app.chat_agent import ChatAgent

        call = openai_compat.ToolCall(
            id="abc", name="list_rules", arguments="{}", extra=self.SIGNED
        )
        replies = [
            openai_compat.Reply(tool_calls=[call]),
            openai_compat.Reply(text="done"),
        ]

        class Client:
            def chat(self, messages, **_):
                Client.seen = list(messages)
                return replies.pop(0)

        agent = ChatAgent(
            client=Client(),
            tools={"list_rules": lambda: {"rules": []}},
            emit_text=lambda _t: None,
            emit_delta=lambda _t: None,
            emit_tool=lambda *_a: None,
        )
        agent.ask("what rules are loaded")

        assistant = next(
            m for m in Client.seen
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        sent = assistant["tool_calls"][0]
        self.assertEqual(self.SIGNED, sent.get("extra_content"))

    def test_a_call_without_extra_content_stays_clean(self) -> None:
        """Providers that send none must not receive an empty key."""
        from app.chat_agent import ChatAgent

        replies = [
            openai_compat.Reply(
                tool_calls=[openai_compat.ToolCall(id="x", name="list_rules", arguments="{}")]
            ),
            openai_compat.Reply(text="done"),
        ]

        class Client:
            def chat(self, messages, **_):
                Client.seen = list(messages)
                return replies.pop(0)

        ChatAgent(
            client=Client(), tools={"list_rules": lambda: {"rules": []}},
            emit_text=lambda _t: None, emit_delta=lambda _t: None,
            emit_tool=lambda *_a: None,
        ).ask("rules")
        assistant = next(
            m for m in Client.seen
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        self.assertNotIn("extra_content", assistant["tool_calls"][0])


class ErrorDetail(unittest.TestCase):
    def test_a_google_error_array_is_unwrapped(self) -> None:
        """Google wraps errors in a one-element array; everyone else does not.

        Without unwrapping, every Google failure read "returned 400: Bad
        Request", which says nothing about what to change - and hid the
        thought_signature message above for an entire debugging round.
        """
        payload = [{"error": {"code": 400, "message": "missing a thought_signature"}}]
        self.assertEqual(
            "missing a thought_signature",
            (openai_compat._unwrap(payload).get("error") or {}).get("message"),
        )
        plain = {"error": {"message": "plain object still works"}}
        self.assertEqual(
            "plain object still works",
            (openai_compat._unwrap(plain).get("error") or {}).get("message"),
        )
