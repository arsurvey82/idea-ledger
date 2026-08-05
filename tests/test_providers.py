"""Capability negotiation tests.

The property under test: a provider that cannot search must not be allowed to
run the evidence stage. Silently proceeding is how a model ends up inventing
competitors, which was the largest measured error in the system this replaces.
"""

from __future__ import annotations

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
