"""Evaluator and transport tests.

Still no key and no network: the provider is a recording fake, and the
Anthropic transport's request-building and response-parsing are pure functions
exercised against recorded payload shapes.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace as NS

from app.core.calibration import MIN_SAMPLES, seed_from_history
from app.core.rules import Rule, RuleTarget
from app.core.types import Evidence, FactBase, Rubric
from app.evaluator import (
    MalformedModelOutput,
    ModelEvaluator,
    ModelRefused,
    load_schemas,
)
from app.pipeline import Candidate
from app.providers.anthropic_transport import AnthropicTransport, TransportUnavailable
from app.providers.base import Capability, Completion

RUBRIC = Rubric(
    version=1,
    dimensions=("demand", "competition", "ease", "capital", "profit", "solo_marketing"),
    tracks=("service", "physical"),
)
FACTS = FactBase({"budget_ceiling_usd": 20_000, "location": "Miami"})
CAND = Candidate(id="c1", name="BenchmarkBridge", track="service", rationale="filing service")


class RecordingProvider:
    """Captures every request and replays queued responses."""

    name = "fake"

    def __init__(self, *responses: object) -> None:
        self.requests: list[dict] = []
        self._queue = list(responses)

    def capabilities(self):
        return frozenset({Capability.STRUCTURED_OUTPUT, Capability.SERVER_SEARCH})

    def default_model(self):
        from app.providers.base import ModelSpec

        return ModelSpec("fake", "fake-1", 1000, 0.0, 0.0)

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        payload = self._queue.pop(0)
        if isinstance(payload, Completion):
            return payload
        return Completion(
            text=json.dumps(payload),
            parsed=payload,
            citations=(),
            input_tokens=10,
            output_tokens=10,
            model_id="fake-1",
        )


def evaluator(provider, **kwargs) -> ModelEvaluator:
    return ModelEvaluator(provider=provider, rubric=RUBRIC, facts=FACTS, **kwargs)


class EvidenceDiscipline(unittest.TestCase):
    def test_an_uncited_competitor_is_discarded_not_downweighted(self) -> None:
        """The single largest measured error in the prior system was competitor
        claims with no source. They do not survive to influence a score."""
        provider = RecordingProvider(
            {
                "exhausted": False,
                "competitors": [
                    {"name": "Real", "url": "https://real.example/x",
                     "positioning": "flat fee", "verbatim": "from $1,500"},
                    {"name": "Imagined", "url": "probably exists",
                     "positioning": "similar", "verbatim": ""},
                ],
            }
        )
        batch = evaluator(provider).gather(CAND, attempt=1)

        self.assertEqual(batch.competitors, ("Real",))
        self.assertEqual(len(batch.evidence), 1)
        self.assertTrue(batch.evidence[0].url.startswith("https://"))

    def test_the_evidence_stage_binds_search(self) -> None:
        provider = RecordingProvider({"exhausted": True, "competitors": []})
        evaluator(provider).gather(CAND, attempt=1)
        self.assertTrue(provider.requests[0]["search"])

    def test_the_generate_stage_does_not_search(self) -> None:
        provider = RecordingProvider({"candidates": []})
        evaluator(provider).propose("energy compliance", 2, prior="")
        self.assertFalse(provider.requests[0]["search"])


class FreshContext(unittest.TestCase):
    def test_the_refuter_never_sees_the_generation_prefix(self) -> None:
        """Sharing context would let the generator's framing anchor its own
        referee, which is how a single-pass review talks itself into a score."""
        provider = RecordingProvider({"refuted": True, "basis": "Levelset owns this", "evidence_ids": []})
        evaluator(provider).refute(CAND, [])

        system = provider.requests[0]["system"]
        self.assertIn("not seen how it was generated", system)
        self.assertNotIn("Operator facts", system)
        self.assertFalse(provider.requests[0]["cache_prefix"])

    def test_the_refuter_is_told_to_default_to_refuted(self) -> None:
        provider = RecordingProvider({"refuted": False, "basis": "none", "evidence_ids": []})
        evaluator(provider).refute(CAND, [])
        self.assertIn("uncertain, answer refuted=true", provider.requests[0]["prompt"])


class FrozenPrefix(unittest.TestCase):
    def test_the_prefix_is_stable_across_candidates(self) -> None:
        """A prefix that varies per candidate cannot be cached, which is the
        difference between one cache write and ninety."""
        ev = evaluator(RecordingProvider())
        self.assertEqual(ev.frozen_prefix("generate"), ev.frozen_prefix("generate"))

    def test_the_calibration_prior_reaches_generate_and_evidence_only(self) -> None:
        store = seed_from_history(
            [{"dimension": "competition", "first_pass": 8, "verified": 3}] * MIN_SAMPLES
        )
        ev = evaluator(RecordingProvider(), calibration=store)
        self.assertIn("optimistic", ev.frozen_prefix("generate"))
        self.assertIn("optimistic", ev.frozen_prefix("evidence"))
        self.assertNotIn("optimistic", ev.frozen_prefix("render"))

    def test_neural_rules_land_in_the_stage_they_were_placed_at(self) -> None:
        rule = Rule(
            id="r1", description="prefer licence moats", target=RuleTarget.NEURAL,
            stage="generate", fragment="Prefer moats that are a licence or a logistics relationship.",
        )
        ev = evaluator(RecordingProvider(), rules=[rule])
        self.assertIn("logistics relationship", ev.frozen_prefix("generate"))
        self.assertNotIn("logistics relationship", ev.frozen_prefix("refute"))

    def test_the_model_is_told_it_never_scores(self) -> None:
        text = evaluator(RecordingProvider()).frozen_prefix("evidence")
        self.assertIn("never assign a total score", text)
        self.assertIn("computed outside this conversation", text)


class Failures(unittest.TestCase):
    def test_a_refusal_is_raised_not_swallowed(self) -> None:
        refusal = Completion(
            text="", parsed=None, citations=(), input_tokens=0, output_tokens=0,
            model_id="fake-1", refused=True, refusal_reason="declined (cyber)",
        )
        with self.assertRaises(ModelRefused):
            evaluator(RecordingProvider(refusal)).propose("x", 1, "")

    def test_a_missing_dimension_is_a_hard_failure(self) -> None:
        """A partial scorecard silently becomes a wrong total, so it is refused."""
        provider = RecordingProvider(
            {"dimensions": [
                {"dimension": "demand", "value": 8, "confidence": 0.6,
                 "falsifier": "x", "evidence_ids": []}
            ]}
        )
        with self.assertRaises(MalformedModelOutput) as ctx:
            evaluator(provider).judge(CAND, [])
        self.assertIn("competition", str(ctx.exception))

    def test_unparseable_output_fails_rather_than_being_salvaged(self) -> None:
        broken = Completion(
            text="not json", parsed=None, citations=(), input_tokens=0,
            output_tokens=0, model_id="fake-1",
        )
        with self.assertRaises(MalformedModelOutput):
            evaluator(RecordingProvider(broken)).propose("x", 1, "")


class Schemas(unittest.TestCase):
    def test_every_call_site_has_a_schema(self) -> None:
        schemas = load_schemas()
        for name in ("propose", "gather", "judge", "refute", "compile_rule"):
            self.assertIn(name, schemas)

    def test_schemas_are_closed_so_strict_mode_can_validate(self) -> None:
        for name, schema in load_schemas().items():
            self.assertFalse(
                schema.get("additionalProperties", True), f"{name} is not closed"
            )
            self.assertIn("required", schema, f"{name} lists no required fields")

    def test_the_judge_schema_cannot_express_a_total(self) -> None:
        """The model must have no way to hand back a number we would trust."""
        props = load_schemas()["judge"]["properties"]["dimensions"]["items"]["properties"]
        self.assertNotIn("total", props)
        self.assertIn("falsifier", props)
        self.assertIn("confidence", props)


class Transport(unittest.TestCase):
    def setUp(self) -> None:
        self.t = AnthropicTransport(api_key="unused-in-these-tests")

    def test_the_system_prompt_carries_the_cache_breakpoint(self) -> None:
        req = self.t.build_request(
            system="frozen", prompt="p", schema=None, search=False, cache_prefix=True
        )
        self.assertEqual(req["system"][0]["cache_control"], {"type": "ephemeral"})

    def test_a_fresh_context_is_not_cached(self) -> None:
        req = self.t.build_request(
            system="s", prompt="p", schema=None, search=False, cache_prefix=False
        )
        self.assertNotIn("cache_control", req["system"][0])

    def test_search_binds_the_server_side_tool(self) -> None:
        req = self.t.build_request(
            system="s", prompt="p", schema=None, search=True, cache_prefix=True
        )
        self.assertEqual(req["tools"][0]["name"], "web_search")

    def test_no_tools_are_declared_when_search_is_off(self) -> None:
        req = self.t.build_request(
            system="s", prompt="p", schema=None, search=False, cache_prefix=True
        )
        self.assertNotIn("tools", req)

    def test_a_schema_is_sent_as_an_output_format(self) -> None:
        req = self.t.build_request(
            system="s", prompt="p", schema={"type": "object"}, search=False, cache_prefix=True
        )
        self.assertEqual(req["output_config"]["format"]["type"], "json_schema")

    def test_a_refusal_is_detected_before_content_is_read(self) -> None:
        """A declined request returns 200 with empty content; indexing content[0]
        would raise on exactly the requests that were refused."""
        response = NS(
            stop_reason="refusal",
            stop_details=NS(category="cyber"),
            content=[],
            model="claude-opus-5",
            usage=NS(input_tokens=5, output_tokens=0),
        )
        result = AnthropicTransport.parse(response, expects_json=True)
        self.assertTrue(result.refused)
        self.assertIn("cyber", result.refusal_reason)

    def test_search_urls_are_extracted_for_the_evidence_stage(self) -> None:
        response = NS(
            stop_reason="end_turn",
            content=[
                NS(type="web_search_tool_result",
                   content=[NS(url="https://a.example/1"), NS(url="https://b.example/2")]),
                NS(type="text", text='{"ok": true}'),
            ],
            model="claude-opus-5",
            usage=NS(input_tokens=10, output_tokens=4),
        )
        result = AnthropicTransport.parse(response, expects_json=True)
        self.assertEqual(result.citations, ("https://a.example/1", "https://b.example/2"))
        self.assertEqual(result.parsed, {"ok": True})

    def test_a_search_error_block_does_not_crash_the_parser(self) -> None:
        """Search failures arrive as a single error object, not a list."""
        response = NS(
            stop_reason="end_turn",
            content=[
                NS(type="web_search_tool_result", content=NS(error_code="max_uses_exceeded")),
                NS(type="text", text="{}"),
            ],
            model="claude-opus-5",
            usage=NS(input_tokens=1, output_tokens=1),
        )
        self.assertEqual(AnthropicTransport.parse(response, expects_json=True).citations, ())

    def test_a_missing_key_is_reported_before_any_network_call(self) -> None:
        with self.assertRaises(TransportUnavailable):
            AnthropicTransport(api_key="").complete(system="s", prompt="p")


if __name__ == "__main__":
    unittest.main()
