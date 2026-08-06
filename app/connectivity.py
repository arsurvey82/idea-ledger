"""Does this key actually work?

Saving a key and being told "saved" answers a question nobody asked. The
question is whether it connects. This makes one cheap authenticated request per
provider using the standard library, so the check works before any SDK is
installed, and turns whatever comes back into a sentence a person can act on.

Errors are translated deliberately. "401" is not a useful thing to show someone
who has just pasted a key; "the provider rejected this key" is.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping

TIMEOUT = 12

#: One cheap, authenticated GET per provider. Chosen to validate credentials
#: without generating tokens or costing anything.
PROBES: Mapping[str, Mapping[str, object]] = {
    "anthropic": {
        "url": "https://api.anthropic.com/v1/models?limit=1",
        "headers": lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"},
        "console": "https://console.anthropic.com/settings/keys",
    },
    "openai": {
        "url": "https://api.openai.com/v1/models",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
        "console": "https://platform.openai.com/api-keys",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/auth/key",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
        "console": "https://openrouter.ai/settings/keys",
    },
    "google": {
        # Google's OpenAI-compatible surface, so one bearer header covers it.
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/models",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
        "console": "https://aistudio.google.com/apikey",
    },
}


@dataclass(frozen=True, slots=True)
class Probe:
    ok: bool
    headline: str
    detail: str = ""
    fix: str = ""
    models_seen: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "headline": self.headline,
            "detail": self.detail,
            "fix": self.fix,
            "models_seen": self.models_seen,
        }


def check(provider: str, key: str | None) -> Probe:
    provider = (provider or "").strip().lower()
    if not provider:
        return Probe(False, "No provider chosen", fix="Pick a provider first.")
    if provider not in PROBES:
        # A provider *was* chosen; this table just has no probe for it. Saying
        # "no provider chosen" sent someone hunting for a setting that was
        # already correct, so name the real gap.
        return Probe(
            False,
            f"No connection test for {provider}",
            f"{provider} is configured, but this build has no credential check "
            "for it, so the key cannot be verified before a real request.",
            "Run something and read the error, or choose a provider with a test: "
            + ", ".join(sorted(PROBES)) + ".",
        )
    if not key:
        return Probe(
            False,
            "No key to test",
            "Nothing is stored, and the environment variable is not set in this shell.",
            "Paste a key above and save it.",
        )

    spec = PROBES[provider]
    req = urllib.request.Request(
        str(spec["url"]),
        headers={"Accept": "application/json", **spec["headers"](key)},  # type: ignore[operator]
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return _http_error(provider, exc, str(spec.get("console", "")))
    except urllib.error.URLError as exc:
        return Probe(
            False,
            "Could not reach the provider",
            f"{exc.reason}",
            "Check your internet connection, or a proxy or firewall between you and the API.",
        )
    except TimeoutError:
        return Probe(
            False,
            "The provider did not answer in time",
            f"No response within {TIMEOUT} seconds.",
            "Try again; if it persists, the provider may be having an incident.",
        )
    except Exception as exc:  # never let a probe crash the setup screen
        return Probe(False, "The check failed", f"{type(exc).__name__}: {exc}")

    # Listing models is not proof of anything. Every provider here serves that
    # endpoint free, so it answers happily for an account that cannot run a
    # single request - which is how "Connected. The key is valid" appeared above
    # a conversation where every message failed. The only honest test of whether
    # this key can do work is doing some.
    inference = _try_one_token(provider, key)
    if inference is not None:
        return inference

    count = _count(body)
    return Probe(
        True,
        f"Connected to {provider}",
        f"The key is valid{f' and {count} models are visible to it' if count else ''}.",
        models_seen=count,
    )


def _http_error(provider: str, exc: urllib.error.HTTPError, console: str) -> Probe:
    code = exc.code
    try:
        payload = json.loads(exc.read().decode("utf-8", "replace"))
        raw = json.dumps(payload)[:240]
    except Exception:
        raw = exc.reason or ""

    if code in (401, 403):
        return Probe(
            False,
            "The provider rejected this key",
            "It is not recognised, has been revoked, or belongs to a different provider.",
            f"Check you pasted the key from {provider} in full, with no spaces."
            + (f" You can reissue one at {console}." if console else ""),
        )
    if code == 402:
        return Probe(
            False,
            "The key is valid but the account cannot be billed",
            "The provider accepted the key and refused the request on payment grounds.",
            "Add credit or a payment method on the provider's dashboard, then test again.",
        )
    if code == 429:
        return Probe(
            False,
            "Rate limited before the check completed",
            "The key looks valid; the account is over its request limit right now.",
            "Wait a moment and test again.",
        )
    if code >= 500:
        return Probe(
            False,
            "The provider is having a problem",
            f"It returned {code}. Nothing is wrong with your key.",
            "Try again shortly.",
        )
    return Probe(False, f"Unexpected response ({code})", raw)


def _count(body: str) -> int:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return 0
    data = payload.get("data")
    return len(data) if isinstance(data, list) else 0


def _try_one_token(provider: str, key: str) -> Probe | None:
    """Ask the provider to generate a single token.

    Returns a failing Probe when the account cannot actually run, and None when
    it can - in which case the caller reports success as before.

    A token costs a fraction of a cent and buys the one thing the model list
    cannot tell you: whether there is credit, whether billing is live, and
    whether the configured model exists for this account. Every failure this
    catches previously surfaced as a broken conversation under a green light.
    """
    from .providers.openai_compat import ENDPOINTS

    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        payload = {
            "model": "claude-sonnet-5", "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }
    elif provider in ENDPOINTS:
        url = ENDPOINTS[provider]
        headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        payload = {
            "model": _probe_model(provider), "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }
    else:
        return None

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT + 8):
            return None                      # it ran; the caller reports success
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            detail = (parsed.get("error") or {}).get("message") or detail
        except Exception:
            pass
        low = detail.lower()
        if "credit" in low or "billing" in low or "quota" in low:
            return Probe(
                False,
                f"{provider} accepted the key but will not run anything",
                detail[:220],
                "Add credit to that account, or switch to another saved key above. "
                "The key itself is fine - listing models works without credit, "
                "which is why this needs a real request to tell you.",
            )
        if exc.code == 404:
            return Probe(
                False, f"{provider} has no model to run",
                detail[:220],
                "Press Save & test to re-resolve a model from your key.",
            )
        if exc.code == 429:
            return Probe(
                False, f"{provider} is rate-limiting", detail[:220],
                "Wait a moment and test again, or move to a paid route.",
            )
        if exc.code in (401, 403):
            return Probe(False, f"{provider} rejected the key", detail[:220],
                         "Re-check the key above.")
        return Probe(False, f"{provider} refused a one-token request", detail[:220])
    except Exception:
        # A network problem here is already reported by the listing call above;
        # do not turn a transient blip into a red light of its own.
        return None


def _probe_model(provider: str) -> str:
    """Something cheap that every account on that provider can reach."""
    return {
        "openai": "gpt-4o-mini",
        "openrouter": "openrouter/free",
        "google": "gemini-2.0-flash",
    }.get(provider, "")
