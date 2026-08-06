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
    if provider not in PROBES:
        return Probe(False, "No provider chosen", fix="Pick a provider first.")
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
