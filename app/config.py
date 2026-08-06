"""Configuration, key resolution, and the first-run state machine.

Two rules shape this module.

The operator's key never enters the repository and never enters the config
file. The config stores a *reference* — which environment variable to read, or
that the key lives in the OS keyring — so a config file accidentally committed
leaks a variable name and nothing else.

The key is never printed, logged, or returned in any report. ``describe()``
reports where a key came from and how it ends, which is what a person needs to
tell two keys apart, and nothing more.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Mapping

APP_DIR_NAME = ".idea-ledger"
CONFIG_FILE = "config.json"

#: Conventional environment variable per provider, used when the operator has
#: not named one. Matching the provider's own documented variable means an
#: existing shell environment usually just works.
DEFAULT_ENV_VARS: Mapping[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


#: What each provider's keys actually look like. Checked before storing, because
#: an unchecked field will happily accept anything: a 302-character browser
#: console traceback was once saved as a key, and the only symptom was eight
#: probe cards all reporting "rejected the key", which points at the account
#: rather than at the paste. Refusing at the door names the real problem.
#: Prefixes are *recognition*, not permission. Providers add formats without
#: warning - Google alone has issued more than one - so a prefix this table has
#: never seen is accepted with a note rather than refused. Refusing on an
#: unknown prefix would mean a stale table in this file could block a perfectly
#: valid key, which is a worse failure than the paste it was meant to catch.
KEY_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "anthropic": ("sk-ant-",),
    "google": ("AIza", "AQ."),
    "openai": ("sk-",),
    "openrouter": ("sk-or-",),
}
MIN_KEY_LEN, MAX_KEY_LEN = 20, 250


def _claimed_by(key: str) -> list[str]:
    """Providers whose known prefixes this key matches.

    ``sk-`` is excluded because it prefixes almost every provider's keys and so
    identifies nothing.
    """
    return [
        provider
        for provider, prefixes in KEY_PREFIXES.items()
        for prefix in prefixes
        if prefix != "sk-" and key.startswith(prefix)
    ]


def key_advisory(provider: str, key: str) -> str:
    """A note about an accepted key, or "" when it looks entirely expected.

    Separate from ``validate_key`` because this never blocks anything. It exists
    so an unrecognised format is visible without being fatal.
    """
    known = KEY_PREFIXES.get(provider)
    if not known or not key or any(key.startswith(p) for p in known):
        return ""
    return (
        f"Saved. Note that {provider} keys usually start "
        f"{' or '.join(repr(p) for p in known)}, and this one starts "
        f"{key[:4]!r}. That is fine if it is what your provider issued - formats "
        "change - but it is worth a glance if the connection test fails."
    )


def validate_key(provider: str, raw: str) -> tuple[str, str]:
    """Return ``(cleaned_key, complaint)``. A non-empty complaint means refuse.

    Structural checks come first and are phrased around what the operator most
    likely did, since the realistic failures are a stray paste, a truncated
    copy, or a key belonging to a different provider.
    """
    key = (raw or "").strip()
    if not key:
        return "", "Paste a key first - the field is empty."
    if any(c.isspace() for c in key):
        head = key.split()[0][:12]
        return "", (
            f"That is not a key: it contains spaces or line breaks, and starts "
            f"'{head}...'. It looks like text pasted from somewhere else. Copy the "
            "key on its own."
        )
    if any(ord(c) < 32 or ord(c) == 127 for c in key):
        return "", "That text contains control characters, so it is not a key."
    if len(key) < MIN_KEY_LEN:
        return "", (
            f"That is only {len(key)} characters. Keys are longer than "
            f"{MIN_KEY_LEN}; the copy was probably cut short."
        )
    if len(key) > MAX_KEY_LEN:
        return "", (
            f"That is {len(key)} characters, far longer than any key. Something "
            "other than a key was pasted."
        )
    # Only a *positive* match against another provider is grounds to refuse. An
    # unrecognised prefix is accepted, because this table cannot be trusted to
    # be current and blocking a valid key is the worse error. The soft case is
    # reported by key_advisory().
    mine = KEY_PREFIXES.get(provider, ())
    if not any(key.startswith(p) for p in mine):
        elsewhere = [p for p in _claimed_by(key) if p != provider]
        if elsewhere:
            article = "an" if elsewhere[0][0] in "aeiou" else "a"
            return "", (
                f"That looks like {article} {elsewhere[0]} key, not "
                f"{'an' if provider[0] in 'aeiou' else 'a'} {provider} one. "
                "Switch the provider above, or paste the matching key."
            )
    return key, ""


class KeySource(str, Enum):
    ENVIRONMENT = "environment"
    KEYRING = "keyring"
    ABSENT = "absent"


class SetupStep(str, Enum):
    CHOOSE_PROVIDER = "choose_provider"
    NAME_ROUTE = "name_route"          # openrouter only
    SUPPLY_KEY = "supply_key"
    REVIEW_CAPABILITIES = "review_capabilities"
    EDIT_FACT_BASE = "edit_fact_base"
    READY = "ready"


def user_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where operator data lives. Never inside the repository.

    Honours an explicit override so a second profile, or a test, never touches
    the real ledger.
    """
    env = os.environ if env is None else env
    override = env.get("IDEA_LEDGER_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / APP_DIR_NAME


@dataclass(frozen=True, slots=True)
class KeyRef:
    """A pointer to a key. Never the key.

    Resolution order is environment first, then the OS secret store. Env wins so
    a shell variable can override a stored key for one session without anyone
    having to delete anything.
    """

    source: KeySource = KeySource.ENVIRONMENT
    env_var: str = ""
    account: str = ""   # the record's name in the OS store; never the secret

    def resolve(
        self, env: Mapping[str, str] | None = None, home: "Path | None" = None
    ) -> str | None:
        env = os.environ if env is None else env
        if self.env_var:
            value = env.get(self.env_var, "").strip()
            if value:
                return value
        if self.account:
            from . import secrets as secret_store

            return secret_store.fetch(home or user_dir(env), self.account)
        return None

    def found_in(
        self, env: Mapping[str, str] | None = None, home: "Path | None" = None
    ) -> KeySource:
        env = os.environ if env is None else env
        if self.env_var and env.get(self.env_var, "").strip():
            return KeySource.ENVIRONMENT
        if self.resolve(env, home):
            return KeySource.KEYRING
        return KeySource.ABSENT


@dataclass(frozen=True, slots=True)
class Config:
    provider: str = ""
    model_id: str = ""             # required for openrouter; defaulted otherwise
    key_ref: KeyRef = field(default_factory=KeyRef)
    search_compensator: str = ""   # e.g. "tavily", when the provider cannot search
    fact_base_edited: bool = False

    # -- persistence -----------------------------------------------------
    @classmethod
    def load(cls, home: Path | None = None) -> "Config":
        path = (home or user_dir()) / CONFIG_FILE
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        key = raw.get("key_ref", {})
        return cls(
            provider=raw.get("provider", ""),
            model_id=raw.get("model_id", ""),
            key_ref=KeyRef(
                source=KeySource(key.get("source", KeySource.ENVIRONMENT.value)),
                env_var=key.get("env_var", ""),
                account=key.get("account", ""),
            ),
            search_compensator=raw.get("search_compensator", ""),
            fact_base_edited=bool(raw.get("fact_base_edited", False)),
        )

    def save(self, home: Path | None = None) -> Path:
        base = home or user_dir()
        base.mkdir(parents=True, exist_ok=True)
        path = base / CONFIG_FILE
        payload = {
            "provider": self.provider,
            "model_id": self.model_id,
            "key_ref": {
                "source": self.key_ref.source.value,
                "env_var": self.key_ref.env_var,
                "account": self.key_ref.account,   # a record name, never a secret
            },
            "search_compensator": self.search_compensator,
            "fact_base_edited": self.fact_base_edited,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    # -- key handling ----------------------------------------------------
    def with_provider(self, provider: str) -> "Config":
        """Selecting a provider switches to that provider's usual variable.

        This must *replace* the previous provider's variable, not fall back to
        it. Keeping the old one is why switching to OpenRouter still reported
        ANTHROPIC_API_KEY, and why the key never appeared to resolve.
        """
        env_var = DEFAULT_ENV_VARS.get(provider, self.key_ref.env_var)
        return replace(
            self,
            provider=provider,
            key_ref=replace(self.key_ref, env_var=env_var, account=provider),
        )

    def key(self, env: Mapping[str, str] | None = None, home: Path | None = None) -> str | None:
        return self.key_ref.resolve(env, home)

    def key_status(
        self, env: Mapping[str, str] | None = None, home: Path | None = None
    ) -> tuple[KeySource, str]:
        """(source, human description). The key itself is never in the result."""
        value = self.key_ref.resolve(env, home)
        if value is None:
            var = self.key_ref.env_var or "(not chosen)"
            return KeySource.ABSENT, f"no key stored, and {var} is not set in this shell"
        source = self.key_ref.found_in(env, home)
        where = (
            f"environment variable {self.key_ref.env_var}"
            if source is KeySource.ENVIRONMENT
            else "your operating system's secret store"
        )
        return source, f"found in {where}, ending {_tail(value)}"

    # -- first-run state machine -----------------------------------------
    def next_step(
        self, env: Mapping[str, str] | None = None, home: Path | None = None
    ) -> SetupStep:
        if not self.provider:
            return SetupStep.CHOOSE_PROVIDER
        # The key comes before the route, not after. Listing a broker's usable
        # routes requires an authenticated call, so asking for a route first
        # asks for something the operator has no way to look up yet.
        if self.key(env, home) is None:
            return SetupStep.SUPPLY_KEY
        if self.provider == "openrouter" and not self.model_id:
            return SetupStep.NAME_ROUTE
        if not self.fact_base_edited:
            return SetupStep.EDIT_FACT_BASE
        return SetupStep.READY

    def describe(
        self, env: Mapping[str, str] | None = None, home: Path | None = None
    ) -> str:
        """The setup screen's status block. Safe to show on screen or paste."""
        source, detail = self.key_status(env, home)
        lines = [
            f"provider   {self.provider or '(not chosen)'}",
            f"model      {self.model_id or '(provider default)'}",
            f"key        {detail}",
        ]
        if self.search_compensator:
            lines.append(f"search     supplied by {self.search_compensator}")
        lines.append(f"next step  {self.next_step(env, home).value}")
        if source is KeySource.ABSENT and self.key_ref.env_var:
            lines += [
                "",
                "To supply the key without writing it anywhere:",
                f"  PowerShell   $env:{self.key_ref.env_var} = 'your-key'",
                f"  bash / zsh   export {self.key_ref.env_var}='your-key'",
            ]
        return "\n".join(lines)


def _tail(secret: str, keep: int = 4) -> str:
    """Last few characters, enough to tell two keys apart and nothing more."""
    return "..." + secret[-keep:] if len(secret) > keep else "(too short to display)"
