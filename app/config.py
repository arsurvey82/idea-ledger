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
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


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
    """A pointer to a key. Never the key."""

    source: KeySource = KeySource.ENVIRONMENT
    env_var: str = ""

    def resolve(self, env: Mapping[str, str] | None = None) -> str | None:
        if self.source is not KeySource.ENVIRONMENT:
            return None  # keyring transport lands with the provider adapters
        env = os.environ if env is None else env
        value = env.get(self.env_var, "").strip()
        return value or None


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
            "key_ref": {"source": self.key_ref.source.value, "env_var": self.key_ref.env_var},
            "search_compensator": self.search_compensator,
            "fact_base_edited": self.fact_base_edited,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    # -- key handling ----------------------------------------------------
    def with_provider(self, provider: str) -> "Config":
        """Selecting a provider pre-fills that provider's usual variable."""
        env_var = self.key_ref.env_var or DEFAULT_ENV_VARS.get(provider, "")
        return replace(
            self, provider=provider, key_ref=replace(self.key_ref, env_var=env_var)
        )

    def key(self, env: Mapping[str, str] | None = None) -> str | None:
        return self.key_ref.resolve(env)

    def key_status(self, env: Mapping[str, str] | None = None) -> tuple[KeySource, str]:
        """(source, human description). The key itself is never in the result."""
        value = self.key_ref.resolve(env)
        if value is None:
            var = self.key_ref.env_var or "(not chosen)"
            return KeySource.ABSENT, f"no key found; expected environment variable {var}"
        return (
            self.key_ref.source,
            f"found in {self.key_ref.env_var}, ending {_tail(value)}",
        )

    # -- first-run state machine -----------------------------------------
    def next_step(self, env: Mapping[str, str] | None = None) -> SetupStep:
        if not self.provider:
            return SetupStep.CHOOSE_PROVIDER
        if self.provider == "openrouter" and not self.model_id:
            return SetupStep.NAME_ROUTE
        if self.key(env) is None:
            return SetupStep.SUPPLY_KEY
        if not self.fact_base_edited:
            return SetupStep.EDIT_FACT_BASE
        return SetupStep.READY

    def describe(self, env: Mapping[str, str] | None = None) -> str:
        """The setup screen's status block. Safe to show on screen or paste."""
        source, detail = self.key_status(env)
        lines = [
            f"provider   {self.provider or '(not chosen)'}",
            f"model      {self.model_id or '(provider default)'}",
            f"key        {detail}",
        ]
        if self.search_compensator:
            lines.append(f"search     supplied by {self.search_compensator}")
        lines.append(f"next step  {self.next_step(env).value}")
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
