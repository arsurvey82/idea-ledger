"""Storing the API key without lying about what that protects.

A local app cannot meaningfully encrypt a secret on its own. Whatever key it
would use to decrypt has to be readable by the same process, sitting next to
the ciphertext, so "encrypted at rest" would be obfuscation with a reassuring
label. That is worse than plaintext, because it invites the operator to relax.

What is real is handing the secret to the operating system's own secret store,
which binds it to the logged-in user account and keeps the decryption key out
of our reach entirely:

    Windows   DPAPI (CryptProtectData), scoped to the Windows user
    macOS     Keychain, via the `security` binary
    Linux     Secret Service, via `secret-tool`, when a keyring is running

What this protects against: another user account on the same machine reading
the file, and the encrypted file being copied to a different machine.

What it does not protect against: anything already running as you. Malware with
your privileges can ask the same OS store for the same secret. Nothing a local
tool does changes that, so nothing here pretends otherwise.

When no store is available, storage is refused rather than downgraded to
plaintext, and the operator is pointed at an environment variable.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SERVICE = "idea-ledger"
VAULT_FILE = "key.dpapi"


class SecretStoreUnavailable(RuntimeError):
    """No OS secret store on this machine; refuse rather than store plaintext."""


@dataclass(frozen=True, slots=True)
class StoreInfo:
    backend: str
    detail: str
    available: bool


# --------------------------------------------------------------------------
# Windows: DPAPI
# --------------------------------------------------------------------------
def _dpapi(protect: bool, data: bytes) -> bytes:
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    fn.argtypes = [
        ctypes.POINTER(BLOB), wintypes.LPCWSTR, ctypes.POINTER(BLOB),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(BLOB),
    ]
    fn.restype = wintypes.BOOL

    buf = ctypes.create_string_buffer(data, len(data))
    src = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out = BLOB()

    ok = fn(ctypes.byref(src), SERVICE, None, None, None, 0, ctypes.byref(out))
    if not ok:
        raise SecretStoreUnavailable(
            f"Windows DPAPI call failed (error {ctypes.get_last_error()})"
        )
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def _win_store(home: Path, account: str, secret: str) -> None:
    blob = _dpapi(True, json.dumps({account: secret}).encode("utf-8"))
    path = home / VAULT_FILE
    existing: dict[str, str] = {}
    if path.exists():
        try:
            existing = json.loads(_dpapi(False, path.read_bytes()).decode("utf-8"))
        except Exception:
            existing = {}
    existing[account] = secret
    blob = _dpapi(True, json.dumps(existing).encode("utf-8"))
    path.write_bytes(blob)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _win_fetch(home: Path, account: str) -> str | None:
    path = home / VAULT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(_dpapi(False, path.read_bytes()).decode("utf-8")).get(account)
    except Exception:
        return None


def _win_delete(home: Path, account: str) -> None:
    path = home / VAULT_FILE
    if not path.exists():
        return
    try:
        data = json.loads(_dpapi(False, path.read_bytes()).decode("utf-8"))
    except Exception:
        path.unlink(missing_ok=True)
        return
    data.pop(account, None)
    path.write_bytes(_dpapi(True, json.dumps(data).encode("utf-8")))


# --------------------------------------------------------------------------
# macOS: Keychain
# --------------------------------------------------------------------------
def _mac_run(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, input=stdin, capture_output=True, text=True, check=False
    )


def _mac_store(account: str, secret: str) -> None:
    r = _mac_run(
        ["security", "add-generic-password", "-U", "-a", account, "-s", SERVICE, "-w", secret]
    )
    if r.returncode != 0:
        raise SecretStoreUnavailable(f"keychain refused the write: {r.stderr.strip()}")


def _mac_fetch(account: str) -> str | None:
    r = _mac_run(["security", "find-generic-password", "-a", account, "-s", SERVICE, "-w"])
    return r.stdout.strip() or None if r.returncode == 0 else None


def _mac_delete(account: str) -> None:
    _mac_run(["security", "delete-generic-password", "-a", account, "-s", SERVICE])


# --------------------------------------------------------------------------
# Linux: Secret Service
# --------------------------------------------------------------------------
def _linux_store(account: str, secret: str) -> None:
    r = subprocess.run(
        ["secret-tool", "store", "--label", f"{SERVICE} ({account})",
         "service", SERVICE, "account", account],
        input=secret, capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise SecretStoreUnavailable(f"secret-tool refused the write: {r.stderr.strip()}")


def _linux_fetch(account: str) -> str | None:
    r = subprocess.run(
        ["secret-tool", "lookup", "service", SERVICE, "account", account],
        capture_output=True, text=True, check=False,
    )
    return r.stdout.strip() or None if r.returncode == 0 else None


def _linux_delete(account: str) -> None:
    subprocess.run(
        ["secret-tool", "clear", "service", SERVICE, "account", account],
        capture_output=True, text=True, check=False,
    )


# --------------------------------------------------------------------------
# public surface
# --------------------------------------------------------------------------
def describe() -> StoreInfo:
    system = platform.system()
    if system == "Windows":
        return StoreInfo(
            "Windows DPAPI",
            "encrypted with a key held by Windows and bound to your user account",
            True,
        )
    if system == "Darwin":
        ok = shutil.which("security") is not None
        return StoreInfo(
            "macOS Keychain",
            "stored in your login keychain" if ok else "the `security` tool was not found",
            ok,
        )
    ok = shutil.which("secret-tool") is not None
    return StoreInfo(
        "Secret Service",
        "stored in your desktop keyring" if ok else
        "no `secret-tool` on PATH; install libsecret-tools or use an environment variable",
        ok,
    )


def store(home: Path, account: str, secret: str) -> StoreInfo:
    info = describe()
    if not info.available:
        raise SecretStoreUnavailable(info.detail)
    home.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    if system == "Windows":
        _win_store(home, account, secret)
    elif system == "Darwin":
        _mac_store(account, secret)
    else:
        _linux_store(account, secret)
    return info


def fetch(home: Path, account: str) -> str | None:
    system = platform.system()
    try:
        if system == "Windows":
            return _win_fetch(home, account)
        if system == "Darwin":
            return _mac_fetch(account)
        return _linux_fetch(account)
    except Exception:
        return None


def delete(home: Path, account: str) -> None:
    system = platform.system()
    if system == "Windows":
        _win_delete(home, account)
    elif system == "Darwin":
        _mac_delete(account)
    else:
        _linux_delete(account)


def tail(secret: str, keep: int = 4) -> str:
    """Enough to tell two keys apart. Never more."""
    return "..." + secret[-keep:] if len(secret) > keep else "(too short to display)"


# --------------------------------------------------------------------------
# A named index of what has been saved.
#
# No OS secret store can be enumerated portably: Windows DPAPI has no listing
# at all, and `security`/`secret-tool` need the account name before they will
# say anything. So the *names* live in a plain file next to the ledger while the
# secrets stay in the OS store. This file is safe to read, back up, or commit by
# accident - it holds labels and last-four digits, never a key.
# --------------------------------------------------------------------------

INDEX_FILE = "keys.json"


@dataclass(frozen=True, slots=True)
class SavedKey:
    account: str      # the OS store record name
    provider: str
    label: str        # what the operator calls it
    tail: str         # last four characters, to tell two keys apart

    def as_dict(self) -> dict[str, str]:
        return {"account": self.account, "provider": self.provider,
                "label": self.label, "tail": self.tail}


def account_for(provider: str, label: str) -> str:
    """A stable record name. Labels are the operator's words, so slug them."""
    slug = "".join(c if c.isalnum() else "-" for c in label.strip().lower()).strip("-")
    slug = "-".join(p for p in slug.split("-") if p)[:40]
    return f"{provider}:{slug}" if slug else provider


def saved(home: Path) -> list[SavedKey]:
    path = home / INDEX_FILE
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    out = [
        SavedKey(
            account=str(r.get("account", "")),
            provider=str(r.get("provider", "")),
            label=str(r.get("label", "")) or str(r.get("account", "")),
            tail=str(r.get("tail", "")),
        )
        for r in rows if isinstance(r, dict) and r.get("account")
    ]
    # Only report records the store can still produce. A key deleted out from
    # under us must not appear in a picker that would then resolve to nothing.
    return [k for k in out if fetch(home, k.account)]


def remember(home: Path, account: str, provider: str, label: str, secret: str) -> None:
    """Record that this key exists. The secret itself is never written here."""
    rows = [k.as_dict() for k in saved(home) if k.account != account]
    rows.append(SavedKey(account, provider, label or account, tail(secret)).as_dict())
    home.mkdir(parents=True, exist_ok=True)
    (home / INDEX_FILE).write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


def forget(home: Path, account: str) -> None:
    """Delete the secret and drop it from the index, in that order."""
    delete(home, account)
    rows = [k.as_dict() for k in saved(home) if k.account != account]
    home.mkdir(parents=True, exist_ok=True)
    (home / INDEX_FILE).write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
