"""Restart the server when its own source changes.

Python has no true hot module replacement: rebinding a live module leaves every
already-imported reference pointing at the old object, so a partial reload is
worse than none. What actually works is the thing Flask and uvicorn do - a
supervisor process that watches the tree and replaces the child.

That gives the same result from the outside. Source changes, the server comes
back on the same port within a second, and the page reloads itself because the
build stamp it holds no longer matches the server's.

Static files need none of this: ``_static`` reads them from disk per request, so
editing ``index.html`` is already live on the next refresh.

Standard library only, like everything else here - no watchdog, no inotify
bindings. Polling mtimes over a few dozen files costs nothing at this size.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

#: Set in the child so it knows not to supervise in turn.
CHILD_ENV = "IDEA_LEDGER_CHILD"

ROOT = Path(__file__).resolve().parent
WATCH_SUFFIXES = (".py", ".html", ".json")
POLL_SECONDS = 0.7


def watched_files(root: Path = ROOT) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.suffix in WATCH_SUFFIXES and "__pycache__" not in path.parts:
            yield path


def fingerprint(root: Path = ROOT) -> tuple:
    """What "changed" means: any watched file added, removed, or touched."""
    stamps = []
    for path in watched_files(root):
        try:
            stamps.append((str(path), path.stat().st_mtime_ns))
        except OSError:
            continue          # deleted between listing and stat; next poll sees it
    return tuple(sorted(stamps))


def build_stamp(root: Path = ROOT) -> str:
    """A short id for the current source. The page compares it across reconnects."""
    latest = 0
    count = 0
    for path in watched_files(root):
        try:
            latest = max(latest, path.stat().st_mtime_ns)
            count += 1
        except OSError:
            continue
    return f"{latest}-{count}"


def supervise(argv: list[str]) -> int:
    """Run the server as a child, replacing it whenever the source changes.

    Returns the child's exit code when interrupted. The child is killed rather
    than asked politely: it is blocked in ``serve_forever`` and holds the port,
    and waiting on a graceful shutdown is the difference between a reload that
    feels instant and one that does not.
    """
    env = dict(os.environ, **{CHILD_ENV: "1"})
    command = [sys.executable, "-m", "app", *argv]

    child = subprocess.Popen(command, env=env)
    seen = fingerprint()
    print(f"watching {len(seen)} files; edit and save to reload", flush=True)

    try:
        while True:
            time.sleep(POLL_SECONDS)

            if child.poll() is not None:
                # It exited on its own - a syntax error, or a port already in
                # use. Keep watching so saving the fix brings it back rather
                # than making someone rerun the command.
                if fingerprint() != seen:
                    seen = fingerprint()
                    print("source changed; retrying", flush=True)
                    child = subprocess.Popen(command, env=env)
                continue

            current = fingerprint()
            if current != seen:
                seen = current
                print("source changed; reloading", flush=True)
                child.kill()
                child.wait(timeout=10)
                child = subprocess.Popen(command, env=env)
    except KeyboardInterrupt:
        pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    return child.returncode or 0
