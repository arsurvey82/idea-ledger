"""Entry point: python -m app"""
from __future__ import annotations

import argparse
import os
import sys

from .reloader import CHILD_ENV, supervise
from .web import serve


def main() -> None:
    p = argparse.ArgumentParser(prog="idea-ledger", description="Local idea ledger.")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument(
        "--no-reload", action="store_true",
        help="do not watch for source changes (reloading is on by default)",
    )
    args = p.parse_args()

    # The supervisor is the parent; the child does the serving. The child is
    # marked by an environment variable rather than a flag, so the command it
    # is given is the same command a person typed.
    supervising = not args.no_reload and not os.environ.get(CHILD_ENV)
    if supervising:
        passthrough = [a for a in sys.argv[1:] if a != "--no-reload"]
        # Only the parent opens a browser. Every reload after the first would
        # otherwise open another tab.
        if "--no-browser" not in passthrough:
            passthrough.append("--no-browser")
            serve_browser = True
        else:
            serve_browser = False
        if serve_browser and not args.no_browser:
            _open_later(args.host, args.port)
        raise SystemExit(supervise(passthrough))

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


def _open_later(host: str, port: int, delay: float = 1.2) -> None:
    """Open the browser once the child has had time to bind the port."""
    import threading
    import webbrowser

    threading.Timer(delay, webbrowser.open, [f"http://{host}:{port}"]).start()


if __name__ == "__main__":
    main()
