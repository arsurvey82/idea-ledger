"""Entry point: python -m app"""
from __future__ import annotations

import argparse

from .web import serve


def main() -> None:
    p = argparse.ArgumentParser(prog="idea-ledger", description="Local idea ledger.")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
