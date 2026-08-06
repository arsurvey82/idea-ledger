"""End-to-end walkthrough against a clean install.

Starts a second server on its own port with an empty IDEA_LEDGER_HOME, then
drives the same HTTP surface the browser uses. Nothing is stubbed: the real key,
the real provider, the real pipeline.

The point is not "did it 200". It is whether a person who has never seen this
could set it up, ask a question in their own words, and read the answer. Each
step records what a person would actually see, and the report at the end keeps
the failures in.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import mkdtemp

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PORT = 8477
BASE = f"http://127.0.0.1:{PORT}"
STEPS: list[tuple[str, str, str]] = []   # (status, title, note)


def record(status: str, title: str, note: str = "") -> None:
    STEPS.append((status, title, note))
    mark = {"ok": "PASS", "warn": "WARN", "bad": "FAIL"}[status]
    print(f"  [{mark}] {title}")
    if note:
        for line in note.splitlines():
            print(f"         {line}")


def call(path: str, body=None, timeout=240):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def listen(events: list, stop: threading.Event) -> None:
    try:
        r = urllib.request.urlopen(BASE + "/api/events", timeout=600)
        for raw in r:
            if stop.is_set():
                return
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass


def main() -> int:
    home = Path(mkdtemp(prefix="ledger-e2e-"))
    env = dict(os.environ, IDEA_LEDGER_HOME=str(home), IDEA_LEDGER_PORT=str(PORT))
    proc = subprocess.Popen(
        [sys.executable, "-m", "app", "--port", str(PORT), "--no-browser"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(f"clean install at {home}\n")

    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/api/setup", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    print("server exited:\n" + (proc.stdout.read() if proc.stdout else ""))
                    return 1
                time.sleep(0.5)
        else:
            record("bad", "server never came up")
            return 1

        # ---- 1. what a first-time visitor lands on ----------------------
        s = call("/api/setup")
        first = s.get("next_step")
        record(
            "ok" if first == "choose_provider" else "warn",
            "Fresh install asks for a provider first",
            f"next_step = {first!r}; key_detail = {s.get('key_detail')!r}",
        )

        st = call("/api/state")
        record(
            "ok" if st.get("rules") and st.get("facts") else "bad",
            "Ships with working rules and a fact base",
            f"{len(st.get('rules', []))} rule(s), {len(st.get('facts', {}))} fact(s); "
            f"{len(st.get('ideas', []))} idea(s)",
        )

        # ---- 2. saving a key, and the model resolving itself -------------
        sys.path.insert(0, str(Path(__file__).parent))
        from app.config import Config

        key = Config.load().key()
        if not key:
            record("bad", "no key available to test with")
            return 1

        t0 = time.monotonic()
        out = call("/api/setup", {"provider": "google", "key": key, "label": "e2e"})
        took = time.monotonic() - t0
        picked = out.get("picked_model")
        record(
            "ok" if picked else "bad",
            "Model resolves from the key, unprompted",
            f"picked {picked!r} in {took:.1f}s; stored: {out.get('stored')}",
        )
        record(
            "ok" if out.get("saved_keys") else "bad",
            "Saved key appears in the picker with its name",
            f"{[(k['label'], k['provider'], k['tail']) for k in out.get('saved_keys', [])]}",
        )

        again = call("/api/setup", {"provider": "google", "key": key, "label": "e2e"})
        record(
            "ok" if "unchanged" in str(again.get("stored")) else "warn",
            "Re-saving the same key says so rather than pretending to change",
            f"{again.get('stored')!r}",
        )

        # ---- 3. connection ---------------------------------------------
        probe = call("/api/test-connection", {})
        record(
            "ok" if probe.get("ok") else "bad",
            "Connection test verifies the key for real",
            f"{probe.get('headline')} - {probe.get('detail')}",
        )

        # ---- 4. capability honesty -------------------------------------
        neg = call("/api/setup").get("negotiation", "")
        refused = [l.strip() for l in neg.splitlines() if "cannot run" in l]
        record(
            "ok" if refused else "warn",
            "Stages it cannot run are refused with a reason, not attempted",
            f"{len(refused)} stage(s) refused; e.g. {refused[0][:88] if refused else '-'}",
        )

        # ---- 5. plain language in, real work out ------------------------
        events: list = []
        stop = threading.Event()
        threading.Thread(target=listen, args=(events, stop), daemon=True).start()
        time.sleep(1)

        asked = "lets run wooden furniture"
        t0 = time.monotonic()
        call("/api/chat", {"message": asked})
        time.sleep(3)
        took = time.monotonic() - t0

        tools = [e for e in events if e.get("kind") == "tool"]
        reply = "".join(e.get("text", "") for e in events if e.get("kind") in ("text", "delta"))
        failed = "model call failed" in reply.lower() or "do not have a command" in reply.lower()
        record(
            "bad" if failed or not tools else "ok",
            f"Understands plain language: {asked!r}",
            f"{len(tools)} tool event(s) in {took:.1f}s; "
            f"tools used: {sorted({t.get('tool') for t in tools})}",
        )
        record(
            "ok" if len(reply) > 120 and not failed else "bad",
            "Reply is prose a person can act on",
            (reply[:300].replace("\n", " ") + "...") if reply else "(empty)",
        )

        # ---- 6. is the result readable? ---------------------------------
        st = call("/api/state")
        ideas = st.get("ideas", [])
        scored = [i for i in ideas if i.get("total") is not None]
        record(
            "ok" if ideas else "bad",
            "The run left ideas in the ledger",
            f"{len(ideas)} idea(s), {len(scored)} scored; "
            f"statuses: {sorted({i['status'] for i in ideas})}",
        )

        if scored:
            d = call(f"/api/idea?id={scored[0]['id']}")
            dims = d.get("dimensions", [])
            with_falsifier = [x for x in dims if x.get("falsifier")]
            record(
                "ok" if dims and with_falsifier else "warn",
                "Each score carries its own falsifier and provenance",
                f"{len(dims)} dimension(s), {len(with_falsifier)} with a falsifier; "
                f"sample: {dims[0].get('dimension')}={dims[0].get('value')} "
                f"({dims[0].get('provenance')}) - "
                f"{str(dims[0].get('falsifier'))[:70]!r}",
            )

        rej = st.get("rejections", [])
        named = [r for r in rej if r.get("cause")]
        record(
            "ok" if not rej or named else "warn",
            "Rejections name the cause",
            f"{len(rej)} rejection(s); e.g. {named[0]['cause'][:80] if named else '-'}",
        )

        # ---- 7. changing a rule in your own words -----------------------
        events.clear()
        call("/api/chat", {"message": "add a rule rejecting anything needing more than 8 hours a week"})
        time.sleep(3)
        rule_tools = [e.get("tool") for e in events if e.get("kind") == "tool"]
        after = call("/api/state")
        activated = len(after.get("rules", [])) > len(st.get("rules", []))
        record(
            "ok" if "preview_rule" in rule_tools else "warn",
            "A rule change is previewed before it is applied",
            f"tools: {sorted(set(rule_tools))}; rules now {len(after.get('rules', []))} "
            f"(was {len(st.get('rules', []))})",
        )
        record(
            "ok" if not activated else "warn",
            "Nothing is activated without an explicit yes",
            "rule count unchanged" if not activated
            else "a rule was activated from one message, without confirmation",
        )

        # ---- 8. can you get the work out? -------------------------------
        try:
            arch = call("/api/archive", {"idea": scored[0]["id"]} if scored else {})
            # This check reported a false PASS first time: the wrong response
            # key gave Path(""), and Path(".").is_dir() is true, so it listed
            # the repository and called it an archive. Assert on the document.
            raw = str(arch.get("archived", ""))
            path = Path(raw) if raw else None
            files = (
                sorted(x.name for x in path.rglob("*") if x.is_file())
                if path and path.is_dir() else []
            )
            body = ""
            if path and (path / "dossier.md").exists():
                body = (path / "dossier.md").read_text(encoding="utf-8")
            headings = [l for l in body.splitlines() if l.startswith("#")]
            record(
                "ok" if raw and files and len(headings) >= 3 else "bad",
                "Archiving writes a dated folder you can open outside the app",
                f"{raw or '(no path returned)'}\n{files}\n"
                f"dossier.md: {len(body)} chars, {len(headings)} headings",
            )
        except urllib.error.HTTPError as exc:
            record("warn", "Archive", f"HTTP {exc.code}")

        stop.set()
        return 0
    finally:
        proc.terminate()
        print("\n" + "=" * 72)
        counts = {k: sum(1 for s, _, _ in STEPS if s == k) for k in ("ok", "warn", "bad")}
        print(f"PASS {counts['ok']}   WARN {counts['warn']}   FAIL {counts['bad']}")
        for status, title, _ in STEPS:
            if status != "ok":
                print(f"  {status.upper():4} {title}")
        print(f"\nclean install left at: {home}")


if __name__ == "__main__":
    raise SystemExit(main())
