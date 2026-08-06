"""Guard the served page against silent breakage.

The Python suite cannot see the browser, so the UI has twice been broken in
ways every test passed through:

* a placeholder that expanded to NUL bytes, which stopped the parser at the
  first one and killed every button on the page;
* a whole-file repair that dropped the ``api`` helper, so every panel threw
  ``ReferenceError: api is not defined`` on load.

Both are cheap to catch and expensive to find by hand. These tests parse and
then *execute* the page's script against a stub DOM, which is enough to catch a
missing global, a syntax error, or a function that recurses with no base case.

They skip when Node is absent rather than fail, because Node is a convenience
here and not a dependency of the application.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"
NODE = shutil.which("node")


def page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


def page_script() -> str:
    match = re.search(r"<script[^>]*>(.*)</script>", page_text(), re.S)
    assert match, "the page has no script block"
    return match.group(1)


#: A DOM stub broad enough to load the page. Every element is a proxy that
#: answers any property, so the script runs to completion instead of stopping at
#: the first attribute this file forgot to model.
HARNESS = r"""
const vm = require('vm'), fs = require('fs');
const missing = new Set(), errors = [];

const el = () => new Proxy({
  style:{}, dataset:{}, classList:{add(){},remove(){},toggle(){},contains:()=>false},
  querySelectorAll:()=>[], querySelector:()=>el(), appendChild(){}, addEventListener(){},
  setAttribute(){}, getAttribute:()=>null, focus(){}, click(){}, remove(){}, insertBefore(){},
  scrollHeight:0, scrollTop:0, clientHeight:0, value:'', textContent:'', innerHTML:'',
  children:[], disabled:false, hidden:false, title:'', checked:false,
}, {get:(t,k)=> k in t ? t[k] : undefined, set:(t,k,v)=>{t[k]=v; return true}});

const RESP = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const sandbox = {
  console:{log(){},warn(){},error(){}}, JSON, Math, Object, Array, String, Number,
  Boolean, Date, Promise, Set, Map, RegExp, Error, parseInt, parseFloat, isNaN,
  setTimeout:()=>0, setInterval:()=>0, clearTimeout(){}, clearInterval(){},
  encodeURIComponent, decodeURIComponent, requestAnimationFrame:f=>f(),
  fetch: async p => ({json: async () => RESP[String(p).split('?')[0]] ?? {}}),
  localStorage:{getItem:()=>null, setItem(){}, removeItem(){}},
  matchMedia:()=>({matches:false, addEventListener(){}}),
  EventSource:function(){ this.addEventListener=()=>{}; },
  document:{documentElement:{dataset:{},style:{}}, body:el(), querySelector:()=>el(),
            querySelectorAll:()=>[], createElement:()=>el(), addEventListener(){},
            getElementById:()=>el()},
};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
process.on('unhandledRejection', e => errors.push('async: ' + e.message));

// has:()=>true forces every free identifier through get, so a name the script
// uses but never defines is recorded rather than throwing on first use.
const ctx = vm.createContext(new Proxy(sandbox, {
  has: () => true,
  get(t, k) {
    if (k in t) return t[k];
    if (typeof k === 'string') missing.add(k);
    return undefined;
  },
}));

try { vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx, {filename:'page.js'}); }
catch (e) { errors.push('load: ' + e.message); }

setTimeout(() => {
  const real = [...missing].filter(k => !/^(Symbol|require|module|exports|process|global)/.test(k));
  console.log(JSON.stringify({missing: real, errors}));
}, 200);
"""

#: Response shapes the page expects. Kept faithful to the server's own payloads;
#: a key missing here shows up as a stub artefact rather than a page defect.
STUB_RESPONSES = {
    "/api/state": {
        "ideas": [{
            "id": "i1", "title": "T", "status": "scored", "track": "a",
            "total": 30, "rubric_version": "1", "dimensions": [],
        }],
        "rules": [], "facts": {}, "assumptions": "", "calibration": "",
        "manifest": "", "rejections": [],
    },
    "/api/setup": {
        "provider": "openrouter", "model_id": "a/b:free", "has_key": True,
        "verified": True, "key_detail": "d", "store_available": True,
        "store_backend": "B", "store_detail": "s", "env_var": "E",
        "negotiation": "n", "config_text": "c", "home": "h", "next_step": "ready",
    },
    "/api/models": {"models": []},
    "/api/logs": {"lines": []},
}


class ThePageIsIntact(unittest.TestCase):
    def test_the_script_holds_no_control_characters(self) -> None:
        """A stray NUL stops the parser dead and takes every handler with it."""
        text = page_text()
        bad = {c for c in text if ord(c) < 32 and c not in "\t\n\r"}
        self.assertFalse(
            bad,
            "control characters in the page: "
            + ", ".join(f"U+{ord(c):04X}" for c in sorted(bad)),
        )

    def test_every_referenced_element_exists(self) -> None:
        """``$('#x')`` with no matching id is a null dereference at runtime."""
        text = page_text()
        present = set(re.findall(r'id="([^"]+)"', text))
        used = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", text))
        self.assertEqual(
            set(), used - present, "selected but not in the document"
        )

    @unittest.skipIf(NODE is None, "node is not installed")
    def test_the_script_loads_with_no_undefined_globals(self) -> None:
        """Executes the page. Catches a helper lost to an edit, as ``api`` was."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "page.js").write_text(page_script(), encoding="utf-8")
            (base / "harness.js").write_text(HARNESS, encoding="utf-8")
            (base / "resp.json").write_text(json.dumps(STUB_RESPONSES), encoding="utf-8")
            proc = subprocess.run(
                [NODE, str(base / "harness.js"), str(base / "page.js"), str(base / "resp.json")],
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual([], report["missing"], "used but never defined")
        self.assertEqual([], report["errors"], "threw while loading")

    def test_no_function_calls_only_itself(self) -> None:
        """``function follow() { if (STICK) follow(); }`` shipped and overflowed."""
        offenders = []
        for match in re.finditer(
            r"function\s+(\w+)\s*\([^)]*\)\s*\{([^{}]*)\}", page_script()
        ):
            name, body = match.group(1), match.group(2)
            calls_self = re.search(r"\b" + re.escape(name) + r"\s*\(", body)
            # A single self-call in a body with no other call is unconditional
            # recursion however it is guarded, because the guard cannot change
            # between the test and the call.
            if calls_self and len(re.findall(r"\w+\s*\(", body)) == 1:
                offenders.append(f"{name}: {body.strip()}")
        self.assertEqual([], offenders, "recursion with no base case")


if __name__ == "__main__":
    unittest.main()
