"""Execute claude_bidirectional's REAL grading cell against constructed results.

The scenario that matters most here is **inert**: a source patch that matched its anchor,
inserted code, and changed nothing — because `_BIDIRECTIONAL_WEIGHT` did not reach the
patched function's globals, say. Every arm is then a copy of the control, and a grading
cell that ranks them would report a sweep of one number four times. Prediction 2 exists to
catch exactly that and is tested here with candidate counts held identical.

Also covered: mechanism-without-outcome (predictions 3 and 4 are deliberately separate,
because the deepcenter veto in `notes/34` did precisely this), a control that fails to
reproduce `notes/35`, and a NaN score column.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NB = REPO / "notebooks" / "claude_bidirectional.ipynb"
WS = [0.0, 0.15, 0.25, 0.40]
LABELS = [f"w{w}" for w in WS] + [f"w{w}+repair" for w in WS]

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def make(scenario: str) -> dict:
    S, A, N, E = {}, {}, {}, {}
    C = {}
    for w in WS:
        # control reproduces notes/35; the blend helps a little, peaking at 0.25
        gain = {0.0: 0.0, 0.15: 0.0035, 0.25: 0.0052, 0.40: 0.0021}[w]
        if scenario == "inert":
            gain = 0.0
        if scenario == "no_outcome":
            gain = -0.0004 * (w / 0.15)      # mechanism fires, score does not
        if scenario == "climbing":
            gain = 0.004 * (w / 0.15)        # best at the largest weight
        base = 0.9179 if scenario != "bad_control" else 0.8600
        mis = 223 - int(40 * w / 0.15) if scenario != "inert" else 223
        if scenario == "no_outcome":
            mis = 223 - int(40 * w / 0.15)   # mislink still falls
        for rep in (False, True):
            lbl = f"w{w}" + ("+repair" if rep else "")
            sc = (base + gain) - (0.0 if rep else 0.0258)
            if scenario == "nan":
                sc = float("nan")
            S[lbl] = {"score": sc, "edge_jaccard": 0.9047 + gain,
                      "division_jaccard": 0.1154}
            A[lbl] = {"fn_mislink": mis, "fn_gap": 86, "fn_detect": 581 + int(20 * w / 0.15)}
            N[lbl], E[lbl] = 318_000, 287_000
        C[f"w{w}"] = 1_000_000 if scenario == "inert" else int(1_000_000 * (1 - 0.03 * w / 0.15))
    return {"arms": LABELS, "weights": WS, "datasets": [f"ds{i}" for i in range(12)],
            "summary": S, "anatomy": A, "nodes": N, "edges": E, "candidates": C,
            "per_dataset": {}}


def main() -> int:
    cells = [c for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    setup, launcher, grading = ("".join(c["source"]) for c in cells)

    print("=" * 80)
    print("the notebook itself")
    print("=" * 80)
    check("three code cells", len(cells) == 3, f"{len(cells)}")
    check("the control arm uses the ORIGINAL function object",
          "if weight == 0.0:" in launcher and "return P.predict_video" in launcher,
          "an exec'd copy at w=0 would make the control differ by a rounding step")
    check("the patch execs into a COPY of the pack namespace",
          "ns = dict(P.__dict__)" in launcher,
          "mutating the pack in place makes the control depend on run order")
    check("an anchor mismatch prints the neighbourhood before dying",
          "ANCHOR DOES NOT MATCH" in launcher and "predict_edges" in launcher,
          "otherwise fixing it costs a second blind GPU round trip")
    check("the dataset list is taken from the cache, with a STRATIFIED fallback",
          "44b6" in launcher and "6bba" in launcher and "round(N_DATASETS" in launcher,
          "notes/34: names[:12] alphabetically inverted the 71/128 embryo split")

    # The worker is written by a .format() at runtime — it must render to valid Python.
    q3 = chr(39) * 3
    a = launcher.index("WORKER.write_text(" + q3) + len("WORKER.write_text(" + q3)
    body = launcher[a:launcher.index(q3, a)]
    body = body.replace("__N_DATASETS__", "12").replace("__WEIGHTS__", "[0.0, 0.15]")
    import ast
    import string
    # Discover the format fields from the body rather than hardcoding them: adding one to
    # the builder would otherwise break this check with a KeyError that says nothing about
    # the notebook. (It just did, for `cellmot`.)
    fields = {f for _, f, _, _ in string.Formatter().parse(body) if f}
    try:
        ast.parse(body.format(**{f: "/" + f for f in fields}))
        check(f"the worker body renders to valid Python (fields: {sorted(fields)})", True)
    except SyntaxError as e:
        check("the worker body renders to valid Python", False, f"line {e.lineno}: {e.msg}")
    except KeyError as e:
        check("the worker body renders to valid Python", False, f"unfilled field {e}")

    # Every field the body asks for must actually be supplied by the launcher's .format().
    supplied = set(re.findall(r"(\w+)=str\(", launcher))
    missing = sorted(fields - supplied)
    check("the launcher supplies every field the worker body asks for", not missing,
          f"missing {missing} — these become KeyError at notebook runtime")

    results = {}
    for scen in ("normal", "inert", "no_outcome", "climbing", "bad_control", "nan"):
        d = Path(f"/tmp/bidir_{scen}")
        d.mkdir(exist_ok=True)
        (d / "bidirectional.json").write_text(json.dumps(make(scen), default=float))
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(grading, "<grading>", "exec"),
                     {"WORK": d, "np": np, "json": json, "__name__": "__main__"})
            results[scen] = buf.getvalue()
            check(f"the grading cell runs under `{scen}`", True)
        except Exception as exc:  # noqa: BLE001
            results[scen] = ""
            check(f"the grading cell runs under `{scen}`", False,
                  f"{type(exc).__name__}: {exc}")

    def seg(txt, i):
        k = f"\n{i}. "
        if k not in txt:
            return ""
        s = txt[txt.index(k):]
        return s[:s.index("\n\n")] if "\n\n" in s else s

    print()
    print("=" * 80)
    print("normal")
    print("=" * 80)
    n = results.get("normal", "")
    print(n[n.index("PREDICTION GRADING"):][:1500])
    for i in range(1, 6):
        body_i = seg(n, i)
        graded = any(w in body_i for w in ("PASS", "FAIL", "NOT GRADED"))
        check(f"prediction {i} produces a verdict", graded,
              "" if graded else "headline with nothing under it")

    print()
    print("--- inert: the patch matched but changed nothing ---")
    print(seg(results.get("inert", ""), 2))
    check("an inert patch fails prediction 2", "FAIL" in seg(results.get("inert", ""), 2))
    check("and says the arms are copies of the control",
          "IDENTICAL" in results.get("inert", ""))

    print()
    print("--- mechanism fires, score does not ---")
    o = results.get("no_outcome", "")
    check("prediction 3 PASSes when mislink falls", "PASS" in seg(o, 3))
    check("prediction 4 FAILs when the score does not follow", "FAIL" in seg(o, 4))
    check("and the divergence is explained, not left bare",
          "does not move the score is still a no" in o)

    print()
    print("--- still climbing at the largest weight ---")
    print(seg(results.get("climbing", ""), 5))
    check("prediction 5 FAILs when the best weight is the largest",
          "FAIL" in seg(results.get("climbing", ""), 5))

    print()
    print("--- the control does not reproduce notes/35 ---")
    b = results.get("bad_control", "")
    check("prediction 1 FAILs on a bad control", "FAIL" in seg(b, 1))
    check("and says nothing below is comparable",
          "Nothing below is comparable" in b)

    print()
    print("--- NaN score column ---")
    q = results.get("nan", "")
    for i in (1, 3, 4, 5):
        check(f"prediction {i} reports NOT GRADED under NaN",
              "NOT GRADED" in seg(q, i), repr(seg(q, i)[-70:]))
    check("prediction 2 still grades under NaN (it reads candidate counts)",
          "PASS" in seg(q, 2) or "FAIL" in seg(q, 2))
    check("no best arm announced under NaN", "NO BEST ARM" in q)

    print()
    print("=" * 80)
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("claude_bidirectional grades correctly under every scenario tried")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
