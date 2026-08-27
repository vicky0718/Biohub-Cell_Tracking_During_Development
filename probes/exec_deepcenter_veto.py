"""Execute claude_deepcenter_veto's REAL grading cell against constructed results.

The grading cell is where a run's conclusion is written, and this project has twice found
it printing a headline with nothing under it when the score column went NaN — which reads
as a pass. So every prediction is exercised under four scenarios, including the two that
matter most here:

  * **alignment failure.** Prediction 1 is load-bearing: if the heatmap is read at the
    wrong voxel the veto still fires, it just fires at random, and every arm below it is
    noise dressed as a measurement. The cell must say so loudly rather than ranking arms.
  * **mechanism/outcome divergence.** Predictions 4 and 5 are deliberately separate — the
    veto can reduce `fn_detect` exactly as designed and still lose on score, because each
    refused gap is two edges not recovered. A grading cell that collapses those into one
    verdict would hide the most likely real result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NB = REPO / "notebooks" / "claude_deepcenter_veto.ipynb"
THRESHOLDS = [0.10, 0.25, 0.40, 0.60]
LABELS = ["norepair", "repair"] + [f"veto{th}" for th in THRESHOLDS]

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def make(scenario: str) -> dict:
    """Results in the worker's own schema, shaped to trip one prediction at a time."""
    S, A, N, E, V = {}, {}, {}, {}, {}
    for lbl in LABELS:
        if lbl == "norepair":
            sc, mis, gap, det, nodes = 0.8806, 411, 177, 218, 129_000
        elif lbl == "repair":
            sc, mis, gap, det, nodes = 0.8958, 382, 159, 254, 130_400
        else:
            th = float(lbl[len("veto"):])
            # fn_detect falls with the threshold (the mechanism), fn_gap rises (the cost)
            det = 254 - int(36 * min(th / 0.4, 1.0))
            gap = 159 + int(40 * th)
            mis, nodes = 382, 130_400 - int(1200 * th)
            sc = {0.10: 0.8969, 0.25: 0.8974, 0.40: 0.8961, 0.60: 0.8930}[th]
            if scenario == "outcome_fail":       # mechanism works, score does not
                sc = 0.8958 - 0.001 * (th / 0.1)
        if scenario == "nan":
            sc = float("nan")
        S[lbl] = {"score": sc, "edge_jaccard": sc - 0.0000, "division_jaccard": 0.0}
        A[lbl] = {"fn_mislink": mis, "fn_gap": gap, "fn_detect": det}
        N[lbl], E[lbl] = nodes, 118_000
    for th in THRESHOLDS:
        if scenario == "degenerate":
            kept = 20_000 if th < 0.2 else 5        # rejects ~nothing, then ~everything
        else:
            kept = int(20_000 * (1.0 - 0.15 * th / 0.1))
        V[f"veto{th}"] = {"proposed": 20_000, "kept": kept}
    align = {}
    for i, nm in enumerate(("44b6_aa", "44b6_bb", "6bba_cc")):
        if scenario == "misaligned":
            align[nm] = {"n": 900, "gt_median": 0.031, "gt_mean": 0.04,
                         "rnd_median": 0.028, "rnd_mean": 0.03,
                         "frac_gt_above_0.25": 0.04, "frac_rnd_above_0.25": 0.03}
        else:
            align[nm] = {"n": 900 + i, "gt_median": 0.86, "gt_mean": 0.79,
                         "rnd_median": 0.019, "rnd_mean": 0.05,
                         "frac_gt_above_0.25": 0.94, "frac_rnd_above_0.25": 0.05}
    return {"arms": LABELS, "datasets": [f"ds{i}" for i in range(12)],
            "thresholds": THRESHOLDS, "alignment": align, "veto": V,
            "summary": S, "anatomy": A, "nodes": N, "edges": E, "per_dataset": {},
            "checkpoint": "/kaggle/input/dc/weights/full_frame_center/best.pt",
            "n_params": 1_234_567}


def main() -> int:
    import io
    import contextlib

    import numpy as np

    cells = json.loads(NB.read_text())["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    grading = "".join(code[-1]["source"])

    print("=" * 78)
    print("the notebook itself")
    print("=" * 78)
    check("the notebook has the expected cell layout", len(code) == 3, f"{len(code)} code cells")
    worker = "".join(code[1]["source"])
    check("arm labels are unique", len(set(LABELS)) == len(LABELS))
    check("this script's labels are the worker's",
          all(f'veto{th}' in worker or f'{th}' in worker for th in THRESHOLDS)
          and '"norepair", "repair"' in worker,
          "the builder's LABELS changed without this script following")
    check("the veto is wired into close_gaps, not applied after it",
          "accept=acc" in worker and "close_gaps" in worker)
    check("a failed checkpoint load raises instead of running an unvetoed control",
          "no deepcenter checkpoint loaded" in worker and "SystemExit" in worker,
          "silently falling back to no-veto would make every arm its own control")
    check("the heatmap cache is sized to hold a whole dataset",
          "max_frames=110" in worker,
          "a small cache recomputes every frame once per arm")

    results = {}
    for scen in ("normal", "misaligned", "degenerate", "outcome_fail", "nan"):
        d = Path(f"/tmp/dcv_{scen}")
        d.mkdir(exist_ok=True)
        (d / "deepcenter_veto.json").write_text(json.dumps(make(scen), default=float))
        buf = io.StringIO()
        ns = {"WORK": d, "np": np, "json": json, "__name__": "__main__"}
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(grading, "<grading>", "exec"), ns)
            results[scen] = buf.getvalue()
        except Exception as exc:  # noqa: BLE001
            results[scen] = ""
            check(f"the grading cell runs under `{scen}`", False,
                  f"{type(exc).__name__}: {exc}")
            continue
        check(f"the grading cell runs under `{scen}`", True)

    print()
    print("=" * 78)
    print("what each scenario concludes")
    print("=" * 78)

    def seg(txt: str, start: str, end: str | None = None) -> str:
        if start not in txt:
            return ""
        s = txt[txt.index(start):]
        return s[:s.index(end)] if end and end in s else s

    n = results.get("normal", "")
    print(seg(n, "PREDICTION GRADING", "=" * 85).rstrip()[:1600])
    for i in range(1, 6):
        body = seg(n, f"\n{i}. ")
        body = body[:body.index("\n\n")] if "\n\n" in body else body
        graded = "PASS" in body or "FAIL" in body or "NOT GRADED" in body
        check(f"prediction {i} is graded under `normal`", graded,
              "" if graded else "printed a headline with no verdict under it")

    m = results.get("misaligned", "")
    print()
    print("--- misaligned ---")
    print(seg(m, "\n1. ", "\n2. ").rstrip())
    check("a misaligned heatmap fails prediction 1", "FAIL" in seg(m, "\n1. ", "\n2. "))
    check("and says the rest of the run is unreadable",
          "NOTHING BELOW IS READABLE" in m,
          "ranking arms off a randomly-firing veto is the failure mode here")

    g = results.get("degenerate", "")
    print()
    print("--- degenerate veto rates ---")
    print(seg(g, "\n3. ", "\n4. ").rstrip())
    check("a veto that rejects ~nothing or ~everything fails prediction 3",
          "FAIL" in seg(g, "\n3. ", "\n4. ") and "degenerate" in g)

    o = results.get("outcome_fail", "")
    print()
    print("--- mechanism works, outcome does not ---")
    print(seg(o, "\n4. ", "=" * 85).rstrip())
    p4 = seg(o, "\n4. ", "\n5. ")
    p5 = seg(o, "\n5. ", "=" * 85)
    check("prediction 4 still PASSes when the mechanism works", "PASS" in p4)
    check("prediction 5 FAILs when the score does not follow", "FAIL" in p5)
    check("and the divergence is explained rather than left as a bare FAIL",
          "costs more than it saves" in p5)
    check("a no-win result points at the SECOND missing model",
          "temporal-unet3d-seed314159" in o,
          "the veto failing says nothing about the linker blend")

    q = results.get("nan", "")
    print()
    print("--- NaN score column ---")
    ng = seg(q, "PREDICTION GRADING")
    for i in (2, 4, 5):
        body = seg(ng, f"\n{i}. ")
        body = body[:body.index("\n\n")] if "\n\n" in body else body
        check(f"prediction {i} reports NOT GRADED under NaN, not silence",
              "NOT GRADED" in body, repr(body[-90:]))
    check("no best arm is announced under NaN", "NO BEST ARM" in q)
    check("prediction 1 still grades under NaN (it does not use the score column)",
          "PASS" in seg(ng, "\n1. ", "\n2. "),
          "alignment is measured from heatmap scores and survives a bad node budget")

    print()
    print("=" * 78)
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("claude_deepcenter_veto grades correctly under every scenario tried")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
