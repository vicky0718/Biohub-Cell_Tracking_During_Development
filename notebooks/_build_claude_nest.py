"""Build notebooks/claude_nest.ipynb — read the hidden test set's node budget. CPU only.

The metric's other half, and the one nobody in the public lineage talks about:

    adj = max(0, edge_J * (1 - 0.1 * ratio)),   ratio = (N_pred - N_est) / N_est

`N_est` is GEFF's `estimated_number_of_nodes`, read **per dataset** — recon §9 measured it
varying 20.8x across datasets, so a global constant turns this multiplier into a trap. The
term is floored at zero and **uncapped above**: predicting fewer nodes than the estimate
multiplies `edge_J` by more than one.

Every public notebook tunes `edge_J`. `BIOHUB_OUTPUT_FILTER_SHORT_TRACKS`,
`OUTPUT_MIN_TRACK_LEN=6` and `SHORT_TRACK_RESCUE_MAX_NODES_ABS=120` say their authors found
pruning pays, but nothing in the 0.934 → 0.941 progression is described as targeting this
ratio, and no published header mentions `N_est` at all.

**What we do not know is which side of the budget the fork sits on.** `claude_fork` (LB
0.937) emitted 118,659 nodes across the four test datasets. If `N_est` is larger, the fork
is already collecting a bonus and pruning further trades `edge_J` for very little. If
`N_est` is smaller, there is an unclaimed multiplier sitting on the table and a pruning pass
pays twice — fewer false edges *and* a larger factor.

This notebook answers that with no GPU and no submission slot. It mounts the competition
data, finds each test dataset's GEFF, and prints `estimated_number_of_nodes` beside the node
counts `claude_fork` actually produced. `notes/65`: Kaggle is handing this account nothing
but P100s, so a CPU notebook also dodges the accelerator problem entirely and gets the 12 h
limit instead of 9 h.

If the test GEFFs carry no metadata — plausible for a hidden test set — that is the finding
and the direction closes here rather than after a wasted GPU run. The listing is printed
either way so the next attempt knows what is actually mounted.
"""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "claude_nest.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Which side of the node budget is the fork on?

```
score = adj_edge_jaccard + 0.1 * division_jaccard
adj   = max(0, edge_J * (1 - 0.1 * (N_pred - N_est) / N_est))
```

`N_est` is GEFF's `estimated_number_of_nodes`, per dataset. The term is floored at zero and
**uncapped above**, so under-predicting multiplies `edge_J` by more than one.

`claude_fork` (LB **0.937**) emitted **118,659** nodes across the four test datasets:

```
6bba_05db0fb1   69,112      44b6_0113de3b   25,425
44b6_0b24845f   18,089      6bba_05b6850b    6,033
```

Two crops each of two embryos — the third pair `notes/49` predicted.

**Pre-registered predictions.**

1. The test GEFFs are mounted and at least one carries `estimated_number_of_nodes`.
   If this fails nothing below is readable and the direction closes here.
2. `N_est` varies by more than 5x across the four datasets (recon §9 measured 20.8x on
   train). A single global budget would be the trap this check exists to avoid.
3. **The interesting one.** The fork's total ratio is *positive* — it over-predicts, and
   the multiplier is currently costing it score rather than paying it.

Prediction 3 is the one I expect to fail: `OUTPUT_MIN_TRACK_LEN=6` and
`OUTPUT_FILTER_SHORT_TRACKS=1` suggest the lineage already prunes hard. Failing it is still
worth knowing, because it prices every future pruning idea at zero.
""")

code(r"""
import os, json, glob
from pathlib import Path

ROOTS = [Path("/kaggle/input")]
print("=== what is mounted ===")
for r in ROOTS:
    for p in sorted(r.glob("*")):
        print(" ", p.name)

# The fork's own predict step used --data-dir /kaggle/input/competitions/<comp>/test,
# so look there first, then fall back to a search for anything named `test`.
CAND = [Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test"),
        Path("/kaggle/input/biohub-cell-tracking-during-development/test")]
TEST = next((p for p in CAND if p.is_dir()), None)
if TEST is None:
    hits = [Path(p) for p in glob.glob("/kaggle/input/**/test", recursive=True)][:5]
    print("\nfallback search for a `test` dir:", hits)
    TEST = hits[0] if hits else None
print("\nTEST =", TEST)
""")

code(r"""
entries = sorted(TEST.iterdir()) if TEST else []
print(f"{len(entries)} entries under TEST")
for p in entries[:12]:
    kind = "dir " if p.is_dir() else "file"
    print(f"  {kind} {p.name}")
    if p.is_dir():
        for q in sorted(p.iterdir())[:6]:
            print(f"        {q.name}")
""")

code(r"""
# `estimated_number_of_nodes` lives in GEFF metadata extras. harness/tracks.py reads it
# with geff.GeffMetadata; here we do not have our harness mounted, so read the zattrs
# directly as well -- a GEFF store is zarr, and its metadata is plain JSON on disk. Two
# independent routes because a silent None from either would look like "absent".
import json
from pathlib import Path

def from_geff(path):
    try:
        from geff import GeffMetadata
        m = GeffMetadata.read(str(path))
        ex = m.extra or {}
        return ex.get("estimated_number_of_nodes")
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"

def from_zattrs(path):
    found = {}
    for zp in list(Path(path).rglob(".zattrs"))[:40] + list(Path(path).rglob("zarr.json"))[:40]:
        try:
            d = json.loads(zp.read_text())
        except Exception:
            continue
        stack = [d]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                for k, v in o.items():
                    if "estimated" in str(k) and "node" in str(k):
                        found[str(zp.relative_to(path))] = v
                    stack.append(v)
            elif isinstance(o, list):
                stack.extend(o)
    return found

geffs = sorted(Path(TEST).rglob("*.geff")) if TEST else []
print(f"{len(geffs)} .geff paths under TEST")
for g in geffs[:8]:
    print(f"\n{g.relative_to(TEST)}")
    print("   geff  :", from_geff(g))
    print("   zattrs:", from_zattrs(g))
""")

code(r"""
# If the .geff route came up empty, sweep the whole mount for the key by name.
import json
from pathlib import Path
hits = {}
for zp in Path("/kaggle/input").rglob("*.json"):
    if zp.stat().st_size > 4_000_000:
        continue
    try:
        txt = zp.read_text()
    except Exception:
        continue
    if "estimated_number_of_nodes" in txt:
        hits[str(zp)] = txt[:400]
        if len(hits) >= 12:
            break
print(f"{len(hits)} json files mention estimated_number_of_nodes")
for k, v in hits.items():
    print("\n", k, "\n   ", v.replace(chr(10), " ")[:300])
""")

code(r"""
# Grade against the fork's actual output. FORK_NODES is measured, not inferred: it is
# `row_type == node` counted per dataset in claude-fork's submission.csv (LB 0.937),
# pulled through /kernels/output/download.
FORK_NODES = {"6bba_05db0fb1": 69112, "44b6_0113de3b": 25425,
              "44b6_0b24845f": 18089, "6bba_05b6850b": 6033}
TOTAL_PRED = sum(FORK_NODES.values())

est = {}
for g in (geffs or []):
    v = from_geff(g)
    if isinstance(v, (int, float)):
        est[g.stem] = float(v)
    else:
        z = from_zattrs(g)
        num = next((x for x in z.values() if isinstance(x, (int, float))), None)
        if num is not None:
            est[g.stem] = float(num)

print("=" * 74)
print("PREDICTION GRADING")
print("=" * 74)

ok1 = bool(est)
print(f"\n1. test GEFFs expose estimated_number_of_nodes  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   Nothing below is readable. The node-budget direction closes here: the test")
    print("   mount does not carry the scorer's N_est, so it cannot be targeted at")
    print("   inference time and can only ever be probed by spending submission slots.")
else:
    print(f"   {len(est)} datasets: " + ", ".join(f"{k}={v:,.0f}" for k, v in est.items()))

    lo, hi = min(est.values()), max(est.values())
    ok2 = hi > 5 * lo
    print(f"\n2. N_est varies more than 5x across datasets  ->  {'PASS' if ok2 else 'FAIL'}")
    print(f"   min {lo:,.0f}   max {hi:,.0f}   ratio {hi / max(lo, 1):.1f}x")

    print(f"\n3. the fork OVER-predicts (ratio > 0, the multiplier is costing it)")
    print(f"   {'dataset':<18}{'N_pred':>10}{'N_est':>12}{'ratio':>10}{'factor':>10}")
    tot_est = 0.0
    for k in sorted(FORK_NODES):
        e = est.get(k)
        n = FORK_NODES[k]
        if e is None:
            print(f"   {k:<18}{n:>10,}{'-':>12}{'-':>10}{'-':>10}")
            continue
        tot_est += e
        r = (n - e) / e
        print(f"   {k:<18}{n:>10,}{e:>12,.0f}{r:>+10.3f}{1 - 0.1 * r:>10.4f}")
    if tot_est:
        R = (TOTAL_PRED - tot_est) / tot_est
        ok3 = R > 0
        print(f"   {'TOTAL':<18}{TOTAL_PRED:>10,}{tot_est:>12,.0f}{R:>+10.3f}"
              f"{1 - 0.1 * R:>10.4f}   ->  {'PASS' if ok3 else 'FAIL'}")
        print(f"\n   The factor multiplies edge_J. At edge_J ~ 0.92 the fork's adjustment is")
        print(f"   worth {0.92 * (1 - 0.1 * R) - 0.92:+.4f} of score right now.")
        if ok3:
            print("   Pruning toward N_est pays TWICE: a larger factor and fewer false edges.")
            print("   Reaching exactly N_est would be worth "
                  f"{0.92 * 1.0 - 0.92 * (1 - 0.1 * R):+.4f} from the factor alone.")
        else:
            print("   The fork is already under budget and collecting the bonus. Every")
            print("   further prune trades edge_J for a factor that is already above 1,")
            print("   which prices the whole pruning direction at roughly zero -- and")
            print("   explains why notes/46, notes/48 and notes/52 all closed.")
print("=" * 74)
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

for i, c in enumerate(CELLS):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT.name}: {len(CELLS)} cells, all code cells parse")
