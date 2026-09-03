"""Build notebooks/claude_probe_anatomy.ipynb — the two facts we cannot get locally.

Two things this session needs and cannot reach from the dev container:

1. **The edge anatomy at the CURRENT chain.** `notes/26` split the edge loss at n=24 on the
   old chain: `fn_mislink` 3.42%, `fn_detect` 1.72%, `fn_gap` 1.53%, and a reachable band of
   +0.047 to +0.079 that contained the whole gap to the cluster. Since then the repair
   chain, ILP weights, bidirectional linking and the relink sweep have all landed and
   `edge_J` moved 0.8902 -> 0.9047. Nobody has re-read the split. `claude_divsweep` computed
   `edge_anatomy` for all 16 arms and wrote it to `divsweep.json`, but the grading cell never
   printed it and `kaggleusercontent.com` is blocked by this container's egress proxy.
   So: mount that kernel's output and print it.

   This is the diagnostic the forum independently converged on. Mendrika Ramarlina (MASTER,
   thread 734604): *"measure the detection ceiling -- the best possible edge score using your
   current detections with an oracle linker"*. Soheil Ayati (18 votes, thread 737101):
   *"break your missed edges into missing endpoint nodes and incorrect associations; in my
   case many 'linking' issues actually originated earlier during node selection"*. Tang
   (MASTER) and hengck23 (GRANDMASTER) both say detection first. `notes/26` says the opposite
   FOR OUR CHAIN -- detection was the smallest bucket. Which of those is true now decides
   whether the next build is a linker or a detector, and it is one print away.

2. **r35's linking code.** `altervation/biohub-r35-spotiflow` is a complete MIT-licensed
   solution. `notes/45`-`47` read its detector half and closed it (0.547 recall against our
   0.996). Its `linker.py` / `ilp.py` -- ~31 KB -- have never been opened, and linking is
   where our remaining gap now sits. Reading them costs nothing and is the last unread
   information in the project.

CPU only. No GPU, no model, no prediction pass.

r35 artefacts are MIT-licensed; the licence and attribution travel with any use.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_probe_anatomy.ipynb")
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Which layer is the gap in, and what does a complete solution's linker do?

Two reads, no compute.

```
0.901 submitted    0.926 bronze    0.944 gold    3,038 teams, top 0.963
config: closed (notes/44, 49)    divisions: closed (notes/43, 50)
```

## 1. The edge anatomy at the current chain

`notes/26` split the edge loss and found **detection was the smallest recoverable bucket**:

```
tp           12,909   93.33%
fn_mislink      473    3.42%   <- largest failure
fn_detect       238    1.72%   <- the detection ceiling
fn_gap          212    1.53%
                              reachable band +0.047 to +0.079
```

That was n=24 on the **old** chain, at `edge_J` 0.8902. The repair chain, ILP weights,
bidirectional linking and the relink sweep have all landed since; `edge_J` is now 0.9047 and
nobody has re-read the split.

The forum converged on this exact diagnostic from the other direction. Mendrika Ramarlina
(MASTER): *"measure the detection ceiling — the best possible edge score using your current
detections with an oracle linker."* Soheil Ayati (18 votes): *"break your missed edges into
missing endpoint nodes and incorrect associations; in my case many 'linking' issues actually
originated earlier during node selection."* Tang (MASTER) and hengck23 (GRANDMASTER) both
answer "detection first" when asked which layer matters.

**`notes/26` says the opposite for our chain.** Whichever is true now decides whether the
next build is a linker or a detector, and `claude_divsweep` already computed it — the
grading cell just never printed it.

## 2. r35's linker

`altervation/biohub-r35-spotiflow` is a complete solution under MIT licence. Its detector
half is closed (`notes/47`: 0.547 recall against our 0.996, and "r35" is a run index, not a
rank — their standing is unknown). Its **linking** half has never been opened, and that is
where the remaining gap is.

*`altervation/biohub-r35-spotiflow` is MIT-licensed. The licence and attribution travel with
any use of its code.*
""")

code(r"""
import json, sys
from pathlib import Path

def find_dir(is_match, roots, max_depth=6):
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop(0)
            try:
                if is_match(d):
                    return d
                if depth >= max_depth:
                    continue
                kids = [e for e in d.iterdir()
                        if e.is_dir() and e.suffix not in (".zarr", ".geff")]
            except (PermissionError, OSError):
                continue
            stack += [(k, depth + 1) for k in kids]
    return None

DIV = find_dir(lambda p: (p / "divsweep.json").is_file(), ["/kaggle/input"])
R35 = find_dir(lambda p: (p / "models").is_dir() and (p / "wheels").is_dir(), ["/kaggle/input"])
print("divsweep output:", DIV)
print("r35:            ", R35)
if DIV is None:
    raise SystemExit("attach claude-divsweep as a kernel source")
""")

md("""## 1. The edge anatomy, per arm""")

code(r"""
D = json.loads((DIV / "divsweep.json").read_text())
A, S = D["anatomy"], D["summary"]
print(f"{len(D['datasets'])} datasets\n")
print(f"{'arm':<14}{'tp':>9}{'fn_gap':>8}{'fn_mislink':>12}{'fn_detect':>11}"
      f"{'reachable':>11}{'edge_J':>9}")
print("-" * 74)
for a in D["arms"]:
    r = A.get(a) or {}
    if not r:
        continue
    print(f"{a:<14}{r.get('tp',0):>9,.0f}{r.get('fn_gap',0):>8,.0f}"
          f"{r.get('fn_mislink',0):>12,.0f}{r.get('fn_detect',0):>11,.0f}"
          f"{r.get('reachable',float('nan')):>11.4f}"
          f"{S.get(a,{}).get('edge_jaccard',float('nan')):>9.4f}")

SHIP = "inc/g2sp6"
r = A.get(SHIP) or {}
if r:
    tp, gap = r.get("tp", 0), r.get("fn_gap", 0)
    mis, det = r.get("fn_mislink", 0), r.get("fn_detect", 0)
    gt = tp + gap + mis + det + r.get("fn_nonconsec", 0)
    fp = r.get("fp", 0) or r.get("n_fp_edges", 0)
    print(f"\n--- the shipped chain ({SHIP}), {gt:,.0f} GT edges ---")
    for k, v in (("tp", tp), ("fn_mislink", mis), ("fn_detect", det), ("fn_gap", gap)):
        print(f"  {k:<12}{v:>9,.0f}{v / gt:>10.2%}")
    print(f"  false-positive edges {fp:,.0f}")
    # notes/26's ceiling arithmetic, recomputed at the current chain.
    now = tp / (tp + fp + (gt - tp)) if (tp + fp + (gt - tp)) else float('nan')
    lo = (tp + gap + mis) / ((tp + gap + mis) + fp + det)
    hi = (tp + gap + mis) / ((tp + gap + mis) + max(fp - mis, 0) + det)
    print(f"\n  edge_J now                              {now:.4f}")
    print(f"  every gap+mislink repaired, FPs kept     {lo:.4f}   {lo - now:+.4f}")
    print(f"  ...and the mislinked FPs become correct  {hi:.4f}   {hi - now:+.4f}")
    print(f"  DETECTION CEILING (fn_detect never fixable) "
          f"{(gt - det) / (gt - det + 0):.4f} of GT edges reachable")
    print(f"\n  VERDICT: {'MISLINKS' if mis > det else 'DETECTION'} dominate "
          f"({mis:,.0f} mislink vs {det:,.0f} undetected)")
    print("  -> the next build is a " + ("LINKER" if mis > det else "DETECTOR"))
""")

md("""## 2. r35's linking code""")

code(r"""
if R35 is None:
    print("r35 not mounted — attach altervation/biohub-r35-spotiflow as a dataset source")
else:
    py = sorted(R35.rglob("*.py"), key=lambda p: -p.stat().st_size)
    print(f"{len(py)} python files under {R35}\n")
    for p in py[:40]:
        print(f"  {p.stat().st_size:>8,}  {p.relative_to(R35)}")
    WANT = ("link", "ilp", "track", "solve", "graph", "assign")
    hits = [p for p in py if any(w in p.name.lower() for w in WANT)]
    print(f"\n=== {len(hits)} linking-related files ===")
    for p in hits:
        print("\n" + "=" * 78)
        print(f"### {p.relative_to(R35)}  ({p.stat().st_size:,} bytes)")
        print("=" * 78)
        try:
            print(p.read_text(errors="replace"))
        except OSError as e:
            print("unreadable:", e)
""")

md("""## 3. Anything r35 documents about its own linking""")

code(r"""
if R35 is not None:
    for pat in ("*.md", "*.yaml", "*.yml", "*.txt"):
        for p in sorted(R35.rglob(pat)):
            if p.stat().st_size > 60_000:
                continue
            t = p.read_text(errors="replace")
            if any(w in t.lower() for w in ("link", "ilp", "track", "solver")):
                print("\n" + "=" * 78)
                print(f"### {p.relative_to(R35)}")
                print("=" * 78)
                print(t[:12_000])
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
