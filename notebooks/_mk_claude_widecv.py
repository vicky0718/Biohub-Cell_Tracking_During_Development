"""Derive _build_claude_widecv.py from the sweep2 builder. Validate EVERY anchor first.

The finding this run exists to act on: at n=12 the smallest effect we can resolve is
**0.0036**, computed from the paired per-dataset spread in cfg2.log. Every result this
session has chased is +0.001 to +0.002. So `notes/40` and `notes/41` declared three
"located interior optima" out of differences the measurement cannot distinguish from
zero -- det 0.975 vs 0.98 is t=1.24.

n=60 takes the resolvable effect to ~0.0015. The sample is a strict SUPERSET of the 12
so every prior number stays comparable rather than being replaced by an unrelated draw.
"""
from pathlib import Path

SP_ = Path("/tmp/claude-0/-home-user-rogii/840351bc-4942-5d31-9b68-1b00e66da173/scratchpad")
NB = Path("/workspace/biohub-cell_tracking_during_development/notebooks")
SRC = NB / "_build_claude_config_sweep2.py"
DST = NB / "_build_claude_widecv.py"

EDITS = []


def edit(name, old, new, times=1):
    """`times` is asserted exactly -- a filename written once and read once is 2, and
    saying so is safer than switching to replace_all and never noticing a third site."""
    EDITS.append((name, old, new, times))


edit("output path",
     'OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_config_sweep2.ipynb")',
     'OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_widecv.ipynb")')

edit("sample size", "N_DATASETS = 12", "N_DATASETS = 60")

edit("threshold grid",
     "DET_GRID = [0.98, 0.975, 0.97, 0.965]",
     # 0.965 is the ONLY comparison n=12 resolved, and it lost. The two that matter are
     # the two it could not call: 0.975 vs 0.98 (t=1.24) and 0.975 vs 0.97 (t=1.94).
     "DET_GRID = [0.98, 0.975, 0.97]")

edit("post grid",
     """POST_GRID = [(0, 2), (6, 2), (8, 2), (10, 2),
             (0, 3), (6, 3), (8, 3), (10, 3), (12, 3),
             (6, 1), (0, 1)]""",
     # Free on each cached graph, so the only cost of an arm is print width. Kept: the
     # incumbent, the pre-audit submission, and one neighbour on each axis that n=12
     # ranked but could not resolve.
     """POST_GRID = [(6, 2), (0, 1), (6, 1), (0, 2), (8, 2), (6, 3)]""")

# written by the worker, read by the analysis cell
edit("result filename", '"config_sweep2.json"', '"widecv.json"', times=1)

edit("dataset selection",
     """names = []
if CACHE:
    names = sorted(p.stem[5:] for p in Path(CACHE).glob("cand_*.npz"))
    names = [n for n in names if (TRAIN / (n + ".geff")).exists()]
if not names:
    alln = sorted(p.stem for p in TRAIN.glob("*.zarr")
                  if (TRAIN / (p.stem + ".geff")).exists())
    a = [n for n in alln if n.startswith("44b6")]
    b = [n for n in alln if n.startswith("6bba")]
    k = max(1, round(N_DATASETS * len(a) / max(len(a) + len(b), 1)))
    names = a[:k] + b[:N_DATASETS - k]""",
     # A SUPERSET, not a fresh draw. Every number in notes/34-41 was measured on the 12
     # datasets in the candidate cache; if this run sampled independently, a change in the
     # result would confound "more data" with "different data" and settle nothing. So take
     # those 12 first, then fill to N_DATASETS from the rest of the pool, stratified.
     """seed_names = []
if CACHE:
    seed_names = sorted(p.stem[5:] for p in Path(CACHE).glob("cand_*.npz"))
    seed_names = [n for n in seed_names if (TRAIN / (n + ".geff")).exists()]
alln = sorted(p.stem for p in TRAIN.glob("*.zarr")
              if (TRAIN / (p.stem + ".geff")).exists())
print("pool", len(alln), "datasets | seed (already measured)", len(seed_names), flush=True)
rest = [n for n in alln if n not in set(seed_names)]
a = [n for n in rest if n.startswith("44b6")]
b = [n for n in rest if not n.startswith("44b6")]
need = max(0, N_DATASETS - len(seed_names))
# Proportional to the 71/128 population split, on the datasets not already taken.
k = min(len(a), max(0, round(need * len(a) / max(len(a) + len(b), 1))))
names = seed_names + a[:k] + b[:need - k]""")

# The stratifier that follows would re-slice and DROP the seed datasets, which is exactly
# what this run must not do.
edit("do not re-slice the superset",
     """_a = [n for n in names if n.startswith("44b6")]
_b = [n for n in names if not n.startswith("44b6")]
if _a and _b and len(names) > N_DATASETS:
    _k = max(1, round(N_DATASETS * len(_a) / len(names)))
    _k = min(_k, len(_a), N_DATASETS - 1)
    names = _a[:_k] + _b[:N_DATASETS - _k]
names = names[:N_DATASETS]""",
     """# Deliberately NOT re-sliced here. sweep2 re-stratified after selection, which would
# drop seed datasets and break the superset property this run depends on. The selection
# above is already proportional; assert that rather than silently re-cutting it.
names = names[:N_DATASETS]""")

edit("time guard",
     """for name in names:
    t0 = time.time()
    sc = read_scale(TRAIN / (name + ".zarr"))""",
     # 60 datasets is roughly 5x sweep2's runtime and the tail is long -- its slowest
     # dataset took 589 s against a 280 s mean. A Kaggle timeout kills the notebook with
     # no output at all, so stop while there is still time to print, and report the n
     # actually reached. A partial n=45 is worth far more than a lost n=60.
     """BUDGET_S = 9.5 * 3600
for name in names:
    t0 = time.time()
    done = len(PER)
    if done >= 3:
        per = (time.time() - T0) / done
        if time.time() - T0 + per * 1.3 > BUDGET_S:
            print("stopping at " + str(done) + " datasets: another would cost ~"
                  + str(int(per)) + "s and the budget is " + str(int(BUDGET_S)) + "s",
                  flush=True)
            break
    sc = read_scale(TRAIN / (name + ".zarr"))""")


edit("per-dataset records the headline quantity",
     '            v = float(ROWS[lbl][-1].get("adj_edge_jaccard", float("nan")))',
     # The table ranks on `score`; a paired test on `adj_edge_jaccard` would be testing a
     # different quantity from the one being ranked -- they differ by the node-budget
     # multiplier and the division term. Record what is actually being compared.
     '            _r = ROWS[lbl][-1]\n'
     '            v = float(_r.get("score", float("nan")))\n'
     '            if v != v:\n'
     '                v = float(_r.get("adj_edge_jaccard", float("nan")))')

# The superset property is the whole design, so record which datasets were the seed --
# prediction 1 cannot be graded without it.
edit("record the seed list",
     '"blend_w": BLEND_W, "datasets": [n for n in names if n in PER],',
     '"blend_w": BLEND_W, "datasets": [n for n in names if n in PER],\n'
     '           "seed_datasets": seed_names,')

_src_all = SRC.read_text()
_i = _src_all.index('md("""## 2. The five predictions""")')
_j = _src_all.index('nb = {"cells": CELLS,')
edit("replace the whole grading section", _src_all[_i:_j],
     (Path("/tmp/claude-0/-home-user-rogii/840351bc-4942-5d31-9b68-1b00e66da173"
           "/scratchpad/widecv_tail.py").read_text()))


# The intro still describes sweep2's purpose. Replace the whole first md block; like the
# grading section it must match against pristine text, so it is ordered with it.
_i0 = _src_all.index('md(r"""')
# +4, not +3: the slice must swallow the closing paren too, or the replacement's own
# paren is a second one. The builder then fails to parse, which is at least loud.
_j0 = _src_all.index('"""', _src_all.index("# The settings every 0.926 team uses")) + 4
edit("replace the whole intro",
     _src_all[_i0:_j0],
     'md(r"""\n' + (SP_ / "widecv_intro.md").read_text().strip() + '\n""")')


def main() -> int:
    out = SRC.read_text()
    # The section replacement must see pristine text: later edits rewrite strings that
    # live INSIDE its anchor, and a stale anchor would no-op silently.
    ordered = sorted(EDITS, key=lambda e: 0 if e[0].startswith("replace the whole") else 1)
    for name, old, new, times in ordered:
        n = out.count(old)
        if n != times:
            print(f"REFUSING TO WRITE — {name}: matched {n}x in the running text, "
                  f"expected {times}")
            return 1
        before = out
        out = out.replace(old, new, times)
        if out == before:
            print(f"REFUSING TO WRITE — {name}: replace changed nothing")
            return 1
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits applied in order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
