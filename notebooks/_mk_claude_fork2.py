"""Build notebooks/claude_fork2.ipynb — the fork, with the division gates set from OUR data.

`claude_fork` (nusrati/0-938, reproduced unmodified) scored **0.937**, rank ~320 of the live
2026-09-04 board. Rank 100 needs **0.940**, so the gap is +0.003.

`claude_divgeom` measured all 151 ground-truth divisions across the 199 training datasets
(`notes/57`). The fork's own division distance gates are still short of that distribution:

    gate                       fork    GT p90   GT p95   GT max   fork rejects
    SAFE_DIV_MAX_UM             9.0     10.05    11.78    13.53      19.9%
    SAFE_DIV_SISTER_MAX_UM     14.0     14.36    15.34    20.30      12.6%

Its own comments justify 7->9 and 12->14 with stated GT statistics that we checked: the
medians and the 12um rejection rate match, but the TAIL does not — sisters reach 20.30um,
not the 13.7 they report, and 7um rejects 53.6% of parent-daughter links rather than 25%.
Their direction is right and their destination is short.

**One coherent change: both distance gates moved to the measured p95 of real divisions.**
Nothing else is touched. The precision gates that filter what the distance gates admit --
SISTER_SYMMETRY_TAU 0.6, DIVERGE_UM 4.5, the DeepCenter safe-div veto -- stay exactly as
they are, which is what makes widening the distance side reasonable rather than reckless.

Scale check: the fork's own validator reports div_J 0.0625, contributing 0.00625 of score.
Moving div_J to ~0.09 is worth ~+0.003 -- the gap. If instead the volume caps bind
(FRAME_FRAC_CAP 0.0076, GLOBAL_FRAC_CAP 0.00375) this is a no-op, which is the same trap
notes/60 recorded for close_gaps' insertion caps and is why the fork prints its resolved
safe-div thresholds and division counts.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "claude_fork2.ipynb"
PROV = HERE / "claude_fork_source.json"

# (label, old env line fragment, new value) -- notes/57's p95 of 151 real GT divisions.
EDITS = [
    ("parent->daughter", 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"',
     'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "11.8"'),
    ("sister<->sister", 'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "14.0"',
     'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "15.3"'),
]

HEADER = """\
# ==========================================================================
# FORK of nusrati/0-938 (public) -- NOT OUR WORK, except the two lines below.
#   lineage  stephennedumpally/pls-upvote-share-higher-scoring-ideas  LB 0.931
#              -> nusrati/0-936  -> nusrati/0-938
# All credit to those authors. The unmodified reproduction scored 0.937 for us.
#
# OUR ONE CHANGE, from claude_divgeom's measurement of all 151 ground-truth
# divisions across the 199 training datasets (notes/57):
#
#     SAFE_DIV_MAX_UM         9.0 -> 11.8    GT p95 11.78 (9.0 rejects 19.9%)
#     SAFE_DIV_SISTER_MAX_UM 14.0 -> 15.3    GT p95 15.34 (14.0 rejects 12.6%)
#
# Their comments justify 7->9 and 12->14 from GT statistics. We checked those:
# medians and the 12um rejection rate match, but the tail does not -- sisters
# reach 20.30um, not the 13.7 they state. Their direction is right and their
# destination is short. The precision gates that filter what these admit
# (SISTER_SYMMETRY_TAU, DIVERGE_UM, the DeepCenter safe-div veto) are unchanged.
# ==========================================================================
"""


def main() -> int:
    if not PROV.exists():
        print(f"REFUSING TO WRITE — {PROV.name} missing; run _mk_claude_fork.py first")
        return 1
    prov = json.loads(PROV.read_text())
    nb = json.loads(prov["source"])
    cells = nb["cells"]

    applied = []
    for label, old, new in EDITS:
        hits = [(i, "".join(c["source"]).count(old)) for i, c in enumerate(cells)
                if c.get("cell_type") == "code" and old in "".join(c["source"])]
        total = sum(n for _, n in hits)
        if total != 1:
            print(f"REFUSING TO WRITE — {label}: matched {total}x across {len(hits)} cells, "
                  f"expected exactly 1.\n  looking for: {old}")
            return 1
        i = hits[0][0]
        src = "".join(cells[i]["source"]).replace(old, new, 1)
        cells[i]["source"] = src.splitlines(keepends=True)
        applied.append((label, old.split("=")[-1].strip(), new.split("=")[-1].strip()))

    if not cells or cells[0].get("cell_type") != "code":
        print("REFUSING TO WRITE — first cell is not code; the header would not run first")
        return 1
    first = "".join(cells[0]["source"])
    cells[0]["source"] = (HEADER + first).splitlines(keepends=True)

    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT.name}: {len(cells)} cells, {len(applied)} gate change(s)")
    for label, o, n in applied:
        print(f"  {label:<18} {o} -> {n}")
    print(f"  sources: {prov['datasetDataSources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
