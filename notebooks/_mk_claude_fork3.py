"""Build notebooks/claude_fork3.ipynb — the fork, with our measured smoothing fix.

`claude_static` swept `linefit_smooth`'s new `static_um` on cached graphs and the result is
monotone to its endpoint:

    s0.0  0.9188 (anchor)   s2.5  0.9206   s5.0       0.9212
    s1.8  0.9197            s3.5  0.9211   mean_only  0.9217   <- best, +0.0030

**The line fit is never better than a plain window mean**, and both embryos agree
(44b6 +0.0012, 6bba +0.0032). notes/26 and notes/27 credited `linefit_smooth` with +0.0086
of the repair chain's +0.0113 -- the major half -- and neither ever compared the fitted
slope against a degree-0 average.

The fork runs the identical mechanism in `linefit_smooth_output_graph`:

    fitted = np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0)

Degree **1** is the line. Degree **0** is the window mean -- exactly the `mean_only` arm.
So our finding transfers as a single character, and +0.0030 is precisely the 0.937 -> 0.940
gap to rank 100 on the live board.

Screened on the fork's own PROXY_SCORE (0.9266 for the unmodified run) before any
submission. notes/49 stands: PROXY runs on training embryos and the test set is a third
pair, so it screens direction rather than magnitude.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "claude_fork3.ipynb"
PROV = HERE / "claude_fork_source.json"

OLD = ("        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) "
       "for axis in range(3)], dtype=np.float64)")
NEW = ("        # DEGREE 0, NOT 1 -- the window mean, not a line fit. claude_static swept this\n"
       "        # on cached graphs and the mean wins monotonically: 0.9188 (line) -> 0.9217\n"
       "        # (mean), +0.0030, with both embryos agreeing. The fitted slope is noise.\n"
       "        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 0), 0.0) "
       "for axis in range(3)], dtype=np.float64)")

HEADER = """\
# ==========================================================================
# FORK of nusrati/0-938 (public) -- NOT OUR WORK, except the one line below.
#   lineage  stephennedumpally/pls-upvote-share-higher-scoring-ideas  LB 0.931
#              -> nusrati/0-936 -> nusrati/0-938
# All credit to those authors. The unmodified reproduction scored 0.937 (PROXY 0.9266).
#
# OUR ONE CHANGE, in linefit_smooth_output_graph:
#     np.polyfit(dts, coords[:, axis], 1)   ->   ..., 0)
#
# Degree 1 fits a LINE through the local track window; degree 0 takes its MEAN.
# claude_static swept exactly this on our own cached graphs and the mean wins
# monotonically all the way to the endpoint:
#     line 0.9188 -> 0.9197 -> 0.9206 -> 0.9211 -> 0.9212 -> mean 0.9217
# +0.0030, both embryos agreeing (44b6 +0.0012, 6bba +0.0032).
#
# The finding came from notes/59: 8.4% of ground-truth links have EXACTLY zero
# displacement (frozen frames, interpolated annotation), so a slope fitted
# through detection jitter is noise -- and it turned out to be noise generally,
# not only on frozen chains.
# ==========================================================================
"""


def main() -> int:
    prov = json.loads(PROV.read_text())
    nb = json.loads(prov["source"])
    cells = nb["cells"]

    hits = [i for i, c in enumerate(cells)
            if c.get("cell_type") == "code" and OLD in "".join(c["source"])]
    total = sum("".join(cells[i]["source"]).count(OLD) for i in hits)
    if total != 1:
        print(f"REFUSING TO WRITE — polyfit line matched {total}x, expected exactly 1")
        return 1
    # No other degree-1 polyfit may exist, or we would be changing only one of several.
    allsrc = "\n".join("".join(c["source"]) for c in cells)
    others = allsrc.count("np.polyfit(")
    if others != 1:
        print(f"REFUSING TO WRITE — {others} np.polyfit call sites, expected exactly 1")
        return 1

    i = hits[0]
    cells[i]["source"] = "".join(cells[i]["source"]).replace(OLD, NEW, 1) \
        .splitlines(keepends=True)
    if cells[0].get("cell_type") != "code":
        print("REFUSING TO WRITE — first cell is not code")
        return 1
    cells[0]["source"] = (HEADER + "".join(cells[0]["source"])).splitlines(keepends=True)

    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT.name}: polyfit degree 1 -> 0, {len(cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
