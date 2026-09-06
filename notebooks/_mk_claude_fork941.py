"""Build fork notebooks for the two public 0.941 kernels — `notes/64` step 2.

    python notebooks/_mk_claude_fork941.py            # both arms
    python notebooks/_mk_claude_fork941.py lb941      # one arm

`notes/64`: we forked `nusrati/0-938` and scored 0.937, and while we spent four GPU runs
sweeping one of its constants the lineage published 0.940 and then two independent 0.941s.
The board moved with it — 0.937 was rank ~330 on 2026-09-04 and is rank 466 today, and rank
100 now costs 0.942. Forking the current head is the cheapest move on the table.

**Unmodified is the point.** `notes/61` reproduced `nusrati/0-938` at −0.001 of its claimed
score, and every modification since has lost: `claude_fork2` (division gates at measured GT
p95) and `claude_forkw085` (0.932, `notes/64` §1). Two arms, no edits, both attribution-only.

    lb941     analyticaobscura/biohub-lb-941                 76 votes, 10 cells
              Its own BIOHUB_SCORE_AXIS reads `public 0.940 base + {DEEPCENTER_SAFE_DIV
              _THRESHOLD: 0.25, GAP_CLOSE_UM: 5.0}`, and reyhanksatria's independent 0.941
              publishes the identical two-step progression as a table. Corroborated twice.

    adaptive  rishabhr0y/941-biohub-fresh-adaptive-assoc     26 votes, 12 cells
              The other route: `SECONDARY_LINK_MODE=adaptive`, and it does NOT set the two
              knobs above. If both land at 0.941 they are separate mechanisms, and the
              combination nobody has run becomes the honest place to look for the +0.001
              that separates rank 115 from rank 100.

Provenance for each is written beside its notebook so a rebuild needs no network call, the
same contract as `claude_fork_source.json`.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

HERE = Path(__file__).resolve().parent

ARMS = {
    "lb941": {
        "user": "analyticaobscura", "slug": "biohub-lb-941",
        "why": ("public 0.940 base + DEEPCENTER_SAFE_DIV_THRESHOLD 0.25 + GAP_CLOSE_UM 5.0\n"
                "#               (its own BIOHUB_SCORE_AXIS says so; reyhanksatria's\n"
                "#               independent 0.941 publishes the same two-step table)"),
    },
    "adaptive": {
        "user": "rishabhr0y", "slug": "941-biohub-fresh-adaptive-assoc",
        "why": ("SECONDARY_LINK_MODE = adaptive, reaching 0.941 WITHOUT the deepcenter\n"
                "#               threshold or the narrower gap radius the other 0.941 uses"),
    },
}

ATTRIBUTION = """\
# ==========================================================================
# FORK -- NOT OUR WORK, and run entirely unmodified.
#
#   source      https://www.kaggle.com/code/{user}/{slug}   (public, claimed LB 0.941)
#   route       {why}
#
# All credit to that author and to the lineage behind it (pilkwang's three
# models; stephennedumpally -> nusrati 0.936 -> 0.938 -> 0.940 -> here).
#
# Why unmodified: notes/61 reproduced nusrati/0-938 at 0.937, one thousandth
# under its claimed score, so this lineage transfers. Every edit we have made
# to it since has LOST -- claude_fork2 (division gates at measured GT p95) and
# claude_forkw085 (SECONDARY_DETECTION_WEIGHT 0.85, LB 0.932 against the
# fork's 0.937, notes/64). This run measures the base and nothing else.
#
# notes/64: the board moved while we swept. 0.937 was rank ~330 on 2026-09-04
# and is rank 466 on 2026-09-06; rank 100 now costs 0.942. A clean 0.941 is
# rank ~115 -- a floor to stand on, not the finish.
#
# Not the metric-hack lineage: that surface was patched and the leaderboard
# recalculated (thread 736937), so those displayed scores are stale.
# ==========================================================================
"""


def build(name: str) -> int:
    arm = ARMS[name]
    user, slug = arm["user"], arm["slug"]
    out = HERE / f"claude_fork941_{name}.ipynb"
    prov = HERE / f"claude_fork941_{name}_source.json"

    blob = K.get_json("/kernels/pull", userName=user, kernelSlug=slug)
    md, src = blob.get("metadata", {}), blob["blob"]["source"]
    sources = md.get("datasetDataSources") or []
    if not any("tracking-support-pack" in s for s in sources):
        print(f"REFUSING TO WRITE — {name}: sources lack the support pack; got {sources}")
        return 1

    prov.write_text(json.dumps({"user": user, "slug": slug,
                                "datasetDataSources": sources,
                                "kernelDataSources": md.get("kernelDataSources"),
                                "enableGpu": md.get("enableGpu"),
                                "enableInternet": md.get("enableInternet"),
                                "currentVersionNumber": md.get("currentVersionNumber"),
                                "source": src}, indent=1))

    nb = json.loads(src)
    cells = nb["cells"]
    # Prepend to the first CODE cell rather than inserting a new one: a fresh cell would
    # shift execution order in a notebook whose early cell sets the env vars everything
    # reads. Markdown lead-ins are fine to skip past; a markdown cell 0 is not the target.
    idx = next((i for i, c in enumerate(cells) if c.get("cell_type") == "code"), None)
    if idx is None:
        print(f"REFUSING TO WRITE — {name}: no code cell found")
        return 1
    if "os.environ" not in "".join(cells[idx]["source"]):
        print(f"REFUSING TO WRITE — {name}: first code cell sets no env vars, "
              f"so this is not the config cell we think it is")
        return 1
    head = ATTRIBUTION.format(user=user, slug=slug, why=arm["why"])
    cells[idx]["source"] = (head + "".join(cells[idx]["source"])).splitlines(keepends=True)

    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out.name}: {len(cells)} cells, {len(src):,} chars from {user}/{slug} "
          f"v{md.get('currentVersionNumber')}, attribution on code cell {idx}")
    print(f"  gpu={md.get('enableGpu')}  net={md.get('enableInternet')}")
    print(f"  sources: {sources}")
    return 0


def main(argv) -> int:
    names = argv or list(ARMS)
    for n in names:
        if n not in ARMS:
            print(f"unknown arm {n!r}; have {list(ARMS)}")
            return 1
        if build(n):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
