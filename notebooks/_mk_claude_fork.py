"""Build notebooks/claude_fork.ipynb — Track A of the top-100 plan, the safety floor.

notes/54: our reproduction of the pack scored LB 0.867 while the cluster ran the SAME
weights at 0.913-0.916. We reproduced their model, never their pipeline, and every notebook
in today's 0.936 band mounts three models where we mount one. Track A banks a score near
the target immediately so the 2026-09-29 deadline is not a cliff; Track B ports the
three-model configuration into our own pipeline as the real build.

Base: `nusrati/0-938`, public, post-metric-fix, mounting all three pilkwang models. Its own
lineage, from the headers: `stephennedumpally/pls-upvote-share-higher-scoring-ideas`
(LB 0.931) -> `nusrati/0-936` (SECONDARY_DETECTION_WEIGHT 0.475 -> 0.80) -> `nusrati/0-938`
(division geometry: SAFE_DIV_MAX_UM 7->9, SISTER_MAX_UM 12->14, plus symmetry and
divergence gates).

This script pulls that notebook and prepends an attribution cell. It does not modify the
science -- a fork run unchanged is the only thing that tests whether the public number
transfers to our account, which is Track A's entire purpose. The pulled source is written
beside the notebook for provenance so the fork is auditable without a network call.

NOT the metric-hack lineage. notes/54 §1: that surface is patched, the leaderboard was
recalculated, and those notebooks display stale pre-fix scores (thread 736937).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "claude_fork.ipynb"
PROV = HERE / "claude_fork_source.json"
USER, SLUG = "nusrati", "0-938"

ATTRIBUTION = f"""\
# ==========================================================================
# FORK -- NOT OUR WORK. Run unmodified as Track A of notes/54's plan.
#
#   source      https://www.kaggle.com/code/{USER}/{SLUG}   (public)
#   lineage     stephennedumpally/pls-upvote-share-higher-scoring-ideas  LB 0.931
#                 -> nusrati/0-936   SECONDARY_DETECTION_WEIGHT 0.475 -> 0.80
#                 -> nusrati/0-938   division geometry (SAFE_DIV_MAX_UM 7->9,
#                                    SISTER_MAX_UM 12->14, symmetry + divergence gates)
#
# All credit to those authors. Reproduced here to measure whether the public
# score transfers to this account, and to diff their three-model pipeline
# against ours -- notes/54: they mount pack + temporal-seed314159 + deepcenter,
# we mount pack alone, and that is the ~0.035 gap.
#
# Not the metric-hack lineage: that surface was patched and the leaderboard
# recalculated (thread 736937), so those displayed scores are stale.
# ==========================================================================
"""


def main() -> int:
    blob = K.get_json("/kernels/pull", userName=USER, kernelSlug=SLUG)
    md, src = blob.get("metadata", {}), blob["blob"]["source"]
    PROV.write_text(json.dumps({"user": USER, "slug": SLUG,
                                "datasetDataSources": md.get("datasetDataSources"),
                                "kernelDataSources": md.get("kernelDataSources"),
                                "enableGpu": md.get("enableGpu"),
                                "currentVersionNumber": md.get("currentVersionNumber"),
                                "source": src}, indent=1))
    nb = json.loads(src)
    cells = nb["cells"]
    if not cells or cells[0].get("cell_type") != "code":
        print("REFUSING TO WRITE — first cell is not code; attribution would not run first")
        return 1
    # Prepend to the FIRST code cell rather than adding a new one: a fresh cell would
    # shift execution order in a notebook whose cell 3 sets the env vars everything reads.
    first = "".join(cells[0]["source"])
    cells[0]["source"] = (ATTRIBUTION + first).splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT.name}: {len(cells)} cells, {len(src):,} chars from {USER}/{SLUG} "
          f"v{md.get('currentVersionNumber')}")
    print(f"  sources: {md.get('datasetDataSources')}")
    print(f"  gpu: {md.get('enableGpu')}   provenance: {PROV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
