"""Generic arm builder: fork a public kernel, optionally with verified env-var edits.

    python notebooks/_mk_claude_arm.py                 # build every arm in ARMS
    python notebooks/_mk_claude_arm.py ckpt948 lb941last

`notes/64` set the working mode for the rest of the competition: the leaderboard is the only
instrument we trust (PROXY misled once, pooled training folds misled once), submission slots
are not scarce (~115 left against ~140 GPU-hours), and the public frontier moves about
+0.001/day. That means many small forks, each changing ONE thing against a base whose score
we know, and no more train-side screening.

Writing a bespoke `_mk_claude_*.py` per arm — there are eleven of them in this directory —
stopped making sense at that cadence. This is the same contract in one place:

* **Provenance is cached.** Every pull is written to `claude_arm_<name>_source.json`, so a
  rebuild needs no network call and the exact bytes we forked stay auditable.
* **Edits are verified or refused.** Each `(old, new)` pair must match exactly once across
  all code cells. `notes/45` and the `claude_divsweep` builder both record the failure this
  prevents: a builder that runs, exits 0, and writes something subtly different.
* **Attribution goes on the first code cell that sets `BIOHUB_*`,** never in a new cell —
  inserting a cell shifts execution order in notebooks whose config cell is read by
  everything after it.

Registry entries carry `base` (the public kernel), `edits`, and `why` (what the arm tests,
which is what gets written into the header and read back when the score lands).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

HERE = Path(__file__).resolve().parent

DC = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center"


def env(key: str, val: str) -> str:
    return f'os.environ["BIOHUB_{key}"] = "{val}"'


ARMS = {
    # ---------------------------------------------------------------- checkpoint_last
    # Mining all 56 top public kernels for their BIOHUB_* config turned up a branch nobody
    # in the 0.938->0.941 lineage has touched: nine kernels point DEEPCENTER_CHECKPOINT at
    # `checkpoint_last.pt` instead of `best.pt`, and one of them claims **0.948** — rank ~20
    # on today's board against rank ~115 for a clean 0.941. Posted 2026-09-01, after the
    # 2026-08-22 metric fix (topic 736937), so it is not a pre-fix ghost score.
    #
    # It is a WEIGHTS change, not a hyperparameter, which is why it is worth two arms: the
    # 0.948 kernel also reverts the division config to the old narrow 7.0/12.0, so forking it
    # alone confounds the checkpoint with that revert.
    "ckpt948": {
        "base": ("zhuzhenghaomax", "biohub-0-948-reproduction-20260901"),
        "edits": [],
        "why": ("claimed LB 0.948 -- the checkpoint_last.pt branch, unmodified.\n"
                "#               Differs from the 0.941 line in TWO ways at once: the DeepCenter\n"
                "#               checkpoint AND a revert to the narrow division config\n"
                "#               (SAFE_DIV_MAX_UM 7.0, SISTER_MAX_UM 12.0, no divergence or\n"
                "#               symmetry gate). This arm tests the claim; lb941last separates it."),
    },
    # The same lever placed on the base we trust. If checkpoint_last is what buys the 0.948
    # then it should also pay on top of the corroborated 0.941 config; if instead it only
    # works with the narrow division settings, these two arms disagree and that is the finding.
    "lb941last": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("DEEPCENTER_CHECKPOINT", f"{DC}/best.pt"),
                   env("DEEPCENTER_CHECKPOINT", f"{DC}/checkpoint_last.pt"))],
        "why": ("the 0.941 config with ONE change: DeepCenter best.pt -> checkpoint_last.pt.\n"
                "#               Isolates the checkpoint from the narrow-division revert that\n"
                "#               ships alongside it in the 0.948 kernel."),
    },
}

ATTRIBUTION = """\
# ==========================================================================
# FORK of a public kernel -- NOT OUR WORK.
#
#   source   https://www.kaggle.com/code/{user}/{slug}
#   arm      {why}
#
# All credit to that author and to the lineage behind it (pilkwang's three
# models; stephennedumpally -> nusrati 0.936 -> 0.938 -> 0.940 -> the 0.941s).
#
# Our edits below this header, if any:
{edits}
#
# notes/64: the leaderboard is the only instrument we trust here. The fork's
# own PROXY_SCORE said +0.0017 for SECONDARY_DETECTION_WEIGHT 0.85 and the
# board said -0.005, which is the second train-side screen to point the wrong
# way. One change per arm, measured where we are scored.
# ==========================================================================
"""


def build(name: str, refresh: bool = False) -> int:
    arm = ARMS[name]
    user, slug = arm["base"]
    prov = HERE / f"claude_arm_{name}_source.json"
    out = HERE / f"claude_arm_{name}.ipynb"

    if prov.exists() and not refresh:
        rec = json.loads(prov.read_text())
    else:
        blob = K.get_json("/kernels/pull", userName=user, kernelSlug=slug)
        md = blob.get("metadata", {})
        rec = {"user": user, "slug": slug,
               "datasetDataSources": md.get("datasetDataSources") or [],
               "kernelDataSources": md.get("kernelDataSources"),
               "enableGpu": md.get("enableGpu"),
               "enableInternet": md.get("enableInternet"),
               "currentVersionNumber": md.get("currentVersionNumber"),
               "source": blob["blob"]["source"]}
        prov.write_text(json.dumps(rec, indent=1))

    sources = rec["datasetDataSources"]
    need = ("tracking-support-pack", "temporal-unet3d-seed314159",
            "deepcenter-unet3d-center-prior")
    missing = [n for n in need if not any(n in s for s in sources)]
    if missing:
        print(f"REFUSING TO WRITE — {name}: sources lack {missing}; got {sources}")
        return 1

    nb = json.loads(rec["source"])
    cells = nb["cells"]

    applied = []
    for old, new in arm["edits"]:
        hits = [i for i, c in enumerate(cells)
                if c.get("cell_type") == "code" and old in "".join(c["source"])]
        total = sum("".join(cells[i]["source"]).count(old) for i in hits)
        if total != 1:
            print(f"REFUSING TO WRITE — {name}: matched {total}x across {len(hits)} cells, "
                  f"expected exactly 1\n  looking for: {old}")
            return 1
        i = hits[0]
        cells[i]["source"] = "".join(cells[i]["source"]).replace(old, new, 1) \
            .splitlines(keepends=True)
        applied.append((old, new))

    idx = next((i for i, c in enumerate(cells)
                if c.get("cell_type") == "code" and "os.environ" in "".join(c["source"])),
               None)
    if idx is None:
        print(f"REFUSING TO WRITE — {name}: no code cell sets BIOHUB_* env vars")
        return 1

    if applied:
        lines = "\n".join(f"#     {o.split('=')[-1].strip()} -> {n.split('=')[-1].strip()}"
                          f"   ({o.split('[')[1].split(']')[0].strip(chr(34))})"
                          for o, n in applied)
    else:
        lines = "#     none -- run unmodified"
    head = ATTRIBUTION.format(user=user, slug=slug, why=arm["why"], edits=lines)
    cells[idx]["source"] = (head + "".join(cells[idx]["source"])).splitlines(keepends=True)

    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out.name}: {len(cells)} cells from {user}/{slug} "
          f"v{rec['currentVersionNumber']}, {len(applied)} edit(s), header on cell {idx}")
    for o, n in applied:
        print(f"  {o}\n    -> {n}")
    return 0


def main(argv) -> int:
    refresh = "--refresh" in argv
    names = [a for a in argv if not a.startswith("-")] or list(ARMS)
    for n in names:
        if n not in ARMS:
            print(f"unknown arm {n!r}; have {list(ARMS)}")
            return 1
        if build(n, refresh=refresh):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
