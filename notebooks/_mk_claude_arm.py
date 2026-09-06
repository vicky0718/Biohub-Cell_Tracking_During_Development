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


# The P100 escape hatch. Kaggle handed this account eight consecutive Tesla P100s on
# 2026-09-06 and `machineShape` is ignored on push, so re-rolling the draw (tools/run_arm.py)
# may not terminate. A P100 is sm_60 and the image's torch builds sm_70+, so the fix is the
# torch our own runs have always used: 2.5.1+cu121 out of `claude-torch-wheelhouse`, which
# does ship sm_60 kernels. MEMORY.md: "the only thing that makes a P100 run".
#
# It goes at the very top of the config cell so it lands before ANY torch import, and it
# targets `/usr/bin/python3` explicitly because the fork runs its detector as a subprocess
# under that interpreter, not under the notebook's `sys.executable`.
WHEELHOUSE = """\
# --- P100 escape hatch (ours) -------------------------------------------------
# Installs torch 2.5.1+cu121 from a mounted wheelhouse when this session drew a
# Tesla P100 (sm_60), which the image torch cannot run. No-op on a T4.
import subprocess as _sp, sys as _sys, pathlib as _pl
# Found by globbing, not by a hardcoded mount path. v1 hardcoded
# /kaggle/input/claude-torch-wheelhouse/wheels, the directory was not there, and the
# `is_dir()` guard turned that into a silent skip -- MEMORY.md's recurring bug class,
# silent-pass-on-missing-input, reproduced exactly. Now a P100 with no wheel is loud.
_wheels = next((p.parent for p in
                _pl.Path("/kaggle/input").glob("*/**/torch-*.whl")), None)
try:
    _name = _sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=60).stdout
except Exception:
    _name = ""
if "P100" not in _name:
    print(f"GPU {_name.strip()!r} -- no torch replacement needed", flush=True)
elif _wheels is None:
    print("P100 AND NO WHEELHOUSE -- this run will die in the first forward pass.",
          flush=True)
    for _p in sorted(_pl.Path("/kaggle/input").glob("*")):
        print("   mounted:", _p.name, flush=True)
else:
    print(f"P100 detected -- installing torch 2.5.1+cu121 from {_wheels}", flush=True)
    # No --no-deps: torch 2.5.1 needs the cu121 nvidia-* runtimes, and the image ships
    # cu128 ones. The wheelhouse carries the full closure (cublas, cudnn, nccl, triton,
    # sympy ...), so let pip resolve it there. torchvision is NOT in the wheelhouse and
    # is not requested -- the fork's detector is a custom UNet3D that does not import it.
    _r = _sp.run(["/usr/bin/python3", "-m", "pip", "install", "--no-index",
                  "--find-links", str(_wheels), "--force-reinstall", "torch==2.5.1+cu121"],
                 capture_output=True, text=True, timeout=3600)
    print(f"wheelhouse install rc={_r.returncode}", flush=True)
    if _r.returncode:
        print(_r.stdout[-2000:], _r.stderr[-2000:], flush=True)
# ------------------------------------------------------------------------------
"""


ARMS = {
    # ------------------------------------------------------------------- the 0.941 floor
    # We are at 0.937, which was rank ~330 on 2026-09-04 and rank 466 on 2026-09-06 without
    # our score changing. These two are the public frontier reproduced unmodified: the
    # baseline every arm below is measured against, and on their own worth +0.004 and about
    # 350 places. Two disjoint routes to the same 0.941 (see `union`).
    "lb941": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [],
        "why": ("claimed LB 0.941, unmodified -- 77 votes, and its own BIOHUB_SCORE_AXIS\n"
                "#               reads `public 0.940 base + {DEEPCENTER_SAFE_DIV_THRESHOLD:\n"
                "#               0.25, GAP_CLOSE_UM: 5.0}`. reyhanksatria's independent 0.941\n"
                "#               publishes the identical two-step table. Corroborated twice."),
    },
    "adaptive": {
        "base": ("rishabhr0y", "941-biohub-fresh-adaptive-assoc"),
        "edits": [],
        "why": ("claimed LB 0.941, unmodified -- the other route, via SECONDARY_LINK_MODE\n"
                "#               `adaptive`, WITHOUT the deepcenter threshold or the narrower\n"
                "#               gap radius that lb941 uses to reach the same score."),
    },
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
                "#               DEPRIORITISED -- history.csv inside the artifact shows val_loss\n"
                "#               bottoming at EPOCH 2 (0.0450) and climbing monotonically to\n"
                "#               0.3167 by epoch 500. best.pt is epoch 2; checkpoint_last.pt is\n"
                "#               epoch 500, seven times worse and badly overfit to sparse labels.\n"
                "#               The config also asserts DEEPCENTER_EXPECTED_EPOCH=2 while\n"
                "#               loading it. Run it only to see whether that guard fires."),
    },
    # -------------------------------------------------------- gradient continuation
    # `notes/64` §1 cost us 0.005 by treating a SWEPT constant as an unexplored one. The
    # distinction it should have drawn, and the one these arms rest on:
    #
    #   SWEPT  -- the author says they explored both sides and this is the peak.
    #             SECONDARY_DETECTION_WEIGHT 0.80 was that, and 0.85 lost.
    #   STEPPED -- the author moved it once, the move paid, and they published and moved on.
    #             Nothing has ever been measured on the far side of the new value.
    #
    # The 0.934 -> 0.941 progression is a chain of single accepted steps: gap-close 5.8 ->
    # 5.0 (+0.001) and the DeepCenter safe-div threshold 0.12 -> 0.25 (+0.001), both
    # published as steps, neither as a sweep. Taking one more step in the same direction is
    # the cheapest unexplored move on the board, and it is unexplored precisely because
    # everyone else is forking the published value rather than continuing past it.
    "gap44": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("GAP_CLOSE_UM", "5.0"), env("GAP_CLOSE_UM", "4.4"))],
        "why": ("gap-close radius one step further down the gradient that just paid.\n"
                "#               5.8 -> 5.0 was worth +0.001 to reyhanksatria and to the lb-941\n"
                "#               base; nobody has published anything below 5.0. Note our own\n"
                "#               notes/60 swept this radius four times and only ever WIDENED it."),
    },
    # --------------------------------------------------- the author's own gate calibration
    # `claude_packcsv` read gate_threshold_metrics.csv out of the DeepCenter artifact -- a
    # threshold sweep its author ran, published inside the dataset, and which nobody in the
    # 0.934 -> 0.941 lineage cites:
    #
    #   thr    n_pred  recall  precision      f1
    #   0.10   38,272   0.900     0.0486  0.0922
    #   0.25   24,029   0.756     0.0650  0.1197   <- where the public lineage stopped
    #   0.30   18,944   0.675     0.0737  0.1329
    #   0.40    9,133   0.392     0.0888  0.1448   <- f1 maximum
    #   0.50    1,337   0.055     0.0845  0.0664   <- gate collapses
    #
    # and gate_summary.json's own note: "Use high precision thresholds as conservative
    # node-rescue gates." The public step 0.12 -> 0.25 paid; 0.40 is where the author's
    # calibration says the gate is best, and 0.50 is where it dies. This is `notes/65` §3's
    # stepped category with the author's data pointing at the next step.
    "dc40": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("DEEPCENTER_SAFE_DIV_THRESHOLD", "0.25"),
                   env("DEEPCENTER_SAFE_DIV_THRESHOLD", "0.40"))],
        "why": ("DeepCenter safe-division veto at the f1 maximum of the author's own\n"
                "#               published sweep (0.40), one step past where the public\n"
                "#               lineage stopped (0.25). Higher = stricter veto on divisions."),
    },
    # The SAME gate, read by a different consumer, and this one has never moved: every
    # public notebook in the matrix sets DEEPCENTER_GAP_THRESHOLD to 0.25 -- it was in the
    # identical column, not the varying one. Gap-closing is a much larger population than
    # divisions, so if the calibration argument is right at all it should show here first.
    "dcgap40": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("DEEPCENTER_GAP_THRESHOLD", "0.25"),
                   env("DEEPCENTER_GAP_THRESHOLD", "0.40"))],
        "why": ("DeepCenter GAP veto at the same f1 maximum. Unlike the safe-div threshold\n"
                "#               this one is 0.25 in every public notebook we mined -- moved by\n"
                "#               nobody, ever, and it gates a far larger population."),
    },
    # The combination nobody has run. rishabhr0y reaches 0.941 through SECONDARY_LINK_MODE
    # `adaptive` and does NOT set the deepcenter threshold or the narrow gap radius;
    # analyticaobscura reaches the same 0.941 through those two and keeps the default link
    # mode. Two disjoint routes to one score is exactly the shape that is worth crossing --
    # and doing it as a one-token edit on the lb-941 base keeps everything else identical.
    "union": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("SECONDARY_LINK_MODE", "low_margin_consensus"),
                   env("SECONDARY_LINK_MODE", "adaptive"))],
        "why": ("cross the two independent 0.941 routes: analyticaobscura's deepcenter 0.25\n"
                "#               + gap 5.0, with rishabhr0y's adaptive link mode on top. Neither\n"
                "#               author has the other's change."),
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
               "competitionDataSources": md.get("competitionDataSources") or [],
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
    prologue = WHEELHOUSE if arm.get("wheelhouse", True) else ""
    cells[idx]["source"] = (head + prologue
                            + "".join(cells[idx]["source"])).splitlines(keepends=True)

    out.write_text(json.dumps(nb, indent=1))

    # The push configuration travels with the arm rather than being retyped at launch --
    # MEMORY.md's loudest rule, after a retyped source list cost `claude_submit_ratio` v1.
    kernels = list(rec.get("kernelDataSources") or [])
    if arm.get("wheelhouse", True):
        kernels.append(f"{K.username()}/claude-torch-wheelhouse")
    (HERE / f"claude_arm_{name}_push.json").write_text(json.dumps({
        "slug": f"claude-arm-{name}", "title": f"Claude arm {name}",
        "notebook": str(out), "dataset_sources": sources,
        "competition_sources": rec.get("competitionDataSources") or [],
        "kernel_sources": kernels, "enable_gpu": True, "enable_internet": False,
    }, indent=1))

    print(f"wrote {out.name}: {len(cells)} cells from {user}/{slug} "
          f"v{rec['currentVersionNumber']}, {len(applied)} edit(s), header on cell {idx}"
          f"{', wheelhouse prologue' if arm.get('wheelhouse') else ''}")
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
