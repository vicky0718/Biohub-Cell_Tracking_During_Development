"""Generic arm builder: fork a public kernel, optionally with verified env-var edits.

    python notebooks/_mk_claude_arm.py                 # build every arm in ARMS
    python notebooks/_mk_claude_arm.py dc40 dcgap40

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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

HERE = Path(__file__).resolve().parent

DC = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center"

# The three mounts every arm on reyhanksatria's 0.946 base needs, and only those.
#
# v1 of this list also named the author's own re-hosted copies --
# `reyhanksatria/biohub-tracking-support-pack` and two siblings -- because
# `/kernels/pull` returns their `datasetDataSources` as `['', '', '']` and there was no way
# to tell from provenance which copies they had used. Listing both was meant as insurance:
# an extra mount costs nothing, a missing one is fatal.
#
# It was insurance against nothing. `/kernels/pull` on our own pushed kernel shows Kaggle
# resolved the author's three straight back to `['', '', '']` -- they are not public, so
# nothing attached -- and every artifact path in the runtime log resolves under `pilkwang/`:
#
#     ARTIFACTS:          /kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1
#     Secondary artifact: /kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1
#
# So the arms have always run on pilkwang's public CC0 datasets alone. Three phantom entries
# on the notebook page suggested it depended on data nobody can see, which is worse than
# useless, and they are gone.
TTA946_SOURCES = [
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
]


def env(key: str, val: str) -> str:
    return f'os.environ["BIOHUB_{key}"] = "{val}"'


def guard(key: str, val: str) -> str:
    """A line of the notebook's own `_EXPECTED_NUMERIC` configuration-drift guard.

    `claude-arm-gap44` died two minutes in on
    ``RuntimeError: Configuration drift detected: {"BIOHUB_GAP_CLOSE_UM":
    {"actual": 4.4, "expected": 5.0}}``. The lb-941 notebook carries a guard cell that
    re-reads six numeric and two text env vars and refuses to run if any differs from a
    hardcoded expectation -- the author protecting their "single intended model-level
    change" from exactly the kind of edit we are making.

    Guarded numerics: DET_THRESHOLD, ILP_APPEARANCE_WEIGHT, ILP_DISAPPEARANCE_WEIGHT,
    GAP_CLOSE_UM, OUTPUT_MIN_TRACK_LEN, BIDIRECTIONAL_EDGE_WEIGHT. Guarded text:
    BIDIRECTIONAL_FUSION_MODE, DUAL_SEED_MIN_CANDIDATE_RETENTION. Everything else --
    including the deepcenter thresholds, the divergence gate, the link mode and both
    secondary weights -- is unguarded, which is why dc40, div15, sew20 and union all ran.

    An arm touching a guarded key must move the guard with it, as a second verified
    replacement, or the run dies before the detector loads.
    """
    return f'    "BIOHUB_{key}": {val},'


def dominance_edits() -> list[tuple[str, str]]:
    """Port rishabhr0y's motion-relink confidence dominance onto our base.

    Every notebook in this lineage ends its edge stage with

        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
        if motion_edges:
            stats['motion_relink_replaced_raw_edges'] = len(edges)
            edges = motion_edges          # <- the entire ILP edge set, discarded

    and our own `run_stats` shows what that costs: `motion_relink_replaced_raw_edges` is
    67,249 on `6bba_05db0fb1`, i.e. **the motion model overwrites essentially every edge the
    ILP produced.** Not some. All of them.

    `rishabhr0y` -- rank 108 at 0.946, four ranks above `analyticaobscura`, whose published
    notebooks we have been forking all week -- replaces that with a reconciliation: a raw ILP
    edge survives if it *strictly beats every conflicting motion edge at both of its
    endpoints*. Relative confidence only, no absolute threshold, strongest first so the graph
    stays one-parent/one-child, and a sparsity guard that abandons the whole reconciliation on
    a frame group with too few dominant candidates.

    Their own naming prices it: the sibling notebook is
    `biohub-edge-density-adaptive-on-0943`, built *on* the result of this one, and the base
    both share is the 0.941 config. **0.941 -> 0.943.**

    Why this and not another knob: it is orthogonal to everything we have measured. `ttasec`
    changes what the association model *sees*; this changes what happens to the association
    model's output when the motion model disagrees with it. Nothing in the TTA family touches
    the ILP-versus-motion arbitration, and nothing in the 0.946 lineage does either.

    Three edits. Their notebook is written with double quotes and ours with single, and it
    carries a `division_reference_motion_edges` line ours does not, so this is a translation
    rather than a copy -- which is the whole reason it is quoted in full here instead of
    lifted by reference.
    """
    # v1 of this port died on `KeyError: 'motion_relink_confidence_restored'` after a full
    # prediction pass. `filter_output_graph` builds `stats` as a PLAIN dict with an explicit
    # key list -- not a Counter, not a defaultdict -- so the two counters the block increments
    # with `+=` have to exist first. rishabhr0y's notebook initialises them in its own
    # version of that dict; ours does not, and the block alone is not the whole port.
    # The other three counters here are plain assignments and need no initialiser.
    stats_old = "stats = {'raw_edges': len(raw_edges), 'dropped_nonconsecutive_edges': 0,"
    stats_new = ("stats = {'raw_edges': len(raw_edges), "
                 "'motion_relink_confidence_restored': 0, "
                 "'motion_relink_confidence_displaced': 0, "
                 "'dropped_nonconsecutive_edges': 0,")

    const_old = ("MOTION_RELINK_LEARNED_BONUS = float(os.environ.get("
                 "'BIOHUB_MOTION_RELINK_LEARNED_BONUS', '0.75'))")
    const_new = (const_old + "\n"
                 "MOTION_RELINK_CONFIDENCE_DOMINANCE = (\n"
                 "    os.environ.get('BIOHUB_MOTION_RELINK_CONFIDENCE_DOMINANCE', '0') != '0'\n"
                 ")\n"
                 "MOTION_RELINK_DOMINANCE_MIN_PER_FRAME = float(\n"
                 "    os.environ.get('BIOHUB_MOTION_RELINK_DOMINANCE_MIN_PER_FRAME', '1.0')\n"
                 ")")

    block_old = (
        "        if motion_edges:\n"
        "            stats['motion_relink_replaced_raw_edges'] = len(edges)\n"
        "            edges = motion_edges\n"
        "        else:\n"
        "            stats['motion_relink_fallback_raw'] = 1\n")
    block_new = (
        "        if motion_edges:\n"
        "            stats['motion_relink_replaced_raw_edges'] = len(edges)\n"
        "\n"
        "            if MOTION_RELINK_CONFIDENCE_DOMINANCE:\n"
        "                motion_by_source: dict[int, list[dict[str, object]]] = {}\n"
        "                motion_by_target: dict[int, list[dict[str, object]]] = {}\n"
        "\n"
        "                for motion_edge in motion_edges:\n"
        "                    motion_by_source.setdefault("
        "int(motion_edge['source_id']), []).append(motion_edge)\n"
        "                    motion_by_target.setdefault("
        "int(motion_edge['target_id']), []).append(motion_edge)\n"
        "\n"
        "                def _finite_prob(edge: dict[str, object]) -> float:\n"
        "                    try:\n"
        "                        value = float(edge.get('edge_prob', 0.0))\n"
        "                    except (TypeError, ValueError):\n"
        "                        return 0.0\n"
        "                    return value if np.isfinite(value) else 0.0\n"
        "\n"
        "                dominance_candidates: list[dict[str, object]] = []\n"
        "\n"
        "                for raw_edge in edges:\n"
        "                    source_id = int(raw_edge['source_id'])\n"
        "                    target_id = int(raw_edge['target_id'])\n"
        "                    conflicts = [edge for edge in (motion_by_source.get(source_id, [])"
        " + motion_by_target.get(target_id, []))"
        " if (int(edge['source_id']), int(edge['target_id'])) != (source_id, target_id)]\n"
        "\n"
        "                    if not conflicts:\n"
        "                        continue\n"
        "                    raw_prob = _finite_prob(raw_edge)\n"
        "\n"
        "                    if raw_prob > max(_finite_prob(edge) for edge in conflicts):\n"
        "                        dominance_candidates.append(raw_edge)\n"
        "                stats['motion_relink_confidence_candidates'] = len(dominance_candidates)\n"
        "                dominance_minimum = int(math.ceil("
        "stats['motion_relink_frames'] * MOTION_RELINK_DOMINANCE_MIN_PER_FRAME))\n"
        "                stats['motion_relink_confidence_minimum'] = dominance_minimum\n"
        "\n"
        "                if len(dominance_candidates) < dominance_minimum:\n"
        "                    stats['motion_relink_confidence_sparse_group_skipped'] = 1\n"
        "                    dominance_candidates = []\n"
        "                reconciled = list(motion_edges)\n"
        "\n"
        "                for raw_edge in sorted("
        "dominance_candidates, key = _finite_prob, reverse = True):\n"
        "                    source_id = int(raw_edge['source_id'])\n"
        "                    target_id = int(raw_edge['target_id'])\n"
        "                    live_conflicts = [edge for edge in reconciled"
        " if (int(edge['source_id']) == source_id"
        " or int(edge['target_id']) == target_id)"
        " and (int(edge['source_id']), int(edge['target_id'])) != (source_id, target_id)]\n"
        "\n"
        "                    if not live_conflicts:\n"
        "                        continue\n"
        "                    raw_prob = _finite_prob(raw_edge)\n"
        "\n"
        "                    if raw_prob <= max(_finite_prob(edge) for edge in live_conflicts):\n"
        "                        continue\n"
        "                    reconciled = [edge for edge in reconciled"
        " if edge not in live_conflicts]\n"
        "\n"
        "                    if not any(int(edge['source_id']) == source_id"
        " and int(edge['target_id']) == target_id for edge in reconciled):\n"
        "                        reconciled.append(dict(raw_edge, confidence_dominance = 1))\n"
        "                    stats['motion_relink_confidence_restored'] += 1\n"
        "                    stats['motion_relink_confidence_displaced'] += len(live_conflicts)\n"
        "                edges = reconciled\n"
        "            else:\n"
        "                edges = motion_edges\n"
        "        else:\n"
        "            stats['motion_relink_fallback_raw'] = 1\n")

    enable = ("os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'\n"
              "os.environ['BIOHUB_SECONDARY_EDGE_FEATURE_TTA'] = '1'")
    return [(stats_old, stats_new), (const_old, const_new), (block_old, block_new),
            (enable, enable + "\nos.environ['BIOHUB_MOTION_RELINK_CONFIDENCE_DOMINANCE'] = '1'")]


def z_tta_edit() -> tuple[str, str]:
    """Average the primary model over the Z-flip as well: eight views become sixteen.

    The whole public lineage runs an "eight-view D4" TTA, and every one of those eight
    transforms acts on dims (-2, -1) alone -- Y and X. Flips of Y, X and YX, two in-plane
    rot90s, a transpose, an anti-transpose. **Z is never touched.**

    The training code is in the support pack, and `scripts/augmentations.py` says what the
    model actually saw:

        def flip_augment(...):
            \"\"\"Random spatial flip: samples uniformly from all 8 axis-aligned symmetries.
            Each of Z, Y, X is independently flipped with probability 0.5.\"\"\"

    So Z-flip is a symmetry this model was **explicitly trained to be invariant to**, and it
    is the one symmetry no public notebook averages over. The reverse is also worth noting:
    the rot90 and transpose views the public TTA does use are *not* in the training set at
    all -- they work because Y and X are isotropic, not because the model learned them.
    Z-flip is the better-justified transform of the two kinds, and it is missing.

    `temporal_unet.py` states the layout as `(B, T, C_out, Z, Y, X)`, so `-3` is Z, which is
    why `.flip(-3)` is the right un-transform and why the author's `(-2, -1)` is Y and X.

    Z-flip commutes with every in-plane transform, so the sixteen-view average is the eight
    existing views plus each of them applied to `imgs.flip(-3)` and mapped back the same way
    with a trailing `.flip(-3)`. `_nv` reaches 16 and the run prints it, so an inert version
    announces itself. Cost is one extra encode per existing view -- the primary predict pass
    doubles, which is why this arm is measured on the visible clips for RUNTIME before it is
    ever proposed for a slot: the graded set is ~17x and the ceiling is 720 minutes.
    """
    old = ("    del imgs_at, det_at, _u_at\n    _nv += 1\n\n"
           "    for f in range(W):\n        det_logits[f] = det_logits[f] / _nv\n")
    new = ("    del imgs_at, det_at, _u_at\n    _nv += 1\n\n"
           "    _zbase = imgs.flip(-3)\n"
           "    _u_z, _d_z = model.encode(_zbase)\n\n"
           "    for f in range(W):\n"
           "        det_logits[f] = det_logits[f] + _d_z[f].flip(-3)\n\n"
           "    if _edge_tta:\n"
           "        _unet_acc += _u_z.flip(-3)\n"
           "    del _u_z, _d_z\n"
           "    _nv += 1\n\n"
           "    for _zd in [(-1,), (-2,), (-2, -1)]:\n"
           "        _u_z, _d_z = model.encode(_zbase.flip(_zd))\n\n"
           "        for f in range(W):\n"
           "            det_logits[f] = det_logits[f] + _d_z[f].flip(_zd).flip(-3)\n\n"
           "        if _edge_tta:\n"
           "            _unet_acc += _u_z.flip(_zd).flip(-3)\n"
           "        del _u_z, _d_z\n"
           "        _nv += 1\n\n"
           "    for _zk in (1, 3):\n"
           "        _u_z, _d_z = model.encode(torch.rot90(_zbase, _zk, dims = (-2, -1)))\n\n"
           "        for f in range(W):\n"
           "            det_logits[f] = det_logits[f] + torch.rot90(_d_z[f], -_zk,"
           " dims = (-2, -1)).flip(-3)\n\n"
           "        if _edge_tta:\n"
           "            _unet_acc += torch.rot90(_u_z, -_zk, dims = (-2, -1)).flip(-3)\n"
           "        del _u_z, _d_z\n"
           "        _nv += 1\n"
           "    _u_z, _d_z = model.encode(_zbase.transpose(-1, -2))\n\n"
           "    for f in range(W):\n"
           "        det_logits[f] = det_logits[f] + _d_z[f].transpose(-1, -2).flip(-3)\n\n"
           "    if _edge_tta:\n"
           "        _unet_acc += _u_z.transpose(-1, -2).flip(-3)\n"
           "    del _u_z, _d_z\n"
           "    _nv += 1\n"
           "    _u_z, _d_z = model.encode(torch.rot90(_zbase, 1,"
           " dims = (-2, -1)).transpose(-1, -2))\n\n"
           "    for f in range(W):\n"
           "        det_logits[f] = det_logits[f] + torch.rot90(_d_z[f].transpose(-1, -2),"
           " -1, dims = (-2, -1)).flip(-3)\n\n"
           "    if _edge_tta:\n"
           "        _unet_acc += torch.rot90(_u_z.transpose(-1, -2), -1,"
           " dims = (-2, -1)).flip(-3)\n"
           "    del _u_z, _d_z, _zbase\n"
           "    _nv += 1\n\n"
           "    for f in range(W):\n        det_logits[f] = det_logits[f] / _nv\n")
    return old, new


def sec_tta_edits() -> list[tuple[str, str]]:
    """Mirror reyhanksatria's +0.005 edge-feature TTA onto the SECONDARY model.

    Every arm so far has edited a value. This one edits the pipeline, and it is the only
    mechanism-level change this project has had a concrete reason to believe in, so the
    reasoning is written out here rather than in a note.

    The 0.946 notebook runs an eight-view D4 ensemble over the detector. `model.encode`
    returns *two* things -- association features and detection logits -- and the published
    0.933 -> 0.941 lineage averaged only the second, throwing away seven of eight feature
    maps and then pairing an eight-view detection map with a one-view feature map. The
    0.946 change is one line of bookkeeping: keep the features too. `+0.005`, no extra
    compute, because the eight forward passes were already happening.

    **The secondary model still has the bug.** Its D4 loop reads

        _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)

    -- eight encodes, eight feature maps discarded, `secondary_unet_out` left at the single
    canonical view and fed straight into `_index_features` -> `predict_edges`, whose output
    is blended at SECONDARY_EDGE_WEIGHT and decides `low_margin_consensus`. Same
    inconsistency, same free compute, on a surface no public notebook touches.

    Six replacements, each matched exactly once (validated before this was written): the
    accumulator, the four view groups, and the average. The finalizer copies the author's
    own two guards -- shape equality and a nonzero mean delta -- so an inert patch **kills
    the run** instead of quietly reproducing 0.946. `notes/68` is the whole reason that
    matters: ckpt948 spent a slot proving an env var that named a nonexistent path had been
    silently ignored.
    """
    def view(tag: str, det: str, imgs: str, acc: str,
             ind: int) -> tuple[str, str, str, str]:
        """One D4 view: capture the feature map, accumulate it, free it."""
        pad = " " * ind
        old = (f"_, {det} = secondary_model.encode({imgs})\n\n"
               f"{pad}for f in range(W):\n{pad}    ")
        new = (f"_su_{tag}, {det} = secondary_model.encode({imgs})\n\n"
               f"{pad}for f in range(W):\n{pad}    ")
        tail_old = f"\n{pad}del {imgs}, {det}\n"
        # `+=`, not the author's `x = x + y`. Their version allocates a second accumulator
        # every view; ours does not, and this block runs on top of the primary TTA's peak
        # on a 16 GB P100. One feature map either way is the difference between a run and
        # an OOM four hours in.
        tail_new = (f"\n\n{pad}if _sec_edge_tta:\n{pad}    _sec_unet_acc += {acc}"
                    f"\n{pad}del {imgs}, {det}, _su_{tag}\n")
        return old, new, tail_old, tail_new

    # The four view groups, in source order. `body` is the detection-accumulation line that
    # sits between the encode and the `del`; it is quoted verbatim so the match is exact.
    groups = [
        ("flip", "secondary_det_flip", "secondary_imgs_flip", "_su_flip.flip(dims)", 16,
         "secondary_det_logits[f] = (secondary_det_logits[f] + secondary_det_flip[f]"
         ".flip(dims))"),
        ("rot", "secondary_det_rot", "secondary_imgs_rot",
         "torch.rot90(_su_rot, -_k, dims = (-2, -1))", 16,
         "secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90("
         "secondary_det_rot[f], -_k, dims = (-2, -1))"),
        ("t", "secondary_det_t", "secondary_imgs_t", "_su_t.transpose(-1, -2)", 12,
         "secondary_det_logits[f] = (secondary_det_logits[f] + secondary_det_t[f]"
         ".transpose(-1, -2))"),
        ("at", "secondary_det_at", "secondary_imgs_at",
         "torch.rot90(_su_at.transpose(-1, -2), -1, dims = (-2, -1))", 12,
         "secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90("
         "secondary_det_at[f].transpose(-1, -2), -1, dims = (-2, -1),)"),
    ]

    edits = [(
        # The accumulator, cloned before the first extra view is added to it.
        "if cfg.det_tta:\n            _secondary_nv = 1\n",
        "if cfg.det_tta:\n            _secondary_nv = 1\n"
        "            _sec_edge_tta = os.environ.get("
        "'BIOHUB_SECONDARY_EDGE_FEATURE_TTA', '0') != '0'\n"
        "            _sec_unet_acc = secondary_unet_out.clone() if _sec_edge_tta else None\n",
    )]
    for tag, det, imgs, acc, ind, body in groups:
        o, n, to, tn = view(tag, det, imgs, acc, ind)
        edits.append((o + body + to, n + body + tn))

    edits.append((
        "            for f in range(W):\n"
        "                secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n",
        "            for f in range(W):\n"
        "                secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n"
        "\n            if _sec_edge_tta:\n"
        "                if _sec_unet_acc.shape != secondary_unet_out.shape:\n"
        "                    raise RuntimeError('Secondary edge-feature TTA shape mismatch: '"
        " + str(tuple(_sec_unet_acc.shape)) + ' vs ' + str(tuple(secondary_unet_out.shape)))\n"
        "                _sec_delta = float((_sec_unet_acc / _secondary_nv"
        " - secondary_unet_out).abs().mean())\n"
        "\n                if _sec_delta == 0.0:\n"
        "                    raise RuntimeError('Secondary edge-feature TTA produced no"
        " feature change')\n"
        "                secondary_unet_out = _sec_unet_acc / _secondary_nv\n"
        "                print('SEC_EDGE_TTA_ACTIVE views =', _secondary_nv,"
        " 'mean_abs_feat_delta =', round(_sec_delta, 6), flush = True)\n"
        "                del _sec_unet_acc\n",
    ))
    # The flag itself. This notebook quotes with ', so `env()` (which quotes with ") would
    # not match -- the builder would refuse, correctly, rather than write a dead arm.
    edits.append((
        "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'",
        "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'\n"
        "os.environ['BIOHUB_SECONDARY_EDGE_FEATURE_TTA'] = '1'",
    ))
    return edits


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
    # ---------------------------------------------------------------- checkpoint_last: RAN
    # Nine of the 56 mined kernels point DEEPCENTER_CHECKPOINT at `checkpoint_last.pt`
    # instead of `best.pt`, and one claims LB 0.948. It ran, and `notes/68` §2 is the result:
    # the variable names a path, the resolved mount is /kaggle/input/datasets/<owner>/<slug>/,
    # the path in the value does not exist, and the notebook falls back to its default --
    # best.pt, epoch 2. Its own log says so. **The whole branch loads the same weights as
    # everyone else.** With that inert, ckpt948 is DET_THRESHOLD 0.965 on an otherwise
    # 0.934-era config (SAFE_DIV_MAX_UM 7.0, SISTER 12.0, no divergence or symmetry gate,
    # deepcenter threshold 0.12) -- strictly worse than lb941, published with a better number.
    # Kept in the registry as the record; not a submission candidate.
    "ckpt948": {
        "base": ("zhuzhenghaomax", "biohub-0-948-reproduction-20260901"),
        "edits": [],
        "why": ("claimed LB 0.948, unmodified. RAN -- its checkpoint change is inert\n"
                "#               (notes/68 §2) and what remains is a 0.934-era config."),
    },
    # `lb941last` lived here: the 0.941 config with DeepCenter best.pt -> checkpoint_last.pt.
    # REMOVED, not deprioritised. `notes/68` §2: the variable names a path, the resolved mount
    # is /kaggle/input/datasets/<owner>/<slug>/..., so the path in the env value does not
    # exist and the notebook silently falls back to its own default -- which is best.pt.
    # Measured in ckpt948's own run log: "Loaded ... best.pt", "DeepCenter checkpoint epoch: 2".
    # The edit cannot take effect, so the arm cannot measure anything.
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
        "edits": [(env("GAP_CLOSE_UM", "5.0"), env("GAP_CLOSE_UM", "4.4")),
                  (guard("GAP_CLOSE_UM", "5.0"), guard("GAP_CLOSE_UM", "4.4"))],
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
    # DEMOTED. `notes/69`: the run-stats counters say the gap gate is already rejecting
    # 79% at 0.25 (checked 230, accepted 48, rejected 182) and a further 596 proposals
    # bypass it entirely on strong motion. Raising it to 0.40 can only remove the 48 that
    # survive. My "it gates a far larger population" was wrong -- 230 gap checks against
    # 356 safe-div checks, and most of the gap traffic never reaches the gate.
    "dcgap40": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("DEEPCENTER_GAP_THRESHOLD", "0.25"),
                   env("DEEPCENTER_GAP_THRESHOLD", "0.40"))],
        "why": ("DeepCenter GAP veto to the author's f1 maximum. DEMOTED -- at 0.25 the gate\n"
                "#               already rejects 79% of what reaches it and only 48 proposals\n"
                "#               survive across the four clips, so the ceiling here is small."),
    },
    # The dominant division filter, and the one the counters point at. Across the four
    # verification clips the safe-division stage sees ~1,960 geometric candidates and
    # `safe_division_divergence_rejected` alone throws out **1,604 of them** -- 82% -- against
    # 104 for the mutual-NN gate and 111 for the DeepCenter veto. `safe_division_skipped_cap`
    # is 0, so `notes/60`'s volume-cap trap is not active and the gate really is the binding
    # constraint.
    #
    # `SAFE_DIV_DIVERGE_UM` is also stepped and stopped: nusrati/0-938 ran 4.5, 0-940 moved it
    # to 2.25 as part of a +0.002, and every 0.941 inherited 2.25. Larger is stricter, so the
    # step that paid was toward MORE divisions, and nothing below 2.25 has been published.
    "div15": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("SAFE_DIV_DIVERGE_UM", "2.25"), env("SAFE_DIV_DIVERGE_UM", "1.5"))],
        "why": ("forward-divergence gate 2.25 -> 1.5, one step further in the direction that\n"
                "#               paid (4.5 -> 2.25). It rejects 82% of all division candidates,\n"
                "#               more than every other division gate combined, and the volume\n"
                "#               caps are not binding behind it (skipped_cap = 0)."),
    },
    # ------------------------------------------------------ the 0.948 claim, second look
    # `ckpt948` (zhuzhenghaomax) ran and its checkpoint change turned out inert (`notes/68`),
    # leaving a 0.934-era config. Re-scraping the public kernels turned up THREE more 0.948
    # claims -- rishabhr0y/biohub-948-sew20 (21 votes), cloudssdut (15), tomako4390 (2) --
    # whose configs are byte-for-byte identical to each other and differ from the arm we ran
    # in exactly ONE value:
    #
    #     SECONDARY_EDGE_WEIGHT   0.15  ->  0.20
    #
    # The author says so in the title: "sew20". So `claude-arm-ckpt948`, which we already
    # ran and do not intend to submit, is the CONTROL for this comparison, and the claim is
    # that one parameter is worth +0.007 -- rank ~20 against rank ~120 for a clean 0.941.
    #
    # rishabhr0y is the same author whose 0.941 carries 27 votes, so this is not an unknown
    # making an unsupported claim. Both arms run: the published configuration as-is, and the
    # same parameter on the base we trust, because those answer different questions.
    "sew948": {
        "base": ("rishabhr0y", "biohub-948-sew20"),
        "edits": [],
        "why": ("claimed LB 0.948, unmodified -- the published configuration. Three\n"
                "#               independent kernels carry it identically, and it differs from\n"
                "#               our already-run ckpt948 only in SECONDARY_EDGE_WEIGHT 0.15 ->\n"
                "#               0.20, which makes ckpt948 the control."),
    },
    "sew20": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("SECONDARY_EDGE_WEIGHT", "0.15"), env("SECONDARY_EDGE_WEIGHT", "0.20"))],
        "why": ("SECONDARY_EDGE_WEIGHT 0.15 -> 0.20 on the corroborated 0.941 base. The 0.948\n"
                "#               kernels carry it alongside a REVERT to the narrow division\n"
                "#               config; this asks whether the parameter stacks with the 0.941\n"
                "#               division work instead of replacing it."),
    },
    # ------------------------------------------------- SEW: the gradient that actually paid
    # SCORED 2026-09-06/07. lb941 reproduced its claimed 0.941 EXACTLY (no reproduction
    # offset, unlike claude_fork at -0.001), and against that baseline:
    #
    #     sew20   SEW 0.15 -> 0.20    0.942   +0.001   <- the only arm that gained
    #     union   LINK_MODE adaptive  0.941    0.000
    #     div15   DIVERGE_UM -> 1.5   0.938   -0.003
    #
    # So the axis is the secondary model's edge weighting, and the step 0.15 -> 0.20 that
    # three 0.948-claiming kernels carry is worth exactly what they implied. Nobody has
    # published anything above 0.20 -- `notes/65` §3's stepped-and-stopped shape again, on
    # the one parameter this project has now measured a gain from.
    "sew25": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("SECONDARY_EDGE_WEIGHT", "0.15"), env("SECONDARY_EDGE_WEIGHT", "0.25"))],
        "why": ("SECONDARY_EDGE_WEIGHT 0.15 -> 0.25, one step past the value that scored\n"
                "#               0.942. 0.20 is where every published notebook stops."),
    },
    "sew30": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("SECONDARY_EDGE_WEIGHT", "0.15"), env("SECONDARY_EDGE_WEIGHT", "0.30"))],
        "why": ("SECONDARY_EDGE_WEIGHT 0.15 -> 0.30, two steps past. Run alongside sew25 so\n"
                "#               the pair shows a slope rather than a single point -- notes/41,\n"
                "#               42 and 44 each recorded a one-sample optimum that was noise."),
    },
    # ------------------------------------------------ combining two confirmed +0.001 knobs
    # SCORED 2026-09-07. The SEW gradient is a PLATEAU, not a slope:
    #
    #     SEW 0.15  0.941      0.20  0.942      0.25  0.942      0.30  0.942
    #
    # It saturates at the first step and every value above 0.20 gives the same 0.942. So
    # there is nothing further to extract along that axis.
    #
    # Meanwhile the public frontier reached 0.942 twice, and one of them is OUR OWN
    # DEPRIORITISED ARM: `busyaprime/biohub-0-942-lb-one-knob-past-the-public-line` is
    # lb941 with DET_THRESHOLD 0.965 -> 0.96, byte-for-byte what `det960` does. We built it,
    # verified it (+404 nodes), and shelved it when `union` falsified the node-count
    # hypothesis. Somebody else submitted it and scored 0.942.
    #
    # So there are now TWO independently-confirmed +0.001 knobs from the same 0.941 base,
    # acting on different stages: detection threshold, and the secondary model's edge
    # weight. **Nobody has combined them.** If the mechanisms are independent the pair is
    # worth +0.002 and lands on 0.943, which is rank 100. If they saturate the same
    # plateau it stays 0.942 and the plateau is a property of the pipeline, not of either
    # knob -- which is worth knowing too.
    #
    # DET_THRESHOLD is guarded, so both arms carry the paired guard edit.
    "sewdet": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("SECONDARY_EDGE_WEIGHT", "0.15"), env("SECONDARY_EDGE_WEIGHT", "0.20")),
                  (env("DET_THRESHOLD", "0.965"), env("DET_THRESHOLD", "0.960")),
                  (guard("DET_THRESHOLD", "0.965"), guard("DET_THRESHOLD", "0.960"))],
        "why": ("the two knobs that have each independently scored 0.942 from this base,\n"
                "#               together for the first time: SEW 0.20 (ours) and DET 0.96\n"
                "#               (busyaprime's, identical to our shelved det960). Different\n"
                "#               stages, so additivity is plausible and 0.943 is rank 100."),
    },
    "det955": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("DET_THRESHOLD", "0.965"), env("DET_THRESHOLD", "0.955")),
                  (guard("DET_THRESHOLD", "0.965"), guard("DET_THRESHOLD", "0.955"))],
        "why": ("detection threshold one step past the new public line. 0.965 -> 0.96 is\n"
                "#               now published at 0.942; below 0.96 is unexplored, and the SEW\n"
                "#               plateau is a warning that this one may saturate too."),
    },
    # ---------------------------------------- 0.942 is a ceiling; switch from knobs to search
    # SCORED 2026-09-07/08. Five arms now sit on exactly 0.942 and nothing goes past it:
    #
    #     sew20 / sew25 / sew30   0.942     the SEW axis, saturated at its first step
    #     det960 (busyaprime)     0.942     the detection threshold
    #     sewdet  (SEW + DET)     0.942     BOTH TOGETHER -- still 0.942, not additive
    #     det955                  0.941     one step further loses
    #
    # `sewdet` is the decisive one. Its node count was exactly `sew20 + det960` (+505 =
    # 101 + 404), so both edits landed and they touch disjoint nodes -- and the score still
    # did not move. Two independent +0.001 knobs that do not add are not two contributions;
    # they are two ways onto the same shelf. **Parameter tuning on this checkpoint is done**,
    # which is what Tang (rank 5, 0.961) said on 2026-08-16: *"the current ckpt has kind of
    # hit a wall, it's hard to get more gain from post-processing alone."*
    #
    # The way off a plateau is a different KIND of change. `analyticaobscura/biohub-lb-942`
    # is one: its axis reads "public 0.939 base + holdout-selected post-process
    # configuration", and it turns on three variables no other notebook sets --
    # VALIDATOR_N_PER_TYPE, PPSWEEP_SELECT_MARGIN, PPSWEEP_MAX_ADJ_LOSS. Those appear only
    # in its config cell, so the sweep itself lives inside the shipped pipeline: the support
    # pack already implements a post-processing search with holdout selection, and nobody
    # except this author has switched it on.
    #
    # That is categorically different from every arm above. A fixed knob applies one value to
    # every movie; a holdout-selected sweep picks a DIFFERENT post-processing configuration
    # per dataset. A plateau in the fixed-knob family says nothing about the adaptive one.
    "pp942": {
        "base": ("analyticaobscura", "biohub-lb-942"),
        "edits": [],
        "why": ("the public 0.942, unmodified -- and a MECHANISM rather than a knob. Turns on\n"
                "#               the pipeline's own holdout-selected post-processing sweep\n"
                "#               (VALIDATOR_N_PER_TYPE, PPSWEEP_SELECT_MARGIN,\n"
                "#               PPSWEEP_MAX_ADJ_LOSS), which no other public notebook sets."),
    },
    # Widen the search. VALIDATOR_N_PER_TYPE is how many movies the sweep validates each
    # candidate configuration on; at 4 the selection is being made on very little evidence,
    # and this project has spent a week learning what small validation samples do (notes/64,
    # notes/66). Doubling it costs runtime, which is the real risk -- notes/69 §3 put the
    # graded rerun at roughly 11 h against a 12 h limit.
    "pp942n8": {
        "base": ("analyticaobscura", "biohub-lb-942"),
        "edits": [(env("VALIDATOR_N_PER_TYPE", "4"), env("VALIDATOR_N_PER_TYPE", "8"))],
        "why": ("the 0.942 sweep, validated on twice as many movies per type. At 4 the\n"
                "#               selection rests on very little evidence. RUNTIME RISK: the\n"
                "#               graded rerun is already ~11 h against a 12 h limit."),
    },
    # The mechanism crossed with the knob. lb-942 does NOT set SECONDARY_EDGE_WEIGHT, so
    # this is the sweep plus the one fixed knob we have confirmed at +0.001 -- and unlike
    # `sewdet`, the two are not both fixed-knob members of the same saturated family.
    "pp942sew": {
        "base": ("analyticaobscura", "biohub-lb-942"),
        "edits": [(env("SECONDARY_EDGE_WEIGHT", "0.15"), env("SECONDARY_EDGE_WEIGHT", "0.20"))],
        "why": ("the 0.942 sweep plus SEW 0.20. lb-942 leaves SECONDARY_EDGE_WEIGHT at 0.15,\n"
                "#               so this crosses the adaptive mechanism with the fixed knob --\n"
                "#               different families, unlike sewdet which crossed two members\n"
                "#               of the same saturated one."),
    },
    # ------------------------------------------------------------- EDGE-FEATURE TTA (0.946)
    # `reyhanksatria/biohub-cell-tracking-0-946-lb`, 11 votes, and its axis is a complete
    # provenance chain: "0.933 baseline -> 0.934 harmonic fusion -> 0.939 wider divisions /
    # calmer fusion -> 0.941 repair adaptation -> **0.946 edge-feature TTA**". Same author
    # whose independent 0.941 corroborated the lb941 config in the first place.
    #
    # **+0.005 from one flag**, `BIOHUB_EDGE_FEATURE_TTA=1`, and 0.946 is exactly what rank
    # 100 costs today. That is four times any gain this project has measured.
    #
    # It cannot be ported to our lb941 base: `EDGE_FEATURE_TTA` appears **zero times** in
    # lb941's source and twice in this one, whose EXPERIMENT_TAG is `edge_feature_tta_0946`.
    # Setting the variable on lb941 would be inert -- `notes/68` exactly. So the arm has to
    # be this notebook.
    #
    # Two hazards, both known. Its `datasetDataSources` come back as ['', '', ''], so the
    # mounts are named here instead -- `TTA946_SOURCES`, which is pilkwang's three public
    # CC0 datasets and nothing else; the runtime log confirms every artifact resolves there.
    # And `notes/69` §3 priced TTA out on runtime at ~11 h against a 12 h limit; that was
    # DETECTION TTA doubling the 3D UNet. Edge-feature TTA re-runs the much cheaper edge
    # model, and the author is publishing a graded 0.946, so it evidently fits for them.
    "tta946": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": [],
        "why": ("claimed LB 0.946, unmodified -- BIOHUB_EDGE_FEATURE_TTA=1, the single\n"
                "#               largest published step in this lineage (+0.005) and exactly\n"
                "#               the score rank 100 costs. Cannot be ported to lb941: the flag\n"
                "#               appears zero times there (notes/68's inert-edit trap)."),
    },
    # ------------------------------------------- the same bug, one model further along
    # RAN 2026-09-08, and it reproduced: `EDGE_TTA_ACTIVE views = 8 mean_abs_feat_delta =
    # 0.323` in the log, 122,791 nodes against lb941's 119,279. The 400-epoch artifact name
    # in its header turned out to be nothing -- lb941, ckpt948 and tta946 all materialise
    # the same primary SHA `12f6881e`, so pilkwang's `-50ep-v1` dataset simply contains a
    # snapshot whose manifest calls itself 400ep. **The +0.005 is code, not weights**, which
    # is the good outcome: code we can read, and extend.
    #
    # Reading it showed the author fixed the primary model and left the secondary one
    # untouched -- eight encodes, eight feature maps discarded (`sec_tta_edits` documents
    # the exact lines). So this is not a knob at a new value; it is the mechanism that just
    # paid +0.005, applied to the one place it has not been applied.
    #
    # Why it matters more than another sweep: 0.946 is a **140-team pile-up** (ranks 41-180
    # on the 2026-09-08 board) because everybody forks this notebook. One thousandth above
    # it is rank 41. The knobs are saturated -- `sewdet` proved SEW and DET are two ways
    # onto one shelf -- so the thousandth has to come from a mechanism, and this is the only
    # one in reach that costs no extra GPU time.
    "ttasec": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": sec_tta_edits(),
        "why": ("extend the 0.946 edge-feature TTA to the SECONDARY model, whose eight D4\n"
                "#               encodes still throw away all eight feature maps. Free at\n"
                "#               runtime (the passes already happen) and it raises the run if\n"
                "#               the features do not move, so it cannot be silently inert."),
    },
    # ------------------------------------------- the symmetry the model learned and nobody uses
    # GRADED 2026-09-09: `ttasec` 0.945, `tta946` 0.944, rank 272 of 3,281. The public
    # frontier is 0.946 and there is nothing above it to fork -- a scan of 906 kernels finds
    # the only >=0.947 titles are the July metric-hack notebooks and the Aug-30 `948` branch
    # `notes/68` proved inert. **0.948 is rank 37 and has to be built, not copied.**
    #
    # The support pack ships its training code, and `scripts/augmentations.py` is the find:
    #
    #     def flip_augment(...):
    #         """Random spatial flip: samples uniformly from all 8 axis-aligned symmetries.
    #         Each of Z, Y, X is independently flipped with probability 0.5."""
    #
    # Two augmentations were used, brightness and flip, and the flip group is the full
    # {Z, Y, X} product. Meanwhile every transform in the public eight-view TTA acts on
    # `(-2, -1)` -- Y and X. **Z is never averaged over, and it is the axis the model was
    # trained to be invariant to.** The rot90 and transpose views that ARE used are not in
    # the training set at all; they work on isotropy, not on anything the model was taught.
    #
    # So this is the same lever that produced the two gains in this lineage -- more of the
    # ensemble the model already supports -- pointed at the one direction nobody has tried.
    # It sits on `ttasec` because that is our best measured arm and the changes are disjoint:
    # `ttasec` averages the SECONDARY model's features over its existing views, `z16` widens
    # the PRIMARY model's view set.
    #
    # RUNTIME IS THE RISK, not correctness. The primary predict pass doubles. `tta946` took
    # 27 min on four clips against a ~17x graded set and a 720 min ceiling, so this arm is
    # run to MEASURE `predict_minutes_total` first and is not proposed for a slot until that
    # number is in. If it does not fit, the fallback is the same edit with only the single
    # pure Z-flip view added (nine views, +12%).
    "ttaz16": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": sec_tta_edits() + [z_tta_edit()],
        "why": ("sixteen-view TTA: the public eight are all in-plane (Y, X), and the model\n"
                "#               was trained with flips over Z as well. Adds the Z-flip of every\n"
                "#               existing view, on top of ttasec. Primary predict time doubles."),
    },
    # ------------------------------- the notebook titles lie, and the board says who to copy
    # The decisive check, and it should have been the first one: **look up the author on the
    # leaderboard**, not at the title of their notebook.
    #
    #     reyhanksatria    rank 293   0.944     <- author of "biohub-cell-tracking-0-946-lb"
    #     analyticaobscura rank 100   0.946     <- author of "biohub-lb-941" and "biohub-lb-942"
    #     rishabhr0y       rank 108   0.946
    #     pilkwang         rank  56   0.946     <- the model author
    #
    # `reyhanksatria` scores **0.944**, exactly what our unmodified fork of their notebook
    # scored. There is no reproduction gap; the title is aspirational and the notebook's own
    # "Verified score progression ... 0.946" is a claim, not a graded result. My "-0.002 we
    # cannot account for, worth more than every knob on the board" was chasing a phantom.
    #
    # It also settles `ttasec` cleanly. Base 0.944 -> ours 0.945 with two independent runs
    # agreeing on the base, so **the secondary edge-feature TTA is +0.001, measured**.
    #
    # And it redirects the copying. `analyticaobscura` publishes at 0.941/0.942 while sitting
    # at 0.946, so their public work is not their best. `rishabhr0y` publishes mechanisms, and
    # this is the one worth having.
    "ttadom": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": sec_tta_edits() + dominance_edits(),
        "why": ("ttasec + rishabhr0y's motion-relink confidence dominance. Today the motion\n"
                "#               model overwrites EVERY ILP edge (67,249 on one clip); this keeps\n"
                "#               a raw edge that strictly beats every conflicting motion edge.\n"
                "#               Their own naming prices it at 0.941 -> 0.943."),
    },
    # ----------------------------------------------- the two biggest levers, stacked
    # RAN 2026-09-09. `ttaz16` is the largest change this project has produced, and the
    # coordinate-keyed diff is what shows it -- keying edges by `node_id` reports churn that
    # is mostly renumbering, so every diff here re-keys nodes by `(dataset, t, z, y, x)` and
    # edges by the coordinate pair:
    #
    #     tta946  -> ttasec     nodes -599    +543      edges -644    +588      0.5%
    #     ttasec  -> ttasecw20  nodes -218    +308      edges -242    +329      0.3%
    #     ttasec  -> ttadse44   nodes -3,017  +4,476    edges -3,406  +4,832    4%
    #     ttasec  -> ttaz16     nodes -22,363 +22,060   edges -26,454 +26,144   22%
    #
    # And 73% of `ttaz16`'s moved nodes land within 2 um of the node they replaced -- one
    # voxel is 1.625 um in z and 0.406 in y/x -- so most of that 22% is the detector
    # *localising better*, not finding different cells. The remaining 15% beyond 5 um are
    # genuinely different detections. `notes/04` measured detection as "essentially the whole
    # contest", and this is the first arm that moves it.
    #
    # Runtime measured rather than guessed: predict 10.74 -> 14.08 min on four clips, so the
    # graded set lands near 510 min against the 720 ceiling. It fits.
    #
    # This arm stacks it with `ttadom`, which acts nowhere near it -- Z-flip averaging changes
    # what the detector sees; confidence dominance changes which edges survive the motion
    # model's overwrite. If each is worth what it looks worth, this is the 0.948 arm.
    "ttaz16dom": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": sec_tta_edits() + dominance_edits() + [z_tta_edit()],
        "why": ("the two largest levers together: sixteen-view TTA over the Z symmetry the\n"
                "#               model was trained on, and rishabhr0y's confidence dominance over\n"
                "#               the motion model's wholesale edge replacement. Disjoint stages."),
    },
    # ---------------------------------------- the threshold every notebook inherited at 0.48
    # `notes/65` §3's best category: a value identical in all 56 kernels mined, never swept,
    # just carried forward from whoever wrote the first notebook -- and this one gates **every
    # candidate edge in every frame pair**, which makes it the largest untouched population in
    # the pipeline. `dse44` was built for the lb941 base and never ran; the base has moved on
    # twice since, and the argument is stronger now, not weaker: `ttasec` changed the very
    # logits this threshold cuts, so a cut tuned for the old ones is the wrong cut.
    #
    # Downward first, on Soheil's diagnosis (rank 2, 0.966): *"many 'linking' issues actually
    # originated earlier during node selection"* -- missing endpoints, not bad associations.
    # A more permissive edge threshold is the direction that addresses that.
    #
    # Unguarded, so one env edit; the receipt's hardcoded `edge_candidate_threshold` moves
    # with it so the artifact does not claim a value the run did not use.
    "ttadse44": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": sec_tta_edits() + [
            ("os.environ['BIOHUB_DUAL_SEED_EDGE_THRESHOLD'] = '0.48'",
             "os.environ['BIOHUB_DUAL_SEED_EDGE_THRESHOLD'] = '0.44'"),
            ("'edge_candidate_threshold': 0.48,", "'edge_candidate_threshold': 0.44,"),
        ],
        "why": ("ttasec + DUAL_SEED_EDGE_THRESHOLD 0.48 -> 0.44. Identical in all 56 public\n"
                "#               kernels and never swept, and it gates every candidate edge in\n"
                "#               every frame pair -- the largest untouched population there is."),
    },
    # -------------------------------------- one movie loses the fusion; the other three do not
    # **Read the corrected version of this. The first one was wrong.**
    #
    # `tta946`'s stdout carries 65 `BIOHUB_RETENTION_GUARD` lines and every one ends
    # `"use_primary": true`, which I read as "the guard has never once passed". It prints
    # only when it *rejects*. Reading a filtered stream as the population is `notes/68`'s
    # config-dump trap and `notes/71`'s rank arithmetic in a third costume.
    #
    # The kernel also writes `retention_guard_*.jsonl`, one record per frame, and that is the
    # population -- 400 frames, 100 per movie:
    #
    #     44b6_0113de3b   100 frames    0 rejected   median retention 1.012
    #     44b6_0b24845f   100 frames   64 rejected                    0.864
    #     6bba_05b6850b   100 frames    1 rejected                    1.014
    #     6bba_05db0fb1   100 frames    0 rejected                    1.011
    #     all 400: 65 rejected (16%),  median 1.004,  max 1.185
    #
    # So the dual-seed fusion applies on **335 of 400 frames**, and the median blend *adds*
    # candidates rather than losing them (retention > 1). `SECONDARY_DETECTION_WEIGHT` is not
    # inert. What is true is narrower and still interesting: on `44b6_0b24845f` alone the
    # fusion is discarded on 64% of frames, and that movie's median retention is 0.864 while
    # every other movie sits above 1.00. One embryo where the two detectors disagree.
    #
    # 0.85 flips **26 frames** -- those in [0.85, 0.90), where the blend discards least -- and
    # leaves 39 vetoed where retention collapses toward 0.45. Soheil (rank 2) named missing
    # endpoint nodes as his dominant error, so letting 20% of candidates go on a bad frame is
    # a risk this arm declines. A narrow test, priced honestly, on an axis nobody has moved.
    #
    # SIX edits, because the notebook defends this number in three separate places and the
    # first attempt found only one of them. v1 moved the two env assignments and
    # `_EXPECTED_TEXT`, ran the whole pipeline, and died at the very end on a *second*
    # hardcoded expectation in the post-run verification block:
    #
    #     RuntimeError: Frame-retention diagnostic contract changed
    #
    # -- `if float(_guard_record['minimum_retention']) != 0.9`, plus a recomputation of the
    # decision as `retention < 0.9`. `notes/70`'s lesson, repeated: grep for the *value*, not
    # for the guard idiom you already know. The report dict's hardcoded 0.9 moves too; it
    # raises nothing, but a receipt that states a threshold the run did not use is the exact
    # thing this project keeps being misled by.
    "ttaret85": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": [
            ("os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'\n"
             "os.environ['BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION'] = '0.90'",
             "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'\n"
             "os.environ['BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION'] = '0.85'"),
            ("print('Calibrated dual-seed runtime patch applied')\n"
             "os.environ['BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION'] = '0.90'",
             "print('Calibrated dual-seed runtime patch applied')\n"
             "os.environ['BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION'] = '0.85'"),
            ("'BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION': '0.90'",
             "'BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION': '0.85'"),
            ("if float(_guard_record['minimum_retention']) != 0.9 or",
             "if float(_guard_record['minimum_retention']) != 0.85 or"),
            ("and float(_guard_record['retention']) < 0.9)",
             "and float(_guard_record['retention']) < 0.85)"),
            ("'configuration': {'minimum_candidate_retention': 0.9,",
             "'configuration': {'minimum_candidate_retention': 0.85,"),
        ],
        "why": ("the dual-seed retention floor 0.90 -> 0.85, which flips 26 of 400 frames --\n"
                "#               all in 44b6_0b24845f, the one movie where the two detectors\n"
                "#               disagree (median retention 0.864 against >1.00 elsewhere)."),
    },
    # --------------------------------- the mechanism and the knob that should follow it
    # RAN 2026-09-08. `ttasec` fired -- `SEC_EDGE_TTA_ACTIVE views = 8 mean_abs_feat_delta =
    # 0.231` on every frame pair, against the primary's 0.323 -- and it moved the pipeline
    # end to end: **101 of 284 run_stats counters changed**, 1,234 edges dropped and 1,178
    # added (~1% of the edge set re-routed), raw detection included. Net counts barely move
    # (-56 nodes, -56 edges) because the churn nearly cancels, which is exactly why
    # `diff_arms` reports added/dropped separately rather than a delta.
    #
    # So the secondary model's features improved by almost as much as the primary's did, and
    # bought a much smaller downstream change -- because the secondary enters at
    # SECONDARY_EDGE_WEIGHT 0.15 plus the low-margin consensus gate. **If the features are
    # better, the weight on them is now too small.** That is the argument `ttasew20` makes
    # blind; this arm makes it on top of the improved features, where it is actually implied.
    #
    # It decomposes cleanly despite being two changes, because `ttasec` is being scored on
    # its own: `ttasecw20 - ttasec` isolates the weight, exactly as `sewdet` was meant to
    # decompose and could not, having no separate score for either half.
    "ttasecw20": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": sec_tta_edits() + [("os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.15'",
                                     "os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.20'")],
        "why": ("ttasec plus SECONDARY_EDGE_WEIGHT 0.15 -> 0.20. The secondary model's edge\n"
                "#               features are now eight-view averaged like the primary's, so\n"
                "#               the weight tuned for single-view features is the wrong one."),
    },
    # ----------------------------------------- the two confirmed knobs, on the new base
    # `sewdet` closed SEW and DET *on lb941*: 0.942 each, 0.942 together, node counts exactly
    # additive. The conclusion recorded then was "two ways onto one shelf" -- and the shelf
    # is a property of what the association head can see, which is precisely what
    # edge-feature TTA changes. So the saturation argument does not carry over unexamined,
    # and re-testing the one knob that ever paid us costs a single unguarded edit.
    #
    # SEW is the right one to move first: it weights the secondary model's edge logits, the
    # same pathway `ttasec` improves. If better secondary features are worth anything, the
    # weight on them should want to be larger, and the two arms read each other.
    "ttasew20": {
        "base": ("reyhanksatria", "biohub-cell-tracking-0-946-lb"),
        "sources": TTA946_SOURCES,
        "edits": [("os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.15'",
                   "os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.20'")],
        "why": ("SECONDARY_EDGE_WEIGHT 0.15 -> 0.20 on the 0.946 base. It is the only knob\n"
                "#               that has ever gained us a thousandth, and it saturated on a\n"
                "#               base whose association features were single-view; this one's\n"
                "#               are eight-view. Unguarded here, so one edit."),
    },
    # ------------------------------------------------- knobs no public notebook has moved
    # The config matrix over 56 top kernels has two columns: values that vary between
    # notebooks, and values that are identical in every single one. The second column is
    # `notes/65` §3's best category -- not swept, not even stepped, just inherited from
    # whoever wrote the first notebook. Ranked by the size of the population each gates:
    #
    #   DUAL_SEED_EDGE_THRESHOLD    0.48   every candidate edge, in every frame pair
    #   DEEPCENTER_GAP_THRESHOLD    0.25   every gap-closure proposal   (-> dcgap40)
    #   ILP_DISAPPEARANCE_WEIGHT    2      every track termination
    #   OUTPUT_MIN_TRACK_LEN        6      every short track
    #
    # DUAL_SEED_EDGE_THRESHOLD is the largest of those by a wide margin, and it is the one
    # place the two detector seeds' disagreement is resolved. Both directions, because
    # nothing about 0.48 says which side of it is better -- `notes/56` closed OUR detection
    # threshold in both directions and that is a different parameter in a different pipeline.
    "dse44": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("DUAL_SEED_EDGE_THRESHOLD", "0.48"),
                   env("DUAL_SEED_EDGE_THRESHOLD", "0.44"))],
        "why": ("dual-seed edge acceptance threshold 0.48 -> 0.44 (more permissive).\n"
                "#               Identical in all 56 public kernels mined -- inherited, never\n"
                "#               moved -- and it gates every candidate edge in every frame pair."),
    },
    "dse52": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("DUAL_SEED_EDGE_THRESHOLD", "0.48"),
                   env("DUAL_SEED_EDGE_THRESHOLD", "0.52"))],
        "why": ("dual-seed edge acceptance threshold 0.48 -> 0.52 (stricter). The other side\n"
                "#               of dse44; nothing about the inherited 0.48 says which way is\n"
                "#               better, so both are run before either is believed."),
    },
    "det960": {
        "base": ("analyticaobscura", "biohub-lb-941"),
        "edits": [(env("DET_THRESHOLD", "0.965"), env("DET_THRESHOLD", "0.960")),
                  (guard("DET_THRESHOLD", "0.965"), guard("DET_THRESHOLD", "0.960"))],
        "why": ("detection threshold one step past the public step. 0.97 -> 0.965 was part of\n"
                "#               the 0.938 -> 0.940 move; below 0.965 is unpublished."),
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

    # Some kernels come back from /kernels/pull with datasetDataSources = ['', '', ''] --
    # reyhanksatria's 0.946 does. The mounts are then unreconstructable from provenance, so
    # the registry entry names them itself. Passing BOTH the author's re-hosted copies and
    # pilkwang's originals is deliberate: the notebook resolves artifacts by slug with
    # ALLOW_ARTIFACT_FALLBACK, and an extra mount costs nothing while a missing one is fatal.
    sources = arm.get("sources") or rec["datasetDataSources"]
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
        # Edits come in two shapes now: `os.environ["BIOHUB_X"] = "v"` and a line of the
        # notebook's own guard dict, `    "BIOHUB_X": v,`. v1 assumed the first and split
        # on "[", which raised IndexError on the second.
        def describe(o, n):
            # v2 assumed every edit was one line and formatted `old -> new` from its tail.
            # `ttasec`'s edits span code blocks, so the embedded newlines walked straight
            # out of the comment and Kaggle raised `IndentationError: unexpected indent`
            # on `del secondary_imgs_flip, secondary_det_flip -> _su_flip.flip(dims)`.
            # A header is documentation; it must never be able to produce executable text.
            m = re.search(r'BIOHUB_([A-Z0-9_]+)', o)
            key = m.group(1) if m else "?"
            if "\n" in o or "\n" in n:
                a, b = o.count("\n") + 1, n.count("\n") + 1
                first = o.strip().split("\n")[0][:52]
                return f"#     {'code':<32} {a} lines -> {b} lines   at `{first}`"
            val = lambda t: t.rstrip(",").split(":")[-1].split("=")[-1].strip().strip('"')
            kind = "guard" if o.lstrip().startswith('"') else "env"
            return f"#     {key:<32} {val(o)} -> {val(n)}   ({kind})"

        lines = "\n".join(describe(o, n) for o, n in applied)
        # Belt and braces: whatever `describe` returns, every line of the header is a
        # comment by the time it reaches the notebook.
        lines = "\n".join(ln if ln.lstrip().startswith("#") else "#     " + ln.strip()
                          for ln in lines.split("\n"))
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
