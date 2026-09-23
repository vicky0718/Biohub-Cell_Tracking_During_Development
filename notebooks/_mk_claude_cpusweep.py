"""Sweep post-processing on a CPU kernel, by reusing a finished GPU run's predictions.

    python notebooks/_mk_claude_cpusweep.py            # from lb50's cached predictions

**Why not TPU.** The obvious answer to an exhausted GPU quota is the other accelerator, and
it does not work here — not for a reason about quotas, but because of what this notebook is.
Cell 5 opens with the author's own verdict:

    if not _torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this notebook. "
                           "Enable a Kaggle GPU accelerator and commit again.")

There is no `torch_xla` anywhere in its 227,000 characters, the one device line is
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`, and both inference stages
shard work by setting `CUDA_VISIBLE_DEVICES` from `torch.cuda.device_count()`. On a TPU VM
that count is 0, so selecting TPU does not move the model onto a TPU — it removes CUDA and
leaves the CPU fallback, with the shard launcher dividing by a device count of zero. Making
it real means porting a UNet3D and a node transformer to XLA and re-verifying against the
T4 baseline, inside six days, in a notebook we did not write.

**What actually unblocks us is that the expensive half already ran.** `MOTION_RELINK_*`,
`GAP_*`, `SAFE_DIV_*` and the rest are **post**-processing: they run on the prediction graphs,
long after the model. Those graphs are in every finished arm's output —
`tracking_repo/predictions/.../<movie>.geff` with `edge_prob` and `edge_dist` on every edge,
for all 4 test clips and all 8 validator movies — and the sweep that scores them is
`scipy.optimize.linear_sum_assignment` and numpy. The notebook's own `PP_CANDIDATES` machinery
re-runs exactly that per candidate, at about 7 CPU-minutes each. **None of it needs a GPU.**

So this builds the arm notebook with its two inference launches replaced by a copy from a
mounted kernel output, and the accelerator turned off. One kernel, no GPU quota, no TPU port,
and the whole ladder measured.

**What it cannot do.** It cannot produce a submission. A graded rerun is handed the hidden
movies, which have no cached predictions, so the copy would find nothing — that is what the
loud failure in `_reuse_predictions` is for, rather than silently predicting nothing. This is
a measuring instrument; the winner it names still costs one GPU run to ship.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import claude_kaggle_api as K          # noqa: E402
from _mk_claude_arm import ARMS, TTA946_SOURCES     # noqa: E402

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- the four edits
# 1. The CUDA gate. Replaced rather than deleted so the log still says, in one line, which
#    mode the run is in -- `notes/68`'s lesson is that a silent fallback is the expensive bug.
CUDA_GATE_OLD = '''import torch as _torch

if not _torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this notebook. Enable a Kaggle GPU accelerator and commit again."
    )
print("CUDA device:", _torch.cuda.get_device_name(0))'''

CUDA_GATE_NEW = '''import torch as _torch
import shutil as _cpu_shutil

CPU_SWEEP_MODE = True
print("=" * 78)
print("CPU SWEEP MODE -- no accelerator. Both inference stages are replaced by a copy")
print("of a finished GPU run's prediction graphs; only post-processing is recomputed.")
print("  torch.cuda.is_available():", _torch.cuda.is_available())
print("=" * 78, flush=True)


def _reuse_predictions(method, stems):
    """Copy cached `.geff` prediction graphs into the directory inference would have filled.

    Raises if the mount is missing or short. That is deliberate and it is the whole safety
    property of this kernel: on a graded rerun the hidden movies have no cached predictions,
    the copy finds nothing, and the run dies **here** rather than quietly scoring an empty
    graph and writing a plausible-looking submission. `notes/68` is the cautionary case --
    a path that silently missed, a default that silently applied, and an arm that measured
    nothing while looking like an experiment.
    """
    candidates = []
    for root in sorted(Path("/kaggle/input").glob("**/tracking_repo/predictions")):
        candidates += sorted(root.glob(f"*/{method}/split_0"))
    if not candidates:
        raise RuntimeError(
            f"CPU SWEEP: no mounted predictions for method {method!r}. Looked under "
            f"/kaggle/input/**/tracking_repo/predictions/*/{method}/split_0 . "
            f"Mounts present: {[p.name for p in Path('/kaggle/input').glob('*')]}")
    source = candidates[0]
    have = {p.stem for p in source.glob("*.geff")}
    want = set(stems)
    if not want <= have:
        raise RuntimeError(
            f"CPU SWEEP: {source} is missing {sorted(want - have)} "
            f"(has {len(have)} graphs). A cached run cannot cover movies it never saw.")
    # Build the destination rather than looking it up. v1 passed
    # `_prediction_dir_for_method(METHOD)` as the destination, and that function is a
    # *lookup*: it globs `predictions/*/<method>/split_0` and raises when the count is not
    # exactly 1. Asking it for the directory we were about to create could only ever raise
    # `Expected exactly one prediction directory for 'unet_transformer', found []`, which is
    # what it did. The layout is `predictions/<user>/<method>/split_0`, and the source mount
    # carries the same <user> segment, so copy it across.
    destination = REPO_DIR / "predictions" / source.parent.parent.name / method / "split_0"
    destination.mkdir(parents=True, exist_ok=True)
    for stem in sorted(want):
        target = destination / f"{stem}.geff"
        if target.exists():
            _cpu_shutil.rmtree(target) if target.is_dir() else target.unlink()
        _cpu_shutil.copytree(source / f"{stem}.geff", target)
    print(f"CPU SWEEP: reused {len(want)} prediction graphs from {source}", flush=True)'''

# 2/3. The two inference launches. Each is a `worker_count = min(2, <cuda devices>, ...)`
#      followed by an `if worker_count >= 2:` shard path and an `else:` single-process path.
#      Forcing the count to 0 takes the shard path out, and the else body becomes the copy.
TEST_COUNT_OLD = "worker_count = min(2, available_gpu_count, len(test_stems))"
TEST_COUNT_NEW = "worker_count = 0  # CPU sweep: never shard, never predict"

TEST_PREDICT_OLD = '''    reason = "SLICE is active" if SLICE else f"only {available_gpu_count} CUDA device(s) available"
    print(f"Using single-process prediction because {reason}.")
    print(" ".join(predict_cmd))
    subprocess.run(
        predict_cmd,
        cwd=REPO_DIR,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )'''
TEST_PREDICT_NEW = '''    _reuse_predictions(METHOD, test_stems)'''

VAL_COUNT_OLD = "val_worker_count = min(2, _torch.cuda.device_count(), len(val_stems))"
VAL_COUNT_NEW = "val_worker_count = 0  # CPU sweep: never shard, never predict"

VAL_PREDICT_OLD = '''        print("VALIDATOR: using single-process prediction (fewer than 2 GPUs or samples).")
        subprocess.run([*predict_val_cmd, "--method", val_method_prefix],
                        cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)'''
VAL_PREDICT_NEW = '''        _reuse_predictions(val_method_prefix, val_stems)'''

# 4. The ladder. Wider than `lbsweep`'s because CPU minutes are not rationed the way GPU
#    hours are -- this is the point of the whole exercise. The notebook scores a
#    `combo(...)` of everything that clears the margin on top, for free.
# Built on top of `lbsweep`, whose ladder is already in place -- `lb50` carries NO_SWEEP and
# has no candidate list to widen. Anchoring on the b8 line rather than the dict header keeps
# the match unique: the builder's attribution comment repeats the header's first line.
LADDER_OLD = '''    "b8":        {"MOTION_RELINK_LEARNED_BONUS": 8.0},
    "b12":       {"MOTION_RELINK_LEARNED_BONUS": 12.0},
    "b20":       {"MOTION_RELINK_LEARNED_BONUS": 20.0},'''
LADDER_NEW = '''    "b8":        {"MOTION_RELINK_LEARNED_BONUS": 8.0},
    "b12":       {"MOTION_RELINK_LEARNED_BONUS": 12.0},
    "b20":       {"MOTION_RELINK_LEARNED_BONUS": 20.0},
    "b35":       {"MOTION_RELINK_LEARNED_BONUS": 35.0},
    "vel000":    {"MOTION_RELINK_VELOCITY_WEIGHT": 0.0},
    "b12vel025": {"MOTION_RELINK_LEARNED_BONUS": 12.0,
                  "MOTION_RELINK_VELOCITY_WEIGHT": 0.25},'''

EDITS = [
    (CUDA_GATE_OLD, CUDA_GATE_NEW),
    (TEST_COUNT_OLD, TEST_COUNT_NEW),
    (TEST_PREDICT_OLD, TEST_PREDICT_NEW),
    (VAL_COUNT_OLD, VAL_COUNT_NEW),
    (VAL_PREDICT_OLD, VAL_PREDICT_NEW),
    (LADDER_OLD, LADDER_NEW),
]


def build(arm: str = "lb50", cache_arm: str = "lb50") -> int:
    src = HERE / f"claude_arm_{arm}.ipynb"
    if not src.exists():
        print(f"REFUSING TO BUILD — {src.name} not built; run _mk_claude_arm.py {arm}")
        return 1
    if arm not in ARMS:
        print(f"REFUSING TO BUILD — {arm!r} is not in ARMS")
        return 1

    nb = json.loads(src.read_text())
    cells = nb["cells"]

    for old, new in EDITS:
        hits = [i for i, c in enumerate(cells)
                if c.get("cell_type") == "code" and old in "".join(c["source"])]
        total = sum("".join(cells[i]["source"]).count(old) for i in hits)
        if total != 1:
            print(f"REFUSING TO BUILD — anchor matched {total}x across {len(hits)} cells, "
                  f"expected exactly 1:\n  {old[:110]!r}")
            return 1
        i = hits[0]
        cells[i]["source"] = "".join(cells[i]["source"]).replace(old, new, 1) \
            .splitlines(keepends=True)

    future = "from __future__ import annotations\n"
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        body = "".join(c["source"])
        compile(future + body.replace(future, "", 1) if future in body else body,
                f"claude_cpusweep_cell{i}", "exec")

    out = HERE / "claude_cpusweep.ipynb"
    out.write_text(json.dumps(nb, indent=1))

    arm_cfg = json.loads((HERE / f"claude_arm_{arm}_push.json").read_text())
    kernels = [k for k in (arm_cfg.get("kernel_sources") or [])
               if "wheelhouse" not in k]          # the wheelhouse is a P100 fix; no GPU here
    kernels.append(f"{K.username()}/claude-arm-{cache_arm}")
    (HERE / "claude_cpusweep_push.json").write_text(json.dumps({
        "slug": "claude-cpusweep", "title": "Claude cpusweep",
        "notebook": str(out), "dataset_sources": arm_cfg["dataset_sources"],
        "competition_sources": arm_cfg.get("competition_sources") or [],
        "kernel_sources": kernels,
        "enable_gpu": False, "enable_internet": False,
    }, indent=1))
    print(f"wrote {out.name}: {len(EDITS)} edits, CPU only, "
          f"predictions reused from claude-arm-{cache_arm}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(build(args[0] if args else "lb50",
                           args[1] if len(args) > 1 else (args[0] if args else "lb50")))
