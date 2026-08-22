"""Build notebooks/claude_temporal_train.ipynb — step 1, the temporal input.

Derived from `_build_claude_detector_earlystop.py`. Two structural changes beyond the
input itself, both forced by `notes/21`:

  * checkpoints are selected on PAIRED recall, not node recall, because node recall was
    provably blind to the failure being fixed;
  * frames are sampled in consecutive RUNS rather than at random, which is what makes a
    paired measurement possible at all — and costs nothing, because a run of L frames
    supplies L training centres from L+2 stored frames.

A radius-0 control trains in the same notebook on the same runs, so the comparison is
against this data pipeline rather than against a remembered number from a different one.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_temporal_train.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

# One list; the header and the grading cell are both generated from it (notes/18 §1).
PREDICTIONS = [
    ("The temporal model improves coherence.",
     "notes/22 sized the remaining coherence deficit at 88% of 66.5 points and named the "
     "cause: the model sees one frame at a time. Measured as: r=1's paired recall beats "
     "r=0's on BOTH folds. Falsified if it does not move, or moves down.",
     "coherence_improves"),
    ("The gain arrives through coherence, not through node recall.",
     "notes/22 separated the two axes -- removing refine bought 7.9 points of temporal "
     "position while node recall FELL 0.0033. If the same holds, r=1's node recall "
     "should move by less than +0.02 while paired recall moves more. Falsified if most "
     "of the gain is recall, which would mean the diagnosis in notes/21 was wrong about "
     "the mechanism and right about the fix only by luck.",
     "gain_is_coherence"),
    ("The radius-0 control reproduces phase 1b.",
     "The data pipeline changed from random frames to consecutive runs. If that change "
     "is neutral, the r=0 arm's node recall should land within 0.05 of notes/20's pu "
     "numbers (0.8826 on 44b6-eval, 0.9148 on 6bba-eval). Falsified if it does not -- in "
     "which case ANY r=1 vs r=0 difference measured here is about the sampling change "
     "and not about temporal input.",
     "control_reproduces"),
]

md(r"""
# Step 1 — temporal input, and a metric that can see whether it worked

`notes/22` closed the previous run with the remaining lever sized rather than guessed:

| | |
|---|---|
| temporal position gap, UNet → DoG | **66.5 points** |
| closed by removing `refine_centroids` | 7.8 points (12 %) |
| **remaining** | **58.7 points (88 %)** |

The remaining 88 % is the thing refinement was never going to fix: **the model sees one
frame at a time.** Nothing in its input or its loss asks it to agree with the frame before
or after. This notebook changes the input.

`Config.temporal_radius=1` stacks `(t−1, t, t+1)` as channels; the network still predicts
the **centre** frame's cell centres. Windows clamp at the movie ends rather than
zero-padding, in training and at inference alike.

## The measurement had to change too, and that matters more than the model

Every training run so far selected its checkpoint on **node recall**. `notes/21` §1 proved
that quantity is blind to the failure being fixed:

| | `unet_cap1.0` | `champion` |
|---|---|---|
| node recall | **0.866** | **0.866** |
| edge Jaccard | 0.6391 | 0.7128 |

Identical node recall across a 0.074 edge-Jaccard gap. Selecting a temporal model on node
recall would optimise a number that provably cannot distinguish the thing the model exists
to do — and would then report success or failure on that number.

So this run selects on **paired recall**: the fraction of ground-truth links with *both*
endpoints matched, which is exactly the numerator of edge recall, computed with the
scorer's own bipartite matching against the real GT edge list. Alongside it, `position`
places that value on the interval `notes/21` §2 defined — `r²` if detections were
independent frame to frame, `r` if the same cells were found every frame. DoG sits at
**+25.2 %**, the per-frame UNet at **−41.3 %**. That is the number to move.

## Two arms, and why the control is not optional

Frames are now sampled in **consecutive runs** rather than at random — a paired measurement
needs consecutive frames, and a run of `L` frames yields `L` training centres from `L+2`
stored frames, so it costs about 25 % of memory and nothing in volume count.

That is a second change, and `notes/18` §1 is the standing lesson about moving two things
at once. So **`r=0` trains here too, on the same runs, with the same code** — if the r=0
control does not reproduce phase 1b, any r=1 vs r=0 difference is about the sampling change
and not about temporal input, and prediction 3 says so in advance.

Arms: `temporal_radius` ∈ {0, 1} × both leave-one-embryo-out folds, loss `pu` (the only
loss that cleared DoG on both folds in `notes/20`). Four trainings, as in phase 1b.
""")

md("## Pre-registered\n\n" + "\n".join(
    f"{i}. **{c}** {w}" for i, (c, w, _) in enumerate(PREDICTIONS, 1)))

code(r"""
import subprocess, sys, time

def sh(*args, **kw):
    # A missing binary raises rather than returning non-zero, which would kill this cell.
    try:
        return subprocess.run(args, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(args, 127, "", str(e))

def pip_install(pkgs, extra=()):
    r = sh(sys.executable, "-m", "pip", "install", "-q", *extra, *pkgs)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r.returncode == 0

gpu = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").stdout.strip()
print(f"accelerator: {gpu or 'NONE'}")
# Kaggle's GPU option gives a P100 (sm_60); the image torch ships sm_70+ only, so CUDA
# reports available and every launch dies. machineShape is read-only on kernels/push, so
# T4 cannot be requested -- replacing torch is the only lever (notes/17 §4).
if "P100" in gpu:
    print("P100 -> installing torch with sm_60 kernels ...")
    t0 = time.time()
    ok = pip_install(["torch==2.5.1"],
                     extra=("--index-url", "https://download.pytorch.org/whl/cu121"))
    print(f"  torch replacement {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")
print("installing geff + zarr ...")
pip_install(["geff", "zarr"])
""")

code(r"""
import sys, os, gc, time, json, hashlib
from pathlib import Path
import numpy as np
import torch

WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device {DEV}")
if DEV.type == "cuda":
    cc = torch.cuda.get_device_capability(0)
    print(f"  sm_{cc[0]}{cc[1]}  built for {torch.cuda.get_arch_list()}")
    try:
        _w = torch.nn.Conv3d(1, 4, 3, padding=1).to(DEV)
        _ = _w(torch.randn(2, 1, 8, 8, 8, device=DEV)).sum().item()
        torch.cuda.synchronize(); print("  GPU smoke test passed")
    except Exception as e:
        raise SystemExit(f"GPU present but unusable: {type(e).__name__}: {str(e)[:200]}")
else:
    raise SystemExit("No GPU. This notebook needs one; set Accelerator -> GPU.")

def find_dir(is_match, roots, max_depth=5):
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

REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
if REPO is None:
    raise SystemExit("Could not find harness/ and pipeline/.")
sys.path.insert(0, str(REPO))

import pipeline.classical as pc
for need in ("dog_response", "open_movie", "load_frame"):
    if not hasattr(pc, need):
        raise SystemExit(f"Snapshot lacks pipeline.classical.{need} — re-upload the repo.")
# The whole run is about temporal_radius. If the attached snapshot predates it, every arm
# below would silently train a single-frame model and the notebook would report a null
# result for a feature it never used.
if "temporal_radius" not in pc.Config.__dataclass_fields__:
    raise SystemExit("The uploaded snapshot's Config has no `temporal_radius` — "
                     "re-upload the repo. Without it BOTH arms here are r=0.")
import pipeline.detector as pd_
if not hasattr(pd_, "paired_recall"):
    raise SystemExit("The uploaded snapshot's pipeline.detector has no `paired_recall` — "
                     "re-upload. Checkpoint selection depends on it.")

from pipeline.classical import Config, dog_response, load_frame, open_movie, detect_frame_dog
from pipeline.detector import (TargetConfig, make_loss_mask, make_target, paired_recall,
                               peaks_from_prob, recall_at_budget)
from pipeline.unet import LOSSES, UNet3D, count_params, predict_volume
from harness.tracks import read_geff, read_estimated_nodes
print("modules imported")

COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Could not find the competition data.")
TRAIN = COMP / "train"
names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})
by_embryo = {}
for n in names:
    by_embryo.setdefault(n.split("_")[0], []).append(n)
print("embryos:", {k: len(v) for k, v in sorted(by_embryo.items())})

CFG = Config(detector="dog", min_separation_um=6.0, dog_rel_threshold=0.005,
             dog_scales=[(1.5, 4.0), (2.5, 6.0)], footprint="ball")
DS = CFG.downsample
TCFG = TargetConfig(loss="masked")     # only pos_radius_um and the mask rules are used
RADII = (0, 1)                         # 0 is the CONTROL, not an afterthought
LOSS_NAME = "pu"                       # notes/20: the only loss clearing DoG on both folds
RUN_LEN = 5                            # consecutive frames per run; L centres from L+2
N_RUNS = 5                             # runs per training dataset -> 25 centres, as 1b
N_EVAL_DS, N_EVAL_RUNS = 8, 2
EPOCHS, BATCH, LR, EVAL_EVERY = 15, 8, 2e-3, 2
PHASE1B_NODE = {"44b6": 0.8826, "6bba": 0.9148}   # notes/20 pu best, by EVAL embryo
def stable_key(n): return int(hashlib.sha1(n.encode()).hexdigest(), 16)
ORDERED = {e: sorted(v, key=stable_key) for e, v in by_embryo.items()}
SPLITS = {e: {"train": v, "eval": v[24:24 + N_EVAL_DS]} for e, v in ORDERED.items()}
for e in SPLITS:
    print(f"{e}: {len(SPLITS[e]['train'])} train datasets, {len(SPLITS[e]['eval'])} eval")
""")

md("""## 1. Build the tensors — consecutive runs, stored once

A radius-1 window needs three frames per sample. Storing each sample's stack outright
would triple memory for no reason: within a run of consecutive frames, neighbours are
**already stored**, so the window is an index rather than a copy.

Each run loads `RUN_LEN + 2` frames and yields `RUN_LEN` centres, every one of which has
real neighbours. Boundary clamping happens only where the movie actually ends, matching
what `predict_dataset` does at inference — a model trained on mid-movie duplicates it never
meets again would be off-distribution in the one place nothing checks.

Both arms train on **the same tensors**. `r=0` reads only the centre channel.
""")

code(r"""
# Returns (vols, idx, tgts, masks, priors, meta), where vols[idx[i]] is sample i's
# (3, Z, Y, X) window and idx[i, 1] indexes its centre frame.
def build(names_subset, n_runs, seed=0):
    vols, idx, tgts, masks, priors, meta = [], [], [], [], [], []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for j, name in enumerate(names_subset):
        arr, _a, scale, voxel_um, q_lo, q_hi = open_movie(TRAIN / f"{name}.zarr", CFG)
        T = int(arr.shape[0])
        gt = read_geff(TRAIN / f"{name}.geff")
        est = read_estimated_nodes(TRAIN / f"{name}.geff")
        # P(annotated | cell), measured per dataset. notes/17 §2: 0.0060 on 44b6 and
        # 0.1182 on 6bba -- a 20x spread that a default would erase.
        prior = float(np.clip(len(gt.t) / max(est, 1.0), 1e-4, TCFG.pu_clip)) if est else 0.05
        gt_ds = gt.zyx / np.array(DS, float)

        L = min(RUN_LEN, T)
        starts = sorted(rng.choice(max(1, T - L + 1),
                                   size=min(n_runs, max(1, T - L + 1)),
                                   replace=False).tolist())
        # Per DATASET, not per run: starts are drawn without replacement but runs still
        # overlap freely (starts 3 and 4 at L=5 share four frames), and a per-run table
        # would store every shared frame once per run. Sharing also means idx[:, 2] and
        # the next centre's idx[:, 1] are the SAME entry rather than two copies.
        slot, norms, dogs, done = {}, {}, {}, set()
        for s in starts:
            # Load s-1 .. s+L, CLAMPED -- identical to Config.temporal_radius's rule, so a
            # run touching t=0 sees the same duplicated frame inference will show it.
            frames = [int(min(max(t, 0), T - 1)) for t in range(s - 1, s + L + 1)]
            centre_ts = set(range(s, s + L))
            for t in frames:
                # A frame first seen as a NEIGHBOUR has no DoG; if a later run makes it a
                # centre it needs one, so recompute rather than skipping on `slot` alone.
                if t in slot and not (t in centre_ts and t not in dogs):
                    continue
                raw = load_frame(arr, t, CFG, q_lo, q_hi)
                if t in centre_ts:
                    # ONE dog_response call, exactly as every earlier training notebook
                    # did -- so the stored volume and the mask's DoG come from the same
                    # normalisation. Calling frame_norm first and dog_response after would
                    # normalise twice and quietly change the mask.
                    norms[t], dogs[t] = dog_response(raw, voxel_um, CFG)
                else:
                    # Neighbours are input channels and nothing else, so they need the
                    # same normalisation but not the filtering.
                    norms[t] = pc.frame_norm(raw, CFG)
                if t not in slot:
                    slot[t] = len(vols)
                    vols.append(norms[t].astype(np.float16))
            for t in range(s, s + L):
                if t in done:
                    continue        # already a centre via an overlapping run
                done.add(t)
                norm, dog = norms[t], dogs[t]
                centres = gt_ds[gt.t == t]
                tgt = make_target(centres, norm.shape, voxel_um, TCFG.pos_radius_um)
                idx.append([slot[max(t - 1, 0)], slot[t], slot[min(t + 1, T - 1)]])
                tgts.append(tgt)
                masks.append(make_loss_mask(norm, dog, tgt, voxel_um, TCFG))
                priors.append(prior)
                meta.append({"name": name, "t": t, "voxel_um": voxel_um,
                             "run": s, "T": T})
        if j % 25 == 0 or j == len(names_subset) - 1:
            print(f"    {j+1:>3}/{len(names_subset)} datasets, {len(idx):>5} centres from "
                  f"{len(vols):>5} frames ({time.time()-t0:.0f}s)", flush=True)
    return (np.stack(vols), np.array(idx, np.int64), np.stack(tgts), np.stack(masks),
            np.array(priors, np.float32), meta)

GT_CACHE = {}
def gt_of(name):
    if name not in GT_CACHE:
        GT_CACHE[name] = read_geff(TRAIN / f"{name}.geff")
    return GT_CACHE[name]
""")

md("""## 2. The DoG baseline, in paired terms

The gate has to be measured in the same units as the thing being gated. `notes/20`'s guard
compared node recall against DoG's node recall; this one compares **paired** recall against
DoG's paired recall, on the same eval runs, at DoG's own detection count per frame.

`position` is printed alongside. DoG measured **+25.2 %** in `notes/21`; anything close to
that here confirms the eval set is behaving like the full corpus.
""")

code(r"""
# Per-sample (coords, t) for one model, at DoG's own per-frame detection budget.
def detections_for(model, ev, radius, caps):
    V, IDX, TG, MK, PR, ME = ev
    out = []
    for i, m in enumerate(ME):
        if radius == 0:
            x = V[IDX[i, 1]].astype(np.float32)
        else:
            x = V[IDX[i]].astype(np.float32)
        prob = predict_volume(model, x, DEV)
        coords, _ = peaks_from_prob(prob, m["voxel_um"], CFG.min_separation_um,
                                    threshold=1e-6, cap=int(caps[i]))
        out.append((coords, m["t"]))
    return out

# Node recall AND paired recall, pooled over datasets.
#
# Paired recall is accumulated per dataset because GT edge indices are per dataset, and
# `frames=` restricts it to the frames actually evaluated -- an edge leaving the run has no
# chance of being matched, and charging the model for it would make every arm look
# incoherent for a reason unrelated to the model.
def score_detections(dets, ME):
    by_ds = {}
    for (coords, t), m in zip(dets, ME):
        by_ds.setdefault(m["name"], {"t": [], "zyx": [], "frames": set(),
                                     "vox": m["voxel_um"]})
        d = by_ds[m["name"]]
        d["t"].append(np.full(len(coords), t, float))
        d["zyx"].append(coords)
        d["frames"].add(int(t))
    recs, pair_hits, pair_tot, node_hits, node_tot = [], 0, 0, 0, 0
    for name, d in by_ds.items():
        g = gt_of(name)
        pt = np.concatenate(d["t"]) if d["t"] else np.zeros(0)
        pz = np.vstack(d["zyx"]) if any(len(z) for z in d["zyx"]) else np.zeros((0, 3))
        keep = np.isin(g.t, sorted(d["frames"]))
        gt_ds = g.zyx[keep] / np.array(DS, float)
        gt_t = g.t[keep]
        if not len(gt_t):
            continue
        recs.append(recall_at_budget(pt, pz, gt_t, gt_ds, d["vox"]))
        # paired_recall needs the FULL node table so edge indices stay valid; `frames=`
        # does the restriction instead of pre-filtering the nodes.
        pr = paired_recall(pt, pz, g.t, g.zyx / np.array(DS, float), g.edges, d["vox"],
                           frames=sorted(d["frames"]))
        if pr["n_edges"]:
            pair_hits += pr["paired"] * pr["n_edges"]
            pair_tot += pr["n_edges"]
            node_hits += pr["node"] * pr["n_edges"]
            node_tot += pr["n_edges"]
    if not pair_tot:
        return {"node": float(np.mean(recs)) if recs else float("nan"),
                "paired": float("nan"), "position": float("nan"), "n_edges": 0}
    p = pair_hits / pair_tot
    r = node_hits / node_tot
    indep, denom = r * r, r - r * r
    return {"node": float(np.mean(recs)), "paired": float(p),
            "paired_node": float(r), "independent": float(indep),
            "position": float((p - indep) / denom) if denom > 1e-9 else float("nan"),
            "n_edges": int(pair_tot)}

EVAL, DOG = {}, {}
for emb in sorted(SPLITS):
    print(f"  eval runs for {emb} ...", flush=True)
    EVAL[emb] = build(SPLITS[emb]["eval"], N_EVAL_RUNS, seed=2)
    V, IDX, TG, MK, PR, ME = EVAL[emb]
    caps, dets = [], []
    for i, m in enumerate(ME):
        coords, _ = detect_frame_dog(V[IDX[i, 1]].astype(np.float32), m["voxel_um"], CFG)
        caps.append(len(coords))
        dets.append((coords, m["t"]))
    DOG[emb] = score_detections(dets, ME) | {"caps": np.array(caps)}
    d = DOG[emb]
    print(f"  DoG on {emb}: node {d['node']:.4f}  paired {d['paired']:.4f}  "
          f"position {d['position']:+.1%}  ({d['n_edges']:,} GT links, "
          f"{np.mean(caps):.0f} det/frame)", flush=True)
print("\n  notes/21 measured DoG at +25.2% on the full corpus; a similar number here "
      "means this eval slice behaves like it.")
""")

md("""## 3. Train, selecting on paired recall

The checkpoint that gets saved is the one with the best **paired** recall, not the best
node recall. Both are printed every evaluation so the two can be read against each other —
prediction 2 is graded on exactly that divergence.

The sub-DoG guard from `notes/20` §4 is kept and retargeted: a checkpoint whose paired
recall does not beat DoG's is not written at all. That guard already earned its place once,
refusing `masked`/`44b6` before it could reach the scorer and produce a clean-looking
refutation.
""")

code(r"""
def train_one(radius, train_emb, eval_emb, data):
    V, IDX, TG, MK, PR, ME = data
    torch.manual_seed(0); np.random.seed(0)
    in_ch = 1 if radius == 0 else 2 * radius + 1
    model = UNet3D(base=16, depth=3, in_ch=in_ch).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    fn = LOSSES[LOSS_NAME]
    n, curve, t0 = len(IDX), [], time.time()
    best = {"paired": -1.0, "epoch": -1, "state": None, "node": float("nan")}
    ev = EVAL[eval_emb]
    caps = DOG[eval_emb]["caps"]
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(n)
        tot = 0.0
        for s in range(0, n, BATCH):
            b = order[s:s + BATCH]
            if radius == 0:
                x = torch.as_tensor(V[IDX[b, 1]].astype(np.float32), device=DEV).unsqueeze(1)
            else:
                # (B, 3, Z, Y, X) gathered by INDEX -- the neighbouring frames are already
                # in V, so the window costs no extra memory.
                x = torch.as_tensor(V[IDX[b]].astype(np.float32), device=DEV)
            y = torch.as_tensor(TG[b], device=DEV).unsqueeze(1).float()
            m = torch.as_tensor(MK[b], device=DEV).unsqueeze(1).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = fn(logits.float(), y, m, prior=float(PR[b].mean()))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(b)
        if ep % EVAL_EVERY == EVAL_EVERY - 1 or ep == EPOCHS - 1:
            sc = score_detections(detections_for(model, ev, radius, caps), ev[5])
            curve.append({"epoch": ep, "loss": tot / n, **{k: v for k, v in sc.items()
                                                           if k != "caps"}})
            star = ""
            # SELECTED ON PAIRED, not node. notes/21 §1: node recall was identical across
            # a 0.074 edge-Jaccard gap, so selecting on it optimises a number that cannot
            # see the failure this run exists to fix.
            if sc["paired"] > best["paired"]:
                best = {"paired": sc["paired"], "node": sc["node"], "epoch": ep,
                        "position": sc["position"],
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                star = "  <- best"
            print(f"    r={radius}/{train_emb} epoch {ep:>2}  loss {tot/n:.5f}  "
                  f"paired {sc['paired']:.4f}  node {sc['node']:.4f}  "
                  f"pos {sc['position']:+.1%}  ({time.time()-t0:.0f}s){star}", flush=True)
    return model, curve, best

RESULTS = {}
embryos = sorted(SPLITS)
for train_emb in embryos:
    eval_emb = [e for e in embryos if e != train_emb][0]
    print(f"\n=== fold: train on {train_emb} "
          f"({len(SPLITS[train_emb]['train'])} datasets) -> evaluate on {eval_emb} ===",
          flush=True)
    data = build(SPLITS[train_emb]["train"], N_RUNS, seed=1)
    V, IDX, TG, MK, PR, _ = data
    print(f"  {len(IDX)} centres from {len(V)} frames, pos {TG.mean():.4%}, "
          f"mask keeps {MK.mean():.1%}, prior med {np.median(PR):.4f} "
          f"({V.nbytes/1e9:.2f} GB fp16)", flush=True)

    for radius in RADII:
        model, curve, best = train_one(radius, train_emb, eval_emb, data)
        base = DOG[eval_emb]
        RESULTS[f"r{radius}|{train_emb}"] = {
            "radius": radius, "train_emb": train_emb, "eval_emb": eval_emb,
            "best_paired": best["paired"], "best_node": best["node"],
            "best_position": best["position"], "best_epoch": best["epoch"],
            "dog_paired": base["paired"], "dog_node": base["node"],
            "dog_position": base["position"],
            "paired_delta": best["paired"] - base["paired"],
            "phase1b_node": PHASE1B_NODE.get(eval_emb),
            "n_centres": int(len(IDX)), "curve": curve}
        print(f"  r={radius}/{train_emb}: BEST paired {best['paired']:.4f} @epoch "
              f"{best['epoch']}  node {best['node']:.4f}  pos {best['position']:+.1%}  "
              f"(DoG paired {base['paired']:.4f})", flush=True)

        # The sub-DoG guard applies to CANDIDATES, not to the control. notes/20 added it to
        # stop unusable weights (0.2654 against DoG's 0.7696) reaching a scorer and
        # masquerading as a clean refutation. But r=0 here is the baseline the candidate is
        # measured against -- a measurement, not a shipment -- and refusing to save it
        # destroys the comparison rather than protecting it.
        #
        # This is not hypothetical: the first run of this notebook refused r0/44b6 at
        # paired 0.8529 against DoG's 0.8550, a -0.0021 tie, which left the scorer unable
        # to compare radii on that fold at all.
        is_control = radius == min(RADII)
        if best["paired"] > base["paired"] or is_control:
            torch.save({"state_dict": best["state"], "loss": LOSS_NAME,
                        "train_emb": train_emb, "base": 16, "depth": 3,
                        "in_ch": 1 if radius == 0 else 2 * radius + 1,
                        "temporal_radius": radius, "downsample": list(DS),
                        "best_epoch": best["epoch"], "best_paired": best["paired"],
                        "best_recall": best["node"], "dog_paired": base["paired"],
                        "dog_recall": base["node"]},
                       WORK / f"claude_temporal_r{radius}_{train_emb}.pt")
            below = best["paired"] <= base["paired"]
            print(f"    saved checkpoint from epoch {best['epoch']}"
                  + ("  (control, saved despite being <= DoG)" if below else ""),
                  flush=True)
        else:
            print(f"    NOT SAVED: paired {best['paired']:.4f} <= DoG "
                  f"{base['paired']:.4f}", flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()

    del data, V, IDX, TG, MK, PR
    gc.collect()
""")

md("""## 4. Grade the pre-registered predictions""")

code(r"""
PREDICTIONS = """ + json.dumps([[c, w, k] for c, w, k in PREDICTIONS]) + r"""

print(f"{'arm':<14} {'paired':>8} {'node':>8} {'position':>10} {'vs DoG':>9} {'ep':>3}")
print("-" * 56)
for emb in embryos:
    other = [e for e in embryos if e != emb][0]
    print(f"{'DoG/'+other:<14} {DOG[other]['paired']:>8.4f} {DOG[other]['node']:>8.4f} "
          f"{DOG[other]['position']:>9.1%} {'—':>9} {'—':>3}")
    for radius in RADII:
        r_ = RESULTS.get(f"r{radius}|{emb}")
        if r_:
            print(f"{'r'+str(radius)+'/'+emb:<14} {r_['best_paired']:>8.4f} "
                  f"{r_['best_node']:>8.4f} {r_['best_position']:>9.1%} "
                  f"{r_['paired_delta']:>+9.4f} {r_['best_epoch']:>3}")

def arm(radius, emb):
    return RESULTS.get(f"r{radius}|{emb}")

pairs = [(arm(0, e), arm(1, e)) for e in embryos]
pairs = [(a, b) for a, b in pairs if a and b]

d_paired = [b["best_paired"] - a["best_paired"] for a, b in pairs]
d_node = [b["best_node"] - a["best_node"] for a, b in pairs]
d_pos = [b["best_position"] - a["best_position"] for a, b in pairs]

print(f"\nr=1 minus r=0, per fold:")
for (a, b), dp, dn, dq in zip(pairs, d_paired, d_node, d_pos):
    print(f"  train {a['train_emb']}: paired {dp:+.4f}   node {dn:+.4f}   "
          f"position {dq:+.1%}")

# Prediction 1: coherence improves on BOTH folds. One fold is an anecdote.
coherence_improves = bool(len(pairs) == len(embryos) and all(d > 0 for d in d_paired))

VERDICTS = {
    "coherence_improves": coherence_improves,
    # Prediction 2 is about the SHAPE of a gain, so it is only answerable if there was
    # one. Without this gate a run where everything got worse still prints "CONFIRMED:
    # the gain arrives through coherence" -- mean(d_paired) > mean(d_node) holds happily
    # when both are negative, and node recall trivially clears a +0.02 ceiling by falling.
    # That is not a confirmation, it is a category error, and it would read as support for
    # the mechanism in exactly the run that refuted it.
    "gain_is_coherence": (None if not coherence_improves else
                          bool(pairs and max(d_node) < 0.02
                               and np.mean(d_paired) > np.mean(d_node))),
    # Prediction 3: the control reproduces phase 1b, so the sampling change is neutral.
    "control_reproduces": bool(pairs and all(
        a["phase1b_node"] is not None and abs(a["best_node"] - a["phase1b_node"]) < 0.05
        for a, _ in pairs)),
}

print()
for i, (claim, why, key) in enumerate(PREDICTIONS, 1):
    v = VERDICTS.get(key)
    label = "N/A" if v is None else ("CONFIRMED" if v else "FALSIFIED")
    print(f"{i}. {label:<10} {claim}"
          + ("" if v is not None else
             "  (unanswerable: prediction 1 failed, so there is no gain to attribute)"))

if not VERDICTS["control_reproduces"]:
    print("\n!! THE CONTROL DID NOT REPRODUCE. Read predictions 1 and 2 as being about "
          "the run-sampling change, NOT about temporal input — the two are confounded "
          "and this run cannot separate them.")

blob = json.dumps({"results": RESULTS,
                   "dog": {e: {k: v for k, v in DOG[e].items() if k != "caps"}
                           for e in DOG},
                   "verdicts": VERDICTS,
                   "d_paired": d_paired, "d_node": d_node, "d_position": d_pos,
                   "setup": {"radii": list(RADII), "loss": LOSS_NAME,
                             "run_len": RUN_LEN, "n_runs": N_RUNS,
                             "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                             "select_on": "paired_recall"}}, indent=2, default=float)
(WORK / "claude_temporal_train_results.json").write_text(blob)
print("\n" + "=" * 70)
print(blob)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
for i, c in enumerate(json.loads(OUT.read_text())["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try: ast.parse("\n".join("pass" if l.strip().startswith("!") else l for l in src.splitlines()))
        except SyntaxError as e: raise SystemExit(f"cell {i}: {e}\n{src}")
