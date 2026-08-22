"""Build notebooks/claude_detector_earlystop.ipynb."""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_detector_earlystop.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

# One list; the header and the grading cell are both generated from it (notes/18 §1).
PREDICTIONS = [
    ("`pu` does not collapse.",
     "notes/19 §2 blames masked's collapse on its unconstrained region; pu constrains the "
     "whole volume, so it should degrade gracefully. Measured as: pu's final-epoch recall "
     "is within 0.05 of its own best. Falsified if pu falls away like masked did.",
     "pu_stable"),
    ("Best-checkpoint selection recovers phase 0b's numbers.",
     "The bug was saving the last checkpoint, not the best. With that fixed, the best "
     "masked checkpoint should reach at least +0.0919 (train 6bba) and -0.0155 "
     "(train 44b6). Falsified if either falls short.",
     "recovers_phase0b"),
    ("`pu` beats `masked` at scale.",
     "Phase 0b ranked masked above pu on 240 volumes. If notes/19 §2 has the mechanism "
     "right, the collapse scales with training-set size and the ranking should reverse "
     "here. Falsified if masked still wins on both folds.",
     "pu_beats_masked"),
]

md(r"""
# Phase 1b — select the checkpoint, and test the collapse mechanism

`notes/19`: phase 1 trained on 13x the data and every prediction failed. Two causes, both
identified, neither of them "the approach is wrong":

1. **The notebook saved the final checkpoint rather than the best one.** It measured
   held-out recall every 5 epochs and then wrote `state_dict()` after the loop. The saved
   `6bba` model scores **0.2654** where its epoch-4 checkpoint scored **0.8106**.
2. **`masked` has an unconstrained region.** The mask keeps positives and *clearly empty*
   voxels — 26-37 % of the volume. The ambiguous middle, where unannotated real cells live
   and where detection actually happens, contributes no gradient. The model reaches
   **exactly 0.00000 loss** by solving bright-vs-dark, and then drifts freely in the middle
   while the loss reports perfection.

`pu` has no such hole: it treats unlabelled voxels as a known mixture rather than ignoring
them, and in phase 0b its loss plateaued at 0.0573 instead of collapsing to zero. So
running `pu` at scale is a **test of the diagnosed mechanism**, not a hyperparameter sweep.

Data is identical to phase 1 (199 datasets x 25 frames). The only changes are:

- the best checkpoint by held-out recall is saved, with the epoch recorded
- held-out recall is measured every **2** epochs, since the peak was at 4
- **the per-dataset prior is passed to the loss again** — phase 1 dropped it, which was
  harmless for `masked` but would have run `pu` on a default 0.05 instead of the measured
  `n_annotated / estimated_number_of_nodes`, a quantity that varies **20x between
  embryos** and is the whole basis of the nnPU correction
- a checkpoint worse than DoG is **not saved at all**, so unusable weights cannot reach the
  scorer and masquerade as a clean refutation
- 15 epochs rather than 40
""")

md("## Pre-registered\n\n" + "\n".join(
    f"{i+1}. **{c}** {w}" for i, (c, w, _) in enumerate(PREDICTIONS)))

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
from pipeline.classical import Config, dog_response, load_frame, open_movie, detect_frame_dog
from pipeline.detector import TargetConfig, make_loss_mask, make_target, peaks_from_prob, recall_at_budget
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
N_FRAMES = 25
N_EVAL_DS, N_EVAL_FR = 8, 8
EPOCHS, BATCH, LR, EVAL_EVERY = 15, 8, 2e-3, 2
LOSS_NAMES = ("masked", "pu")
PHASE0B = {"6bba": +0.0919, "44b6": -0.0155}   # masked, by TRAINING embryo
def stable_key(n): return int(hashlib.sha1(n.encode()).hexdigest(), 16)
ORDERED = {e: sorted(v, key=stable_key) for e, v in by_embryo.items()}
SPLITS = {e: {"train": v, "eval": v[24:24 + N_EVAL_DS]} for e, v in ORDERED.items()}
for e in SPLITS:
    print(f"{e}: {len(SPLITS[e]['train'])} train datasets, {len(SPLITS[e]['eval'])} eval")
""")

md("""## 1. Build the tensors

Per-dataset priors are collected again — phase 1 dropped them, and `pu` is meaningless
without the measured value. Data is built once per fold and both losses train on it, so the
20-minute build is paid twice, not four times.
""")

code(r"""
def build(names_subset, n_frames, seed=0):
    vols, tgts, masks, priors, meta = [], [], [], [], []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for j, name in enumerate(names_subset):
        arr, _a, scale, voxel_um, q_lo, q_hi = open_movie(TRAIN / f"{name}.zarr", CFG)
        T = int(arr.shape[0])
        gt = read_geff(TRAIN / f"{name}.geff")
        est = read_estimated_nodes(TRAIN / f"{name}.geff")
        # P(annotated | cell), measured per dataset. notes/17 §2: this runs 0.0060 on
        # 44b6 and 0.1182 on 6bba -- a 20x spread that a default would erase.
        prior = float(np.clip(len(gt.t) / max(est, 1.0), 1e-4, TCFG.pu_clip)) if est else 0.05
        gt_ds = gt.zyx / np.array(DS, float)
        for t in sorted(rng.choice(T, size=min(n_frames, T), replace=False).tolist()):
            norm, dog = dog_response(load_frame(arr, t, CFG, q_lo, q_hi), voxel_um, CFG)
            centres = gt_ds[gt.t == t]
            tgt = make_target(centres, norm.shape, voxel_um, TCFG.pos_radius_um)
            vols.append(norm.astype(np.float16))
            tgts.append(tgt)
            masks.append(make_loss_mask(norm, dog, tgt, voxel_um, TCFG))
            priors.append(prior)
            meta.append({"name": name, "t": t, "voxel_um": voxel_um})
        if j % 25 == 0 or j == len(names_subset) - 1:
            print(f"    {j+1:>3}/{len(names_subset)} datasets, {len(vols):>5} volumes "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return (np.stack(vols), np.stack(tgts), np.stack(masks),
            np.array(priors, np.float32), meta)

GT_CACHE = {}
def gt_of(name):
    if name not in GT_CACHE:
        GT_CACHE[name] = read_geff(TRAIN / f"{name}.geff")
    return GT_CACHE[name]

EVAL, DOG = {}, {}
for emb in sorted(SPLITS):
    print(f"  eval slice for {emb} ...", flush=True)
    EVAL[emb] = build(SPLITS[emb]["eval"], N_EVAL_FR, seed=2)
    v, tg, mk, pr, me = EVAL[emb]
    caps, recs = [], []
    for i, m in enumerate(me):
        g = gt_of(m["name"])
        centres = g.zyx[g.t == m["t"]] / np.array(DS, float)
        coords, _ = detect_frame_dog(v[i].astype(np.float32), m["voxel_um"], CFG)
        caps.append(len(coords))
        if len(centres):
            recs.append(recall_at_budget(np.zeros(len(coords)), coords,
                                         np.zeros(len(centres)), centres, m["voxel_um"]))
    DOG[emb] = {"caps": np.array(caps), "recall": float(np.mean(recs))}
    print(f"  DoG on {emb}: recall {DOG[emb]['recall']:.4f} at {np.mean(caps):.0f} det/frame",
          flush=True)
""")

md("""## 2. Train, keeping the best checkpoint

The whole point of this run: the best `state_dict` by held-out recall is copied to CPU and
kept, and that is what gets written. Phase 1 measured this curve and then saved the final
model anyway, throwing away 0.5452 of recall on one fold.
""")

code(r"""
def evaluate(model, emb):
    v, tg, mk, pr, me = EVAL[emb]
    caps = DOG[emb]["caps"]
    recs, cnts = [], []
    for i, m in enumerate(me):
        g = gt_of(m["name"])
        centres = g.zyx[g.t == m["t"]] / np.array(DS, float)
        if not len(centres):
            continue
        prob = predict_volume(model, v[i].astype(np.float32), DEV)
        coords, _ = peaks_from_prob(prob, m["voxel_um"], CFG.min_separation_um,
                                    threshold=1e-6, cap=int(caps[i]))
        cnts.append(len(coords))
        recs.append(recall_at_budget(np.zeros(len(coords)), coords,
                                     np.zeros(len(centres)), centres, m["voxel_um"]))
    return float(np.mean(recs)), float(np.mean(cnts))

def train_one(loss_name, train_emb, eval_emb, V, TG, MK, PR):
    torch.manual_seed(0); np.random.seed(0)
    model = UNet3D(base=16, depth=3).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    fn = LOSSES[loss_name]
    n, curve, t0 = len(V), [], time.time()
    best = {"recall": -1.0, "epoch": -1, "dets": 0.0, "state": None}
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(n)
        tot = 0.0
        for s in range(0, n, BATCH):
            b = order[s:s + BATCH]
            x = torch.as_tensor(V[b].astype(np.float32), device=DEV).unsqueeze(1)
            y = torch.as_tensor(TG[b], device=DEV).unsqueeze(1).float()
            m = torch.as_tensor(MK[b], device=DEV).unsqueeze(1).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            # prior= restored: phase 1 omitted it, which silently defaults pu to 0.05.
            loss = fn(logits.float(), y, m, prior=float(PR[b].mean()))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(b)
        if ep % EVAL_EVERY == EVAL_EVERY - 1 or ep == EPOCHS - 1:
            r, c = evaluate(model, eval_emb)
            curve.append({"epoch": ep, "loss": tot / n, "recall": r, "dets": c})
            star = ""
            if r > best["recall"]:
                best = {"recall": r, "epoch": ep, "dets": c,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                star = "  <- best"
            print(f"    {loss_name}/{train_emb} epoch {ep:>2}  loss {tot/n:.5f}  "
                  f"held-out {r:.4f} ({c:.0f} det)  ({time.time()-t0:.0f}s){star}",
                  flush=True)
    return model, curve, best

RESULTS = {}
embryos = sorted(SPLITS)
for train_emb in embryos:
    eval_emb = [e for e in embryos if e != train_emb][0]
    print(f"\n=== fold: train on {train_emb} ({len(SPLITS[train_emb]['train'])} datasets) "
          f"-> evaluate on {eval_emb} ===", flush=True)
    V, TG, MK, PR, _ = build(SPLITS[train_emb]["train"], N_FRAMES, seed=1)
    print(f"  {len(V)} volumes, pos {TG.mean():.4%}, mask keeps {MK.mean():.1%}, "
          f"~{TG.sum()/7.0:,.0f} annotated cells, prior med {np.median(PR):.4f} "
          f"({V.nbytes/1e9:.2f} GB fp16)", flush=True)

    for loss_name in LOSS_NAMES:
        model, curve, best = train_one(loss_name, train_emb, eval_emb, V, TG, MK, PR)
        final = curve[-1]
        base = DOG[eval_emb]["recall"]
        RESULTS[f"{loss_name}|{train_emb}"] = {
            "loss": loss_name, "train_emb": train_emb, "eval_emb": eval_emb,
            "best_recall": best["recall"], "best_epoch": best["epoch"],
            "final_recall": final["recall"], "dog_recall": base,
            "best_delta": best["recall"] - base, "final_delta": final["recall"] - base,
            "collapse": best["recall"] - final["recall"],
            "phase0b_delta": PHASE0B[train_emb] if loss_name == "masked" else None,
            "n_volumes": int(len(V)), "curve": curve}
        print(f"  {loss_name}/{train_emb}: BEST {best['recall']:.4f} @epoch "
              f"{best['epoch']} (delta {best['recall']-base:+.4f})   "
              f"final {final['recall']:.4f}   collapse "
              f"{best['recall']-final['recall']:+.4f}", flush=True)

        # Refuse to ship weights worse than the classical detector they replace. Phase 1
        # saved a 0.2654-recall model against DoG's 0.7696, and the scorer would have
        # loaded it and produced a clean-looking refutation of the whole idea.
        if best["recall"] > base:
            torch.save({"state_dict": best["state"], "loss": loss_name,
                        "train_emb": train_emb, "base": 16, "depth": 3,
                        "downsample": list(DS), "best_epoch": best["epoch"],
                        "best_recall": best["recall"], "dog_recall": base},
                       WORK / f"claude_unet_{loss_name}_{train_emb}.pt")
            print(f"    saved checkpoint from epoch {best['epoch']}", flush=True)
        else:
            print(f"    NOT SAVED: best {best['recall']:.4f} <= DoG {base:.4f}", flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()

    del V, TG, MK, PR
    gc.collect()
""")

code(r"""
PREDICTIONS = """ + json.dumps([[c, w, k] for c, w, k in PREDICTIONS]) + r"""
print("=" * 92)
print(f"{'loss':<8}{'train':<7}{'eval':<7}{'best':>8}{'@ep':>5}{'final':>8}"
      f"{'collapse':>10}{'delta':>9}{'phase0b':>10}")
for k in sorted(RESULTS, key=lambda k: -RESULTS[k]["best_delta"]):
    r = RESULTS[k]
    p0 = f"{r['phase0b_delta']:+.4f}" if r["phase0b_delta"] is not None else "-"
    print(f"{r['loss']:<8}{r['train_emb']:<7}{r['eval_emb']:<7}{r['best_recall']:>8.4f}"
          f"{r['best_epoch']:>5}{r['final_recall']:>8.4f}{-r['collapse']:>+10.4f}"
          f"{r['best_delta']:>+9.4f}{p0:>10}")

def by_loss(ln):
    return {r["train_emb"]: r for r in RESULTS.values() if r["loss"] == ln}
mk, pu = by_loss("masked"), by_loss("pu")

verdicts = {
    "pu_stable": all(r["collapse"] <= 0.05 for r in pu.values()),
    "recovers_phase0b": all(mk[e]["best_delta"] >= PHASE0B[e] - 1e-9 for e in mk),
    "pu_beats_masked": all(pu[e]["best_delta"] > mk[e]["best_delta"] for e in pu),
}
for i, (claim, why, key) in enumerate(PREDICTIONS):
    print(f"\n=== prediction {i+1}: {claim} ===")
    print(f"    {why}")
    print(f"  -> {'CONFIRMED' if verdicts[key] else 'FALSIFIED'}")

print("\ncollapse (best - final), the notes/19 §2 mechanism:")
for ln in LOSS_NAMES:
    for e, r in sorted(by_loss(ln).items()):
        print(f"  {ln:<8} train {e}: {r['collapse']:+.4f}"
              f"   (best {r['best_recall']:.4f} @ep {r['best_epoch']} "
              f"-> final {r['final_recall']:.4f})")

for k in sorted(RESULTS):
    print(f"\ncurve {k}:")
    for pt in RESULTS[k]["curve"]:
        print(f"    epoch {pt['epoch']:>2}  loss {pt['loss']:.5f}  recall {pt['recall']:.4f}")

saved = sorted(p.name for p in WORK.glob("claude_unet_*.pt"))
print(f"\nweights written: {saved or 'NONE'}")
best_arm = max(RESULTS, key=lambda k: RESULTS[k]["best_delta"])
print(f"best arm: {best_arm} -> delta {RESULTS[best_arm]['best_delta']:+.4f}")
print("NOTE: claude_detector_score expects exactly ONE checkpoint per embryo. If both "
      "losses saved, keep only the winning loss's two files when attaching.")

payload = {"results": RESULTS, "verdicts": verdicts,
           "dog": {e: {"recall": DOG[e]["recall"],
                       "mean_caps": float(DOG[e]["caps"].mean())} for e in DOG},
           "saved": saved, "best_arm": best_arm,
           "setup": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "eval_every": EVAL_EVERY,
                     "n_frames": N_FRAMES,
                     "params": count_params(UNet3D(base=16, depth=3))}}
blob = json.dumps(payload, indent=2, default=str)
(WORK / "claude_detector_earlystop_results.json").write_text(blob)
# kaggleusercontent is proxy-blocked from the agent container: files cannot be fetched,
# only logs (notes/17 §4).
print("\n===== RESULTS JSON BEGIN =====")
print(blob)
print("===== RESULTS JSON END =====")
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
