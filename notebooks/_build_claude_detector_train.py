"""Build notebooks/claude_detector_train.ipynb."""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_detector_train.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

# The pre-registered claims live HERE, once, and both the header and the grading cell are
# generated from this list. notes/18 §1: the previous notebook declared new predictions in
# its header and graded the old ones, which is not pre-registration at all.
PREDICTIONS = [
    ("Both folds beat DoG at 1.0x cap.",
     "Phase 0b had only the 6bba-trained model winning (+0.0919); the 44b6-trained one "
     "lost at -0.0155. Falsified if either fold is still negative.",
     "both_beat_dog"),
    ("Both folds beat their phase 0b counterpart.",
     "masked scored +0.0919 (train 6bba) and -0.0155 (train 44b6) on the same eval sets. "
     "Falsified if either fails to improve on that.",
     "both_beat_phase0b"),
    ("No overfitting at 40 epochs.",
     "notes/18 §4 argues the model is data-starved, not capacity-starved, so 13x more "
     "supervision should not start memorising. Falsified if held-out recall peaks before "
     "the final checkpoint.",
     "no_overfit"),
]

md(r"""
# Phase 1 — the same model, 13x the data

`notes/18` §4: the binding constraint is supervision, not capacity. Phase 0b trained on
**24 of 199 datasets and 10 of 100 frames** — 1.2 % of the available frames — which works
out to roughly **590 annotated cells** for the `44b6`-trained model and **1,860** for the
`6bba`-trained one. The weak direction was exactly the small one.

This run uses **all 199 datasets and 25 frames each**, and changes nothing else. Same
architecture (base 16, depth 3, ~350 k parameters), same `masked` loss, same targets, same
optimiser. Phase 0b's other mistake was moving the threshold and the epoch count together
so that nothing could be attributed to either; this one moves one thing.

`masked` is the loss because `notes/18` picked it: best arm, improves with training where
`naive` degrades, and unlike `pu` it needs no per-dataset prior — which matters when that
prior varies 20x between embryos.

Folds stay **leave-one-embryo-out**: two models, each trained on one embryo, each scored
on the other. The hidden test is two embryos we have never seen, so a model that has seen
both would flatter itself.
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
        torch.cuda.synchronize()
        print("  GPU smoke test passed")
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
TCFG = TargetConfig(loss="masked")
LOSS_NAME = "masked"
N_FRAMES = 25          # of 100; phase 0b used 10 from 24 datasets
N_EVAL_DS, N_EVAL_FR = 8, 8    # the SAME eval slice phase 0b used, so deltas compare
EPOCHS, BATCH, LR = 40, 8, 2e-3
PHASE0B = {"6bba": +0.0919, "44b6": -0.0155}   # masked, by TRAINING embryo
def stable_key(n): return int(hashlib.sha1(n.encode()).hexdigest(), 16)
ORDERED = {e: sorted(v, key=stable_key) for e, v in by_embryo.items()}
SPLITS = {e: {"train": v, "eval": v[24:24 + N_EVAL_DS]} for e, v in ORDERED.items()}
for e in SPLITS:
    print(f"{e}: {len(SPLITS[e]['train'])} train datasets (all), "
          f"{len(SPLITS[e]['eval'])} eval datasets")
""")

md("""## 1. Build the tensors

Volumes are stored as **float16** — 199 datasets x 25 frames is ~5,000 volumes of 64³, and
float32 would be 5.2 GB for the images alone before targets and masks. Each fold's data is
built, trained on, and freed before the next, so peak memory is one fold rather than two.
""")

code(r"""
def build(names_subset, n_frames, seed=0, want_gt_index=False):
    vols, tgts, masks, meta = [], [], [], []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for j, name in enumerate(names_subset):
        arr, _a, scale, voxel_um, q_lo, q_hi = open_movie(TRAIN / f"{name}.zarr", CFG)
        T = int(arr.shape[0])
        gt = read_geff(TRAIN / f"{name}.geff")
        gt_ds = gt.zyx / np.array(DS, float)
        for t in sorted(rng.choice(T, size=min(n_frames, T), replace=False).tolist()):
            norm, dog = dog_response(load_frame(arr, t, CFG, q_lo, q_hi), voxel_um, CFG)
            centres = gt_ds[gt.t == t]
            tgt = make_target(centres, norm.shape, voxel_um, TCFG.pos_radius_um)
            vols.append(norm.astype(np.float16))
            tgts.append(tgt)
            masks.append(make_loss_mask(norm, dog, tgt, voxel_um, TCFG))
            meta.append({"name": name, "t": t, "voxel_um": voxel_um})
        if j % 25 == 0 or j == len(names_subset) - 1:
            print(f"    {j+1:>3}/{len(names_subset)} datasets, {len(vols):>5} volumes "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return np.stack(vols), np.stack(tgts), np.stack(masks), meta

GT_CACHE = {}
def gt_of(name):
    if name not in GT_CACHE:
        GT_CACHE[name] = read_geff(TRAIN / f"{name}.geff")
    return GT_CACHE[name]

# DoG baseline on each embryo's eval slice -- the number every fold is measured against,
# and the source of the per-frame cap. Recomputed here rather than hardcoded from
# notes/18 so a data change cannot silently invalidate the comparison.
EVAL, DOG = {}, {}
for emb in sorted(SPLITS):
    print(f"  eval slice for {emb} ...", flush=True)
    EVAL[emb] = build(SPLITS[emb]["eval"], N_EVAL_FR, seed=2)
    v, tg, mk, me = EVAL[emb]
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

md("""## 2. Train one model per fold

Held-out recall is measured every 5 epochs, which is what makes prediction 3 checkable —
"no overfitting" is a claim about the shape of a curve, not a single endpoint.
""")

code(r"""
def evaluate(model, emb):
    v, tg, mk, me = EVAL[emb]
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

RESULTS = {}
embryos = sorted(SPLITS)
for train_emb in embryos:
    eval_emb = [e for e in embryos if e != train_emb][0]
    print(f"\n=== fold: train on {train_emb} ({len(SPLITS[train_emb]['train'])} datasets) "
          f"-> evaluate on {eval_emb} ===", flush=True)
    V, TG, MK, _ = build(SPLITS[train_emb]["train"], N_FRAMES, seed=1)
    n_cells = TG.sum() / 7.0
    print(f"  {len(V)} volumes, pos {TG.mean():.4%}, mask keeps {MK.mean():.1%}, "
          f"~{n_cells:,.0f} annotated cells "
          f"({V.nbytes/1e9:.2f} GB fp16 + {TG.nbytes/1e9:.2f} + {MK.nbytes/1e9:.2f})",
          flush=True)

    torch.manual_seed(0); np.random.seed(0)
    model = UNet3D(base=16, depth=3).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    fn = LOSSES[LOSS_NAME]
    n, curve, t0 = len(V), [], time.time()
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
            loss = fn(logits.float(), y, m)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(b)
        if ep % 5 == 4 or ep == EPOCHS - 1:
            r, c = evaluate(model, eval_emb)
            curve.append({"epoch": ep, "loss": tot / n, "recall": r, "dets": c})
            print(f"    epoch {ep:>2}  loss {tot/n:.5f}  held-out recall {r:.4f} "
                  f"({c:.0f} det)  ({time.time()-t0:.0f}s)", flush=True)

    r, c = evaluate(model, eval_emb)
    base = DOG[eval_emb]["recall"]
    RESULTS[train_emb] = {
        "eval_emb": eval_emb, "recall": r, "dets": c, "dog_recall": base,
        "delta": r - base, "phase0b_delta": PHASE0B[train_emb],
        "n_volumes": int(len(V)), "n_cells": float(n_cells), "curve": curve}
    print(f"  FINAL: recall {r:.4f} vs DoG {base:.4f}  delta {r-base:+.4f}   "
          f"(phase 0b was {PHASE0B[train_emb]:+.4f})", flush=True)
    torch.save({"state_dict": model.state_dict(), "loss": LOSS_NAME,
                "train_emb": train_emb, "base": 16, "depth": 3,
                "downsample": list(DS), "n_volumes": int(len(V))},
               WORK / f"claude_unet_{LOSS_NAME}_{train_emb}.pt")
    del V, TG, MK, model, opt
    gc.collect(); torch.cuda.empty_cache()
""")

md("""## 3. How long does inference take on CPU?

The submission has no internet, so it cannot install a torch that works on a P100, and the
image's torch is GPU-incompatible with that card. CPU inference may be the only offline
path — so measure it here rather than discover it at hour three of a scored rerun.
""")

code(r"""
import copy
probe = UNet3D(base=16, depth=3)
probe.load_state_dict(torch.load(WORK / f"claude_unet_{LOSS_NAME}_{embryos[0]}.pt",
                                 map_location="cpu")["state_dict"])
probe.eval()
cpu = torch.device("cpu")
v = EVAL[embryos[1]][0][0].astype(np.float32)
_ = predict_volume(probe, v, cpu, amp=False)          # warm up
t0 = time.time()
N_PROBE = 5
for _ in range(N_PROBE):
    _ = predict_volume(probe, v, cpu, amp=False)
per = (time.time() - t0) / N_PROBE
total_h = per * 200 * 100 / 3600
print(f"CPU inference: {per*1000:.0f} ms per 64^3 volume")
print(f"  -> a ~200-dataset test set x 100 frames = {per*200*100:,.0f}s = {total_h:.2f} h")
print(f"  -> {'FITS' if total_h < 8 else 'DOES NOT FIT'} inside the 12 h submission cap "
      f"with room for detection, linking and CSV writing")
""")

code(r"""
PREDICTIONS = """ + json.dumps([[c, w, k] for c, w, k in PREDICTIONS]) + r"""
print("=" * 78)
print(f"{'train on':<10}{'eval on':<10}{'recall':>9}{'DoG':>9}{'delta':>10}{'phase 0b':>11}{'gain':>9}")
for te in sorted(RESULTS):
    r = RESULTS[te]
    print(f"{te:<10}{r['eval_emb']:<10}{r['recall']:>9.4f}{r['dog_recall']:>9.4f}"
          f"{r['delta']:>+10.4f}{r['phase0b_delta']:>+11.4f}"
          f"{r['delta']-r['phase0b_delta']:>+9.4f}")

verdicts = {}
verdicts["both_beat_dog"] = all(RESULTS[te]["delta"] > 0 for te in RESULTS)
verdicts["both_beat_phase0b"] = all(
    RESULTS[te]["delta"] > RESULTS[te]["phase0b_delta"] for te in RESULTS)
def peaked_early(c):
    best = max(range(len(c)), key=lambda i: c[i]["recall"])
    return best < len(c) - 1
verdicts["no_overfit"] = not any(peaked_early(RESULTS[te]["curve"]) for te in RESULTS)

# Graded against the SAME list the header rendered -- notes/18 §1, where the header
# declared one set of predictions and the summary cell graded another notebook's.
for i, (claim, why, key) in enumerate(PREDICTIONS):
    print(f"\n=== prediction {i+1}: {claim} ===")
    print(f"    {why}")
    print(f"  -> {'CONFIRMED' if verdicts[key] else 'FALSIFIED'}")

for te in sorted(RESULTS):
    print(f"\nheld-out recall curve, trained on {te}:")
    for pt in RESULTS[te]["curve"]:
        print(f"    epoch {pt['epoch']:>2}  loss {pt['loss']:.5f}  recall {pt['recall']:.4f}")

payload = {"results": RESULTS, "dog": {e: {"recall": DOG[e]["recall"],
                                           "mean_caps": float(DOG[e]["caps"].mean())}
                                       for e in DOG},
           "verdicts": verdicts,
           "setup": {"loss": LOSS_NAME, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                     "n_frames": N_FRAMES, "params": count_params(UNet3D(base=16, depth=3))}}
blob = json.dumps(payload, indent=2, default=str)
(WORK / "claude_detector_train_results.json").write_text(blob)
# kaggleusercontent is proxy-blocked from the agent container, so output FILES cannot be
# fetched -- only logs. Print it (notes/17 §4).
print("\n===== RESULTS JSON BEGIN =====")
print(blob)
print("===== RESULTS JSON END =====")
print("\nweights written:", sorted(p.name for p in WORK.glob("claude_unet_*.pt")))
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
