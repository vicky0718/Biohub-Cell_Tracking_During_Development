"""Build notebooks/claude_detector_matched.ipynb."""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_detector_matched.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Phase 0b — the same three losses, this time actually at matched count

`notes/17` §1: phase 0 said "matched detection count" and did not deliver it. The DoG
count was passed as a *cap*, but `peaks_from_prob`'s `threshold=0.05` bound first, so the
arms spent between **51 % and 101 %** of the budget and the absolute-recall column
compared detectors at different densities — the exact error `notes/09` §2 exists to warn
about.

Corrected on the quantity that was meant to be measured, **every learned arm beat DoG on
recall per detection, in both directions, 1.05x to 1.76x** — including the arm that looked
worst. So the ranking is unknown; only the headline survives.

Two changes here, and nothing else, so the comparison is clean:

1. `threshold=1e-6`, so the **cap** decides the count, not the threshold.
2. The cap is swept at **0.5x / 1.0x / 1.5x** of DoG's per-frame count, because the whole
   question is a trade curve, not a point.

Also 25 epochs instead of 12 — phase 0's loss was still falling at the last one
(masked 0.0123 -> 0.0022 between epochs 4 and 11).

## Pre-registered

1. **At genuinely matched count (1.0x), every learned arm beats DoG.** *Falsified if* any
   arm fails to, which would mean phase 0's win came from spending fewer detections.
2. **`naive`'s directional split narrows once the count is matched.** `notes/17` §2 says
   the split is real and dose-dependent — it fails where 166 of 167 cells are unannotated
   — but part of the −0.0856 was the 51 % cap utilisation. *Falsified if* the gap between
   its two directions does not shrink.
3. **Recall per detection falls as the cap rises.** Detections are added weakest-first, so
   0.5x should give the best ratio and 1.5x the worst. *Falsified if* it is flat or rising,
   which would mean the detector is not ranking cells usefully at all.

Weights are written to `/kaggle/working` so a later notebook can take this kernel as a
`kernelDataSource` — `kaggleusercontent.com` is blocked from the agent container, so
downloading them is not an option (`notes/17` §4).
""")

code(r"""
import subprocess, sys, time

def sh(*args, **kw):
    # A missing binary raises FileNotFoundError rather than returning non-zero, which
    # would kill this cell outright on a CPU-only machine that has no nvidia-smi.
    try:
        return subprocess.run(args, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(args, 127, "", str(e))

def pip_install(pkgs, extra=()):
    r = sh(sys.executable, "-m", "pip", "install", "-q", *extra, *pkgs)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r.returncode == 0

# Which GPU did Kaggle give us? Ask nvidia-smi, NOT torch -- we may have to replace torch
# before importing it, and a module cannot be cleanly re-imported afterwards.
gpu = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").stdout.strip()
print(f"accelerator: {gpu or 'NONE'}")

# Kaggle's plain GPU option hands out a Tesla P100 (compute capability sm_60), and the
# image's torch 2.10+cu128 ships kernels for sm_70 and up only. CUDA then reports itself
# available and every launch dies with "no kernel image is available for execution on the
# device" -- which is what killed version 1 of this notebook, 250s in. machineShape is
# read-only on the kernels/push API, so T4 cannot be requested; replacing torch is the
# only lever we have. The cu121 builds carry sm_50 through sm_90.
if "P100" in gpu:
    print("P100 detected -> installing a torch with sm_60 kernels (a few minutes) ...")
    t0 = time.time()
    ok = pip_install(["torch==2.5.1"],
                     extra=("--index-url", "https://download.pytorch.org/whl/cu121"))
    print(f"  torch replacement {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")

print("installing geff + zarr ...")
pip_install(["geff", "zarr"])
""")

code(r"""
import sys, os, time, json, hashlib
from pathlib import Path
import numpy as np
import torch

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device {DEV}"
      + (f"  {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))
if DEV.type != "cuda":
    print("!! no GPU — this will be slow. Settings -> Accelerator -> GPU.")
else:
    cc = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    print(f"  compute capability sm_{cc[0]}{cc[1]}   torch built for {arch}")
    # Fail in SECONDS, not after four minutes of data building. torch.cuda.is_available()
    # is true on a P100 whose kernels do not exist; only an actual launch proves it.
    try:
        _a = torch.randn(8, 1, 8, 8, 8, device=DEV)
        _w = torch.nn.Conv3d(1, 4, 3, padding=1).to(DEV)
        _ = _w(_a).sum().item()
        torch.cuda.synchronize()
        print("  GPU smoke test passed (conv3d + sync)")
    except Exception as e:
        raise SystemExit(
            f"GPU present but unusable: {type(e).__name__}: {str(e)[:200]}\n"
            f"sm_{cc[0]}{cc[1]} is not in this torch's arch list {arch}. "
            "The pip cell should have replaced torch; check its output above.")

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
    raise SystemExit("Could not find harness/ and pipeline/. Add the project dataset as an input.")
sys.path.insert(0, str(REPO))

import pipeline.classical as pc
for need in ("dog_response", "open_movie", "load_frame"):
    if not hasattr(pc, need):
        raise SystemExit(f"The uploaded snapshot lacks pipeline.classical.{need} — re-upload the repo.")
from pipeline.classical import Config, dog_response, load_frame, open_movie, detect_frame_dog
from pipeline.detector import (TargetConfig, make_loss_mask, make_target,
                               peaks_from_prob, recall_at_budget)
from pipeline.unet import LOSSES, UNet3D, count_params, predict_volume
from harness.tracks import read_geff, read_estimated_nodes
print("detector modules imported")

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
N_TRAIN_DS, N_EVAL_DS = 24, 8      # datasets per embryo
N_TRAIN_FR, N_EVAL_FR = 10, 8      # frames per dataset
def stable_key(n): return int(hashlib.sha1(n.encode()).hexdigest(), 16)
""")

md("""## 1. Build the tensors

One frame is one network input: at `downsample=(1,4,4)` the volume is 64x64x64, isotropic
at 1.625 um, so there is no patching and no tiling seam. Everything a loss variant might
need — normalised volume, DoG response, hard target, mask, prior — is precomputed here so
the three arms train on byte-identical inputs.
""")

code(r"""
TCFG = TargetConfig(loss="masked")

def build_split(names_subset, n_frames, seed=0):
    vols, tgts, masks, priors, meta = [], [], [], [], []
    rng = np.random.default_rng(seed)
    for name in names_subset:
        zpath = TRAIN / f"{name}.zarr"
        arr, _attrs, scale, voxel_um, q_lo, q_hi = open_movie(zpath, CFG)
        T = int(arr.shape[0])
        gt = read_geff(TRAIN / f"{name}.geff")
        est = read_estimated_nodes(TRAIN / f"{name}.geff")
        # P(annotated | cell). Measured, not tuned -- this is what makes the PU arm honest.
        prior = float(np.clip(len(gt.t) / max(est, 1.0), 1e-4, TCFG.pu_clip)) if est else 0.05

        ts = sorted(rng.choice(T, size=min(n_frames, T), replace=False).tolist())
        gt_ds = gt.zyx / np.array(DS, float)      # GT is in FULL-res voxels; we work downsampled
        for t in ts:
            vol = load_frame(arr, t, CFG, q_lo, q_hi)
            norm, dog = dog_response(vol, voxel_um, CFG)
            centres = gt_ds[gt.t == t]
            tgt = make_target(centres, norm.shape, voxel_um, TCFG.pos_radius_um)
            mask = make_loss_mask(norm, dog, tgt, voxel_um, TCFG)
            vols.append(norm.astype(np.float32)); tgts.append(tgt); masks.append(mask)
            priors.append(prior)
            meta.append({"name": name, "t": t, "n_gt": int(len(centres)),
                         "voxel_um": voxel_um, "scale": scale})
    return (np.stack(vols), np.stack(tgts), np.stack(masks),
            np.array(priors, np.float32), meta)

SPLITS = {}
for emb, ns in sorted(by_embryo.items()):
    ordered = sorted(ns, key=stable_key)
    SPLITS[emb] = {"train": ordered[:N_TRAIN_DS], "eval": ordered[N_TRAIN_DS:N_TRAIN_DS + N_EVAL_DS]}
    print(f"{emb}: {len(SPLITS[emb]['train'])} train datasets, {len(SPLITS[emb]['eval'])} eval")

t0 = time.time()
DATA = {}
for emb in SPLITS:
    DATA[(emb, "train")] = build_split(SPLITS[emb]["train"], N_TRAIN_FR, seed=1)
    DATA[(emb, "eval")] = build_split(SPLITS[emb]["eval"], N_EVAL_FR, seed=2)
    for split in ("train", "eval"):
        v, tg, mk, pr, me = DATA[(emb, split)]
        print(f"  {emb}/{split}: {v.shape[0]} volumes {v.shape[1:]}  "
              f"pos {tg.mean():.4%}  mask keeps {mk.mean():.1%}  "
              f"prior min/med/max {pr.min():.4f}/{np.median(pr):.4f}/{pr.max():.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
""")

md("""## 2. The DoG baseline, at the detection count every arm must match

This is the number to beat, and it also *sets the budget*: each evaluation frame gets
capped at exactly the count DoG emitted there.
""")

code(r"""
def dog_eval(emb):
    v, tg, mk, pr, me = DATA[(emb, "eval")]
    caps, recalls, per_ds = [], [], {}
    for i, m in enumerate(me):
        gt = GT_CACHE[m["name"]]
        centres = gt.zyx[gt.t == m["t"]] / np.array(DS, float)
        coords, _ = detect_frame_dog(v[i], m["voxel_um"], CFG)
        caps.append(len(coords))
        if len(centres):
            r = recall_at_budget(np.zeros(len(coords)), coords, np.zeros(len(centres)),
                                 centres, m["voxel_um"])
            recalls.append(r); per_ds.setdefault(m["name"], []).append(r)
    return np.array(caps), float(np.mean(recalls)), per_ds

GT_CACHE = {}
for emb in SPLITS:
    for split in ("train", "eval"):
        for n in SPLITS[emb][split]:
            if n not in GT_CACHE:
                GT_CACHE[n] = read_geff(TRAIN / f"{n}.geff")

DOG = {}
for emb in SPLITS:
    caps, rec, _ = dog_eval(emb)
    DOG[emb] = {"caps": caps, "recall": rec}
    print(f"DoG on {emb}: recall {rec:.4f} at {caps.mean():.0f} detections/frame "
          f"(min {caps.min()}, max {caps.max()})")
""")

md("""## 3. Train the three variants, both directions

Train on one embryo, evaluate on the other. With two embryos that is the harshest split
available and the one that matches the hidden test, which is two embryos we have never
seen. Both directions are run because the two differ in density by more than 3x — a
variant that only wins training-sparse-testing-dense is a density artefact.
""")

code(r"""
EPOCHS, BATCH, LR = int(os.environ.get("BIOHUB_EPOCHS", 25)), 4, 2e-3

def train_one(loss_name, train_emb, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    v, tg, mk, pr, me = DATA[(train_emb, "train")]
    model = UNet3D(base=16, depth=3).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    fn = LOSSES[loss_name]
    n = len(v)
    t0, hist = time.time(), []
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(n)
        tot = 0.0
        for s in range(0, n, BATCH):
            b = order[s:s + BATCH]
            x = torch.as_tensor(v[b], device=DEV).unsqueeze(1)
            y = torch.as_tensor(tg[b], device=DEV).unsqueeze(1).float()
            m = torch.as_tensor(mk[b], device=DEV).unsqueeze(1).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                logits = model(x)
            loss = fn(logits.float(), y, m, prior=float(pr[b].mean()))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(b)
        hist.append(tot / n)
        if ep % 4 == 0 or ep == EPOCHS - 1:
            print(f"    {loss_name}/{train_emb} epoch {ep:>2}  loss {hist[-1]:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model, hist

CAP_MULTS = (0.5, 1.0, 1.5)

def eval_model(model, eval_emb):
    # Returns {mult: (recall, mean_dets)}. threshold=1e-6 so the CAP decides the count,
    # which is what phase 0 got wrong: at 0.05 the threshold bound first and the arms
    # spent 51-101% of budget while claiming to be matched.
    v, tg, mk, pr, me = DATA[(eval_emb, "eval")]
    caps = DOG[eval_emb]["caps"]
    acc = {mult: ([], []) for mult in CAP_MULTS}
    for i, m in enumerate(me):
        gt = GT_CACHE[m["name"]]
        centres = gt.zyx[gt.t == m["t"]] / np.array(DS, float)
        if not len(centres):
            continue
        prob = predict_volume(model, v[i], DEV)
        for mult in CAP_MULTS:
            cap = max(1, int(round(mult * caps[i])))
            coords, _ = peaks_from_prob(prob, m["voxel_um"], CFG.min_separation_um,
                                        threshold=1e-6, cap=cap)
            acc[mult][0].append(recall_at_budget(np.zeros(len(coords)), coords,
                                                 np.zeros(len(centres)), centres,
                                                 m["voxel_um"]))
            acc[mult][1].append(len(coords))
    return {mult: (float(np.mean(r)), float(np.mean(c))) for mult, (r, c) in acc.items()}

RESULTS = {}
embryos = sorted(SPLITS)
for train_emb in embryos:
    eval_emb = [e for e in embryos if e != train_emb][0]
    for loss_name in ("naive", "masked", "pu"):
        print(f"\n=== train on {train_emb} -> evaluate on {eval_emb}, loss={loss_name} ===")
        model, hist = train_one(loss_name, train_emb)
        curve = eval_model(model, eval_emb)
        base = DOG[eval_emb]["recall"]
        base_dets = float(DOG[eval_emb]["caps"].mean())
        rec, cnt = curve[1.0]
        RESULTS[(train_emb, loss_name)] = {
            "eval_emb": eval_emb, "recall": rec, "dog_recall": base,
            "delta": rec - base, "detections": cnt, "dog_detections": base_dets,
            "curve": {str(k): v for k, v in curve.items()}, "loss_hist": hist}
        for mult in CAP_MULTS:
            r, c = curve[mult]
            rpd = r / max(c, 1e-9)
            print(f"    cap {mult:.1f}x  recall {r:.4f}  dets {c:>6.1f}  "
                  f"rec/det {rpd:.5f}  ({rpd / (base / base_dets):.2f}x DoG)", flush=True)
        print(f"  AT MATCHED COUNT: recall {rec:.4f} vs DoG {base:.4f} "
              f"delta {rec-base:+.4f}  ({cnt:.0f} vs {base_dets:.0f} dets)", flush=True)
        torch.save({"state_dict": model.state_dict(), "loss": loss_name,
                    "train_emb": train_emb, "base": 16, "depth": 3},
                   WORK / f"claude_unet_{loss_name}_{train_emb}.pt")
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
""")

code(r"""
print("=" * 76)
print(f"{'train on':<10} {'eval on':<10} {'loss':<8} {'recall':>8} {'DoG':>8} {'delta':>9} {'dets':>7}")
for (train_emb, loss_name), r in sorted(RESULTS.items(), key=lambda kv: -kv[1]["delta"]):
    print(f"{train_emb:<10} {r['eval_emb']:<10} {loss_name:<8} {r['recall']:>8.4f} "
          f"{r['dog_recall']:>8.4f} {r['delta']:>+9.4f} {r['detections']:>7.0f}")

print("\n=== prediction 1: a learned detector beats DoG at matched detection count ===")
best = max(RESULTS.items(), key=lambda kv: kv[1]["delta"])
print(f"  best arm: {best[0][1]} trained on {best[0][0]} -> {best[1]['delta']:+.4f}")
print(f"  -> {'CONFIRMED' if best[1]['delta'] > 0 else 'FALSIFIED'}")

print("\n=== prediction 2: the naive loss is the worst of the three ===")
for train_emb in sorted(SPLITS):
    ds = {ln: RESULTS[(train_emb, ln)]['delta'] for ln in ('naive', 'masked', 'pu')}
    worst = min(ds, key=ds.get)
    print(f"  trained on {train_emb}: " + "  ".join(f"{k}={v:+.4f}" for k, v in ds.items())
          + f"   worst={worst}")
naive_worst = all(RESULTS[(e, 'naive')]['delta'] <= min(RESULTS[(e, l)]['delta']
                  for l in ('masked', 'pu')) for e in SPLITS)
print(f"  -> {'CONFIRMED' if naive_worst else 'FALSIFIED'}")

print("\n=== prediction 3: the winner wins in BOTH directions ===")
for loss_name in ("naive", "masked", "pu"):
    ds = [RESULTS[(e, loss_name)]["delta"] for e in sorted(SPLITS)]
    both = all(d > 0 for d in ds)
    print(f"  {loss_name:<8} " + "  ".join(f"{d:+.4f}" for d in ds)
          + f"   both positive: {both}")
winners = [l for l in ("naive", "masked", "pu")
           if all(RESULTS[(e, l)]["delta"] > 0 for e in SPLITS)]
print(f"  -> {'CONFIRMED: ' + str(winners) if winners else 'FALSIFIED: no variant wins both directions'}")

payload = {f"{te}|{ln}": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                          for k, v in r.items()}
           for (te, ln), r in RESULTS.items()}
payload["dog"] = {e: {"recall": DOG[e]["recall"], "mean_caps": float(DOG[e]["caps"].mean())}
                  for e in DOG}
payload["setup"] = {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                    "n_train_ds": N_TRAIN_DS, "n_train_frames": N_TRAIN_FR,
                    "params": count_params(UNet3D(base=16, depth=3))}
# kaggleusercontent is blocked from the agent container, so output FILES cannot be
# fetched -- only logs. Print the payload; do not rely on downloading it.
blob = json.dumps(payload, indent=2, default=str)
(WORK / "claude_detector_matched_results.json").write_text(blob)
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
