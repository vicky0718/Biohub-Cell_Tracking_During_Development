"""Build notebooks/claude_zhpilot.ipynb — a detector trained on DENSE labels.

`notes/44` closed the configuration direction: even using all 199 competition datasets the
smallest measurable effect is **0.0015**, and every config result sits at or below it. The
bar for anything worth running is now explicit — more than 0.0015 to be measurable, more
than 0.01 to be worth a submission slot.

`notes/43` found the one thing that clears it on paper:

    competition GT nodes, ALL 199 datasets :   133,318     (~6.7 per frame)
    kkunizaw/biohub-zh001r, ONE embryo     : 1,357,051     (~926 per frame)

Not more embryos — one — but **densely** annotated, and in exactly the geometry
`pipeline/unet.py` was built for: 64^3 isotropic, which at 1.625 um/voxel is the same
104 um field of view as the competition's `(64, 256, 256)` anisotropic volumes. Strided
`[:, ::4, ::4]` takes one to the other, which is the same downsample the pack itself uses.

## Why dense labels are the mechanism, not just more of the same

`notes/21` diagnosed why five from-scratch detectors failed, and it was **not** label
scarcity in the ordinary sense — it was **temporal incoherence**. A per-frame detector
finds different-but-plausible cells frame to frame, so edges between them fail to match
even where node recall is high, and it lands *below* the independence bound. A fixed DoG
filter sits above that bound for free, because the same filter answers the same structure
identically every time.

Sparse supervision is a plausible cause of exactly that degeneracy. With ~7 labelled cells
per frame out of ~900 present, almost any self-consistent subset satisfies the loss; the
model is never told which cells to commit to, so it never learns to commit to the same ones
twice. Dense supervision removes the freedom — every cell is labelled, so there is one
right answer per frame.

The strongest evidence that this is the real story is already in our own code:
`pipeline/unet.py` ships `masked_loss` and `pu_loss` — positive-unlabelled machinery built
solely to cope with sparse labels. On dense labels plain BCE is simply correct, and this
notebook uses `naive_loss`. If the diagnosis is right, that substitution is the whole point.

## What would make this fail

`notes/23` §2c: a learned detector's CV->LB offset is 0.061 worse than the classical
champion's, so a learned arm needs **CV ~0.813** to be expected to match the 0.752 floor.
Five prior runs topped out at **0.649**. And zh001r's labels are Ultrack-generated, not
hand-curated, so this is a domain shift as well as a density gain. `biohub-zmnscrops`, the
12.9 GB companion, returns 403, so one embryo is the whole external option.

Predictions 2 and 4 are the ones that close the direction if they fail, and both are
mechanism tests rather than score tests -- they ask whether dense labels fixed the thing
`notes/21` diagnosed, which is cheaper and more informative than asking whether the number
went up.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_zhpilot.ipynb")
N_EVAL = 12          # competition datasets for the transfer test
N_VAL_CLIPS = 12     # zh001r clips held out of training
EPOCHS = 16
CELLS = []
Q3 = chr(39) * 3


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    src = (src.replace("__N_EVAL__", str(N_EVAL))
              .replace("__N_VAL_CLIPS__", str(N_VAL_CLIPS))
              .replace("__EPOCHS__", str(EPOCHS)))
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# A detector trained on dense labels

```
0.901  submitted        0.926 bronze     0.944 gold
measurable  > 0.0015     worth a slot  > 0.01        (notes/44)
```

`notes/44` closed configuration: even with all 199 competition datasets the smallest
measurable effect is 0.0015, and everything that direction produced is at or below it.
`notes/43` found the only untested thing that clears the bar on paper.

```
competition GT nodes, ALL 199 datasets :   133,318     ~6.7 per frame
zh001r, ONE embryo                     : 1,357,051     ~926 per frame
```

## The mechanism, not just more data

`notes/21` diagnosed the failure of five from-scratch detectors as **temporal
incoherence** — a per-frame detector picks different-but-plausible cells each frame, so the
edges between them do not match, and it falls *below* the independence bound while a fixed
DoG filter sits above it for free.

With ~7 labelled cells per frame out of ~900, almost any self-consistent subset satisfies
the loss. The model is never told which cells to commit to. Dense labels remove that
freedom.

Our own code is the evidence: `pipeline/unet.py` carries `masked_loss` and `pu_loss`,
positive-unlabelled machinery that exists **only** to cope with sparse labels. On dense
labels plain BCE is correct, and this run uses `naive_loss`. That substitution is the point.

## Design

Train on zh001r (60 clips), hold out 12. Then evaluate on **competition** data, which is
the only thing that matters, against the DoG detector **run in the same notebook on the same
frames** — not against a number quoted from an older note. Three of this project's
measurement errors were a control compared against something it was not comparable to.

Competition volumes are `(100, 64, 256, 256)`; `[:, ::4, ::4]` gives the 64³ isotropic grid
zh001r is in, the same strided downsample the pack uses internally.

## Pre-registered predictions

Thresholds sit where the decision changes. `notes/42`: a threshold below the measurement's
resolution is a coin flip with a paper trail.

1. **It fits its own domain** — peak recall > 0.80 on held-out zh001r clips. If a 2M-param
   UNet cannot fit dense labels on the data it trained on, nothing below is interpretable.
2. **`paired_recall`'s position score beats the per-frame independence bound on competition
   data.** The mechanism test, and the exact quantity `notes/21` built for it. Failure here
   means dense labels did **not** fix temporal incoherence and the direction is closed.
3. **Node recall at budget on competition data > 0.90** — DoG-class detection.
4. **It beats DoG on `paired_recall`,** same datasets, same frames, same run. The
   comparison that decides whether this can replace anything.
5. **Transfer holds** — competition recall within 0.15 of held-out zh001r recall. A larger
   collapse means the Ultrack domain shift dominates the density gain, which is the honest
   reading if it happens.
""")

code(r"""
import os, subprocess, sys, time, json
from pathlib import Path

T_START = time.time()
WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)

def sh(*a, **kw):
    try:
        return subprocess.run(a, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(a, 127, "", str(e))

def pip_install(pkgs, extra=()):
    r = sh(sys.executable, "-m", "pip", "install", "-q", *extra, *pkgs)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r.returncode == 0

print(sh("nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader").stdout.strip()
      or "no GPU")

def find_dir(is_match, roots, max_depth=6):
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

PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir()
                and (p / "wheels").is_dir() and "seed314159" not in str(p),
                ["/kaggle/input"])
REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
ZH = find_dir(lambda p: any(p.glob("zh001r_iso.npy")), ["/kaggle/input"], max_depth=4)
TORCH_WH = find_dir(
    lambda p: p.name == "wheels" and any(x.name.startswith("torch-") for x in p.iterdir()),
    ["/kaggle/input"])
for label, val in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
                   ("zebrahub", ZH), ("torch wheels", TORCH_WH)):
    print(f"  {label:<14} {val}")
missing = [l for l, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
                          ("zebrahub", ZH)) if v is None]
if missing:
    raise SystemExit(f"not mounted: {missing}")
TRAIN = COMP / "train"

t0 = time.time()
ok1 = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                  extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok1 else 'FAILED'} ({time.time()-t0:.0f}s)")
if TORCH_WH is None:
    print("!! no torch wheelhouse — the P100 is sm_60 and the image torch ships sm_70+ "
          "kernels, so CUDA will fail at the first conv.")
else:
    t0 = time.time()
    ok2 = pip_install(["torch==2.5.1"], extra=("--no-index", f"--find-links={TORCH_WH}"))
    print(f"torch wheels {'ok' if ok2 else 'FAILED'} ({time.time()-t0:.0f}s)")

probe = sh(sys.executable, "-c",
           "import numpy, torch, zarr, scipy; ok=False\n"
           "if torch.cuda.is_available():\n"
           "    try:\n"
           "        w=torch.nn.Conv3d(1,4,3,padding=1).cuda()\n"
           "        _=w(torch.randn(2,1,8,8,8,device='cuda')).sum().item()\n"
           "        torch.cuda.synchronize(); ok=True\n"
           "    except Exception as e: print('GPU BROKEN:', type(e).__name__, str(e)[:120])\n"
           "print('numpy', numpy.__version__, '| torch', torch.__version__, '| gpu_ok', ok)")
print(probe.stdout.strip() or probe.stderr.strip()[-1500:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import in a fresh interpreter")
if "gpu_ok True" not in probe.stdout:
    raise SystemExit("GPU unusable — training on CPU will not finish in the budget")
""")

md("""
## 1. Train on zh001r, then evaluate on competition data

Everything runs in a subprocess. The pack wheels replace numpy under this kernel, so
`scipy` here no longer matches it — `claude_divdata` v1 died on exactly that — and
`pipeline/detector.py` needs scipy. A fresh interpreter picks up the installed numpy.
""")

WORKER_BODY = r'''
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

T0 = time.time()
sys.path.insert(0, "{repo}")
PACK = Path("{pack}"); TRAIN = Path("{train}"); WORK = Path("{work}"); ZH = Path("{zh}")

import zarr
from pipeline.unet import UNet3D, naive_loss, predict_volume, count_params
from pipeline.detector import peaks_from_prob, recall_at_budget, paired_recall
from pipeline.classical import Config, detect_frame_dog
from harness.tracks import read_geff, read_scale

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("worker numpy " + np.__version__ + " torch " + torch.__version__ + " on " + str(DEV),
      flush=True)

N_EVAL = __N_EVAL__
N_VAL_CLIPS = __N_VAL_CLIPS__
EPOCHS = __EPOCHS__
ISO_UM = (1.625, 1.625, 1.625)          # the isotropic grid zh001r lives in
COMP_UM = (1.625, 0.40625, 0.40625)     # competition voxel scale
DS = (1, 4, 4)                          # strided downsample to reach the isotropic grid

# ---------------------------------------------------------------- zh001r
vol = np.load(ZH / "zh001r_iso.npy", mmap_mode="r")
tgt = np.load(ZH / "zh001r_tgt.npy", mmap_mode="r")
nodes = np.load(ZH / "zh001r_nodes.npz")
print("volume " + str(vol.shape) + "  target " + str(tgt.shape)
      + "  node arrays " + str(len(nodes.files)), flush=True)
if vol.shape != tgt.shape:
    raise SystemExit("volume and target shapes disagree: " + str(vol.shape)
                     + " vs " + str(tgt.shape))
N_CLIP, N_FR = vol.shape[0], vol.shape[1]

rng = np.random.default_rng(0)
perm = rng.permutation(N_CLIP)
val_clips = sorted(int(c) for c in perm[:N_VAL_CLIPS])
train_clips = sorted(int(c) for c in perm[N_VAL_CLIPS:])
print("train clips " + str(len(train_clips)) + "  val clips " + str(len(val_clips)),
      flush=True)

def clip_frame_nodes(clip, frame):
    # keys are f0..f1439 in clip-major order; each row is (t_in_clip, z, y, x)
    a = nodes["f" + str(clip * N_FR + frame)]
    return np.asarray(a[:, 1:], dtype=float)

# ---------------------------------------------------------------- model
model = UNet3D(base=16, depth=3, in_ch=1).to(DEV)
print("params " + str(count_params(model)), flush=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
BATCH = 8
pairs = [(c, f) for c in train_clips for f in range(N_FR)]
print("train samples " + str(len(pairs)), flush=True)

def batch_of(idx):
    xs = np.stack([np.asarray(vol[c, f], dtype=np.float32) / 255.0 for c, f in idx])
    ys = np.stack([np.asarray(tgt[c, f], dtype=np.float32) / 255.0 for c, f in idx])
    x = torch.from_numpy(xs).unsqueeze(1).to(DEV)
    y = torch.from_numpy(ys).unsqueeze(1).to(DEV)
    return x, y

def augment(x, y):
    # v1's val loss bottomed at epoch 1 and rose for six more. That is a diversity
    # problem, not a label problem: zh001r is 1,440 volumes from ONE embryo against the
    # competition's 19,900 from two, and dense labelling does not supply diversity.
    # Flips and yx-rotations are EXACT symmetries of an isotropic cubic grid, so they add
    # views without inventing structure. Image and target get the identical transform.
    dims = list()
    for d in (2, 3, 4):
        if int(rng.integers(0, 2)):
            dims.append(d)
    if dims:
        x = torch.flip(x, dims=dims); y = torch.flip(y, dims=dims)
    k = int(rng.integers(0, 4))
    if k:
        x = torch.rot90(x, k, dims=(3, 4)); y = torch.rot90(y, k, dims=(3, 4))
    return x, y

HIST = list()
BEST = dict(val=float("inf"), epoch=-1)
CKPT = WORK / "zh_detector.pth"
for ep in range(EPOCHS):
    model.train()
    order = rng.permutation(len(pairs))
    tot, nb = 0.0, 0
    for i in range(0, len(order) - BATCH + 1, BATCH):
        idx = [pairs[j] for j in order[i:i + BATCH]]
        x, y = augment(*batch_of(idx))
        # naive BCE: dense labels make the positive-unlabelled machinery unnecessary,
        # and using it here would reintroduce the very assumption this run is testing.
        loss = naive_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.item()); nb += 1
    model.eval()
    vtot, vnb = 0.0, 0
    with torch.no_grad():
        # No augmentation on validation -- the val number has to mean the same thing
        # every epoch or best-on-val selects on noise.
        for c in val_clips[:4]:
            for f in range(0, N_FR, 5):
                x, y = batch_of([(c, f)])
                vtot += float(naive_loss(model(x), y).item()); vnb += 1
    v = vtot / max(vnb, 1)
    HIST.append(dict(epoch=ep, train=tot / max(nb, 1), val=v))
    if v < BEST["val"]:
        BEST = dict(val=v, epoch=ep)
        torch.save(model.state_dict(), CKPT)
    print("epoch " + str(ep) + "  train " + format(tot / max(nb, 1), ".5f")
          + "  val " + format(v, ".5f") + "  " + str(int(time.time() - T0)) + "s"
          + ("  *best" if BEST["epoch"] == ep else ""), flush=True)

# Evaluate the BEST checkpoint, not the last. v1 would have scored its epoch-7 model,
# which its own val curve showed was the worst of the eight.
if BEST["epoch"] < 0:
    raise SystemExit("no checkpoint was ever better than inf; training did not run")
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.eval()
print("evaluating epoch " + str(BEST["epoch"]) + " (val "
      + format(BEST["val"], ".5f") + ")", flush=True)

# ---------------------------------------------------------------- held-out zh001r
def peaks_of(prob, sep_um=4.0, cap=800):
    # peaks_from_prob returns (coords, scores). v1 indexed the tuple and died after 370 s
    # of training, so the unpack lives here and no call site can repeat it. The cap is
    # what the docstring says does the real work of bounding a node budget.
    coords, _scores = peaks_from_prob(prob, ISO_UM, sep_um, cap=cap)
    return coords

zh_rec = list()
for c in val_clips:
    hit, tot_n = 0, 0
    for f in range(0, N_FR, 4):
        prob = predict_volume(model, np.asarray(vol[c, f], dtype=np.float32) / 255.0, DEV)
        pk = peaks_of(prob)
        gt = clip_frame_nodes(c, f)
        tot_n += len(gt)
        if len(pk) and len(gt):
            d = np.linalg.norm(
                (pk[:, None, :] - gt[None, :, :]) * np.asarray(ISO_UM), axis=2)
            hit += int((d.min(axis=0) <= 7.0).sum())
    zh_rec.append(hit / max(tot_n, 1))
ZH_RECALL = float(np.mean(zh_rec))
print("held-out zh001r recall " + format(ZH_RECALL, ".4f"), flush=True)

# ---------------------------------------------------------------- competition transfer
names = sorted(p.stem for p in TRAIN.glob("*.zarr")
               if (TRAIN / (p.stem + ".geff")).exists())
a = [n for n in names if n.startswith("44b6")]
b = [n for n in names if not n.startswith("44b6")]
k = max(1, round(N_EVAL * len(a) / max(len(a) + len(b), 1)))
names = a[:k] + b[:N_EVAL - k]
print(str(len(names)) + " competition datasets: " + str(sum(1 for n in names
      if n.startswith("44b6"))) + " x 44b6", flush=True)

cfg = Config()
ROWS = list()
for name in names:
    t0 = time.time()
    arr = zarr.open_group(str(TRAIN / (name + ".zarr")), mode="r")["0"]
    gt = read_geff(TRAIN / (name + ".geff"))
    sc = read_scale(TRAIN / (name + ".zarr"))
    T = arr.shape[0]
    lt, lz = list(), list()
    dt, dz = list(), list()
    for t in range(T):
        raw = np.asarray(arr[t, ::DS[0], ::DS[1], ::DS[2]], dtype=np.float32)
        lo, hi = np.percentile(raw, 1.0), np.percentile(raw, 99.9)
        nrm = np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        prob = predict_volume(model, nrm, DEV)
        pk = peaks_of(prob)
        if len(pk):
            # back to competition voxel indices: y,x were strided by 4
            full = pk * np.asarray([DS[0], DS[1], DS[2]], dtype=float)
            lt.append(np.full(len(full), t, np.int64)); lz.append(full)
        # DoG on the SAME frame, same run -- three of this project's measurement errors
        # were a control compared against something it was not comparable to.
        dpk = detect_frame_dog(raw, ISO_UM, cfg)
        if len(dpk):
            dfull = np.asarray(dpk, dtype=float) * np.asarray(
                [DS[0], DS[1], DS[2]], dtype=float)
            dt.append(np.full(len(dfull), t, np.int64)); dz.append(dfull)
    lt = np.concatenate(lt) if lt else np.zeros(0, np.int64)
    lz = np.concatenate(lz) if len(lz) else np.zeros((0, 3), float)
    dt = np.concatenate(dt) if dt else np.zeros(0, np.int64)
    dz = np.concatenate(dz) if len(dz) else np.zeros((0, 3), float)

    row = dict(name=name, n_learned=int(len(lt)), n_dog=int(len(dt)),
               n_gt=int(len(gt.t)))
    for tag, pt, pz in (("learned", lt, lz), ("dog", dt, dz)):
        try:
            row[tag + "_recall"] = float(recall_at_budget(pt, pz, gt.t, gt.zyx, COMP_UM))
        except Exception as e:
            row[tag + "_recall"] = float("nan")
            print("  recall failed for " + tag + ": " + type(e).__name__ + " "
                  + str(e)[:120], flush=True)
        try:
            pr = paired_recall(pt, pz, gt.t, gt.zyx, gt.edges, COMP_UM)
            row[tag + "_paired"] = dict(
                (str(kk), float(vv)) for kk, vv in dict(pr).items()
            ) if not isinstance(pr, (int, float)) else dict(value=float(pr))
        except Exception as e:
            row[tag + "_paired"] = dict(error=type(e).__name__ + " " + str(e)[:160])
    ROWS.append(row)
    print("  " + name + "  learned n=" + str(row["n_learned"])
          + " rec=" + format(row.get("learned_recall", float("nan")), ".4f")
          + " | dog n=" + str(row["n_dog"])
          + " rec=" + format(row.get("dog_recall", float("nan")), ".4f")
          + "  " + str(int(time.time() - t0)) + "s", flush=True)
    (WORK / "zhpilot.json").write_text(json.dumps(
        dict(history=HIST, best=BEST, zh_recall=ZH_RECALL, zh_per_clip=zh_rec,
             rows=ROWS, datasets=[r["name"] for r in ROWS]), default=float))

print("worker done in " + str(int(time.time() - T0)) + " s", flush=True)
'''

code('''
import subprocess, sys, time
WORKER = WORK / "run_zhpilot.py"
WORKER.write_text(""" + BODY + """.format(
    pack=str(PACK), repo=str(REPO), train=str(TRAIN), zh=str(ZH), work=str(WORK)))

t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", str(WORKER)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print("worker exited", rc, "after", int(time.time() - t0), "s")
if rc != 0:
    raise SystemExit("worker failed (" + str(rc) + ")")
'''.replace('""" + BODY + """', Q3 + WORKER_BODY + Q3))

md("""## 2. The five predictions""")

code(r"""
import json, numpy as np
D = json.loads((WORK / "zhpilot.json").read_text())
H, ROWS = D["history"], D["rows"]
ZH_RECALL = D["zh_recall"]

B = D.get("best") or {}
print("training  (evaluated at epoch "
      + str(B.get("epoch", "?")) + ", the val minimum)")
for h in H:
    star = "  *" if h["epoch"] == B.get("epoch") else ""
    print(f"  epoch {h['epoch']:>2}  train {h['train']:.5f}  val {h['val']:.5f}{star}")
vals = [h["val"] for h in H]
if vals and B.get("epoch", 0) < len(vals) - 1:
    print(f"  val rose {vals[-1] - min(vals):+.5f} after epoch {B.get('epoch')} — "
          "1,440 volumes from one embryo is the binding constraint, not labels")

def col(k):
    v = [r[k] for r in ROWS if k in r and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")

LR, DR = col("learned_recall"), col("dog_recall")
print(f"\nheld-out zh001r recall      {ZH_RECALL:.4f}")
print(f"competition recall  learned {LR:.4f}   dog {DR:.4f}")
print(f"nodes per dataset   learned {col('n_learned'):,.0f}   dog {col('n_dog'):,.0f}"
      f"   gt {col('n_gt'):,.0f}")

# paired_recall's shape is not assumed -- print whatever it returned, then look for the
# position/coherence field by name rather than by position in a tuple.
pr = next((r.get("learned_paired") for r in ROWS if isinstance(r.get("learned_paired"), dict)
           and "error" not in r["learned_paired"]), None)
print(f"\npaired_recall fields: {sorted(pr) if pr else 'UNAVAILABLE'}")
err = next((r["learned_paired"].get("error") for r in ROWS
            if isinstance(r.get("learned_paired"), dict) and "error" in r["learned_paired"]),
           None)
if err:
    print(f"  paired_recall errored: {err}")

def pfield(tag, *cands):
    out = []
    for r in ROWS:
        d = r.get(tag + "_paired")
        if isinstance(d, dict) and "error" not in d:
            for c in cands:
                if c in d and d[c] == d[c]:
                    out.append(float(d[c])); break
    return float(np.mean(out)) if out else float("nan")

LP = pfield("learned", "position", "paired", "value", "score")
DP = pfield("dog", "position", "paired", "value", "score")
IND = pfield("learned", "independent", "independence", "baseline")
print(f"paired position   learned {LP:.4f}   dog {DP:.4f}   independence bound {IND:.4f}")

print("\n" + "=" * 84)
print("PREDICTION GRADING")
print("=" * 84)

print("\n1. peak recall > 0.80 on held-out zh001r clips")
ok1 = ZH_RECALL > 0.80
print(f"   {ZH_RECALL:.4f}  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   The model does not fit dense labels on its own domain. Nothing below is")
    print("   interpretable, and the problem is the training setup, not the data.")

print("\n2. paired position beats the per-frame independence bound on competition data")
if LP != LP or IND != IND:
    print(f"   NOT GRADED — learned {LP:.4f}, bound {IND:.4f}; paired_recall did not")
    print("   return the fields this grading assumed. Its output is printed above.")
else:
    ok2 = LP > IND
    print(f"   {LP:.4f} vs {IND:.4f}  ->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   Dense labels did NOT fix temporal incoherence. That was the whole")
        print("   mechanism, and the direction closes here.")

print("\n3. competition node recall at budget > 0.90")
ok3 = LR == LR and LR > 0.90
print(f"   {LR:.4f}  ->  {'PASS' if ok3 else 'FAIL'}")

print("\n4. the learned detector beats DoG on paired position")
if LP != LP or DP != DP:
    print("   NOT GRADED — one side missing")
else:
    ok4 = LP > DP
    print(f"   learned {LP:.4f} vs dog {DP:.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   DoG still wins, as it has in every prior attempt. Dense labels closed")
        print("   some of the gap or none of it, but not enough to replace anything.")

print("\n5. transfer holds — competition recall within 0.15 of held-out zh001r")
gap = ZH_RECALL - LR if LR == LR else float("nan")
ok5 = gap == gap and abs(gap) < 0.15
print(f"   zh001r {ZH_RECALL:.4f} - competition {LR:.4f} = {gap:+.4f}"
      f"  ->  {'PASS' if ok5 else 'FAIL'}")
if not ok5:
    print("   The Ultrack domain shift dominates the density gain. One embryo of")
    print("   algorithm-generated labels is not the same distribution as the target.")

print("\n" + "=" * 84)
print(f"zh001r {ZH_RECALL:.4f} | competition learned {LR:.4f} dog {DR:.4f} "
      f"| paired learned {LP:.4f} dog {DP:.4f}")
print("=" * 84)
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
