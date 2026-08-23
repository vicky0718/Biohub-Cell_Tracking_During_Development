"""Build notebooks/claude_pack_train.ipynb — train THEIR architecture on OUR splits.

The pack's `train_unet_transformer.py` takes `--splits`, a path to a JSON file naming each
fold's train/test datasets. Their own splits file was never published, which is why
`notes/24` §2 concluded no honest CV of their weights is obtainable. Supplying our own
leave-one-embryo-out split fixes that completely: the held-out embryo is one we choose, so
the resulting number is a real generalisation estimate rather than a training-set score.

This also sidesteps the external-model question entirely -- the weights would be ours,
trained on competition data, using CC0 source.

Run 1 measures COST, not quality. Their published checkpoint is at epoch 402; what matters
first is seconds per epoch, because that decides whether 402 is reachable at all.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_pack_train.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Train their architecture on our splits

`notes/24` §2: their published weights cannot be honestly cross-validated, because the
`dataset_splits.json` naming what `split_0` trained on was never shipped, is absent from the
competition data, and is only *read* by their training script rather than generated. Every
one of the 199 training datasets is therefore potentially contaminated.

Training the same architecture ourselves removes that problem entirely. We choose the
split, so the held-out embryo is genuinely held out.

It also removes the external-model question: these weights would be **ours**, trained on
competition data, using CC0-licensed source.

## What we already know about why our own detector fell short

`notes/23` §6 compared our attempt with theirs:

| | ours | theirs |
|---|---|---|
| parameters | 350,809 | **2,077,996** |
| epochs | 15 | **402** |
| temporal | frames as input channels | **per-voxel attention across time** |
| linking | hand-written Hungarian | **learned**, joint `det_loss` + `edge_loss` |
| `downsample` | (1, 4, 4) | (1, 4, 4) — identical |

All four differences are addressed by using their trainer. The geometry already matched,
so our data pipeline was never the problem.

## This run measures cost, not quality

Their checkpoint is at **epoch 402**. Whether that is reachable here is the question that
decides if this path is viable at all, and it is answered by seconds-per-epoch — not by a
score. So run 1 trains a **small number of epochs on a subset** and reports the projection.

Committing to a long run before knowing the per-epoch cost is how a 12 h ceiling gets
discovered at hour eleven.
""")

code(r"""
import subprocess, sys, time, os, json
from pathlib import Path

def sh(*a, **kw):
    try:
        return subprocess.run(a, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(a, 127, "", str(e))

def pip_install(pkgs, extra=()):
    r = sh(sys.executable, "-m", "pip", "install", "-q", *extra, *pkgs)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
    return r.returncode == 0

gpu = sh("nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader").stdout.strip()
print(f"accelerator: {gpu or 'NONE'}")
if "P100" in gpu:
    print("P100 (sm_60) -> replacing torch; the image build ships sm_70+ only")
    t0 = time.time()
    ok = pip_install(["torch==2.5.1"],
                     extra=("--index-url", "https://download.pytorch.org/whl/cu121"))
    print(f"  torch replacement {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")

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

PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir(),
                ["/kaggle/input"])
if PACK is None:
    raise SystemExit("Support pack not mounted (need repo/ and weights/).")
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Competition data not mounted.")
TRAIN = COMP / "train"
print(f"pack: {PACK}\ndata: {TRAIN}")

# Their wheel set, installed wholesale. PyPI resolution produced a numpy split
# (notes/24 §1); the pack's set is coherent because its author ran it.
WHEELS = PACK / "wheels"
t0 = time.time()
ok = pip_install([str(p) for p in sorted(WHEELS.glob("*.whl"))],
                 extra=("--no-index", f"--find-links={WHEELS}"))
print(f"pack wheels {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")

probe = sh(sys.executable, "-c",
           "import numpy, torch, zarr, tracksdata; "
           "print('numpy', numpy.__version__, '| torch', torch.__version__, "
           "'| cuda', torch.cuda.is_available())")
print(probe.stdout.strip() or probe.stderr.strip()[-800:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import cleanly in a fresh interpreter")
""")

md("""## 1. Our split, written in their format

Leave-one-embryo-out, which is what every CV number in this project has used. Fold 0 trains
on `6bba` and tests on `44b6`; fold 1 reverses it. Their loader reads `folds[i]["test"]`
(seen in `predict()`); `"train"` and `"val"` are written too, and a missing key will raise
loudly rather than silently training on the wrong set.
""")

code(r"""
names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})
by_emb = {}
for n in names:
    by_emb.setdefault(n.split("_")[0], []).append(n)
print("embryos:", {k: len(v) for k, v in sorted(by_emb.items())})
embryos = sorted(by_emb)
if len(embryos) < 2:
    raise SystemExit(f"need >=2 embryos for leave-one-out; found {embryos}")

# Run 1 is a COST measurement, so it trains on a subset. Set to None for the real run.
N_TRAIN_SUBSET = 20
N_TEST_SUBSET = 4
EPOCHS = 2
WINDOW_SIZE = 2                      # their config.json
UNET_LAYERS = "32,64,128"            # their config.json
BATCH_SIZE = 4                       # theirs defaults to 16; P100 has 16 GB

folds = []
for held in embryos:
    tr = [n for e, v in by_emb.items() if e != held for n in sorted(v)]
    te = sorted(by_emb[held])
    if N_TRAIN_SUBSET:
        tr = tr[:N_TRAIN_SUBSET]
    if N_TEST_SUBSET:
        te = te[:N_TEST_SUBSET]
    folds.append({"train": tr, "val": te, "test": te, "held_out_embryo": held})
    print(f"fold {len(folds)-1}: train {len(tr)} (not {held}) -> test {len(te)} ({held})")

SPLITS = Path("/kaggle/working/our_splits.json")
SPLITS.write_text(json.dumps(folds, indent=1))
print(f"\nwrote {SPLITS}")
# The contamination that made their weights unusable cannot recur here: assert it.
for i, f in enumerate(folds):
    overlap = set(f["train"]) & set(f["test"])
    assert not overlap, f"fold {i} leaks {len(overlap)} datasets between train and test"
    embs_tr = {n.split("_")[0] for n in f["train"]}
    assert f["held_out_embryo"] not in embs_tr, f"fold {i} trains on the held-out embryo"
print("verified: no dataset and no embryo appears on both sides of any fold")
""")

md("""## 2. Train, and measure what an epoch costs""")

code(r"""
# Their dataspec sets WEIGHTS_PATH = <repo>/weights, computed from __file__ -- and the
# pack is a READ-ONLY mount, so the trainer dies at
#   OSError: [Errno 30] Read-only file system: '.../repo/weights'
# before the first epoch. Copying the repo somewhere writable fixes every derived path at
# once (weights/, predictions/, results/) rather than shimming them one at a time.
import shutil
RUNREPO = Path("/kaggle/working/repo")
if not RUNREPO.exists():
    shutil.copytree(PACK / "repo", RUNREPO)
    print(f"copied repo -> {RUNREPO}")
for sub in ("weights", "predictions", "results"):
    (RUNREPO / sub).mkdir(parents=True, exist_ok=True)

TRAINER = RUNREPO / "scripts" / "train_unet_transformer.py"
if not TRAINER.exists():
    raise SystemExit(f"trainer not found at {TRAINER}")

env = dict(os.environ)
env["PYTHONPATH"] = f"{RUNREPO/'src'}:{RUNREPO/'scripts'}"
env["BIOHUB_DATA_DIR"] = str(TRAIN)
env["USER"] = "claude"

cmd = [sys.executable, "-u", str(TRAINER),
       "--data-dir", str(TRAIN),
       "--splits", str(SPLITS),
       "--split", "0",
       "--epochs", str(EPOCHS),
       "--batch-size", str(BATCH_SIZE),
       "--window-size", str(WINDOW_SIZE),
       "--unet-layers", UNET_LAYERS,
       "--single-gpu"]
print("running:", " ".join(cmd), "\n", flush=True)

t0 = time.time()
proc = subprocess.Popen(cmd, cwd=str(RUNREPO / "scripts"), env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
lines = []
for line in proc.stdout:
    line = line.rstrip()
    lines.append(line)
    print(line, flush=True)
rc = proc.wait()
elapsed = time.time() - t0
print(f"\ntrainer exited {rc} after {elapsed:.0f}s")
""")

code(r"""
print(f"{EPOCHS} epoch(s) on {N_TRAIN_SUBSET or 'all'} datasets took {elapsed:.0f}s")
if rc == 0 and EPOCHS:
    per_epoch = elapsed / EPOCHS
    print(f"  ~{per_epoch:.0f}s per epoch at this subset size")
    full = (len(names) - len(by_emb[embryos[0]])) if N_TRAIN_SUBSET else N_TRAIN_SUBSET
    scale = full / max(1, N_TRAIN_SUBSET) if N_TRAIN_SUBSET else 1.0
    print(f"  scaling to the full {full}-dataset fold: ~{per_epoch*scale:.0f}s/epoch")
    for ep in (50, 100, 402):
        h = per_epoch * scale * ep / 3600
        flag = "" if h <= 11 else "   <- exceeds a single 12 h session"
        print(f"    {ep:>3} epochs: {h:>6.1f} h{flag}")
    print("\nTheir published checkpoint is at epoch 402. If that does not fit, the "
          "options are fewer epochs, a smaller model, or chained sessions resuming "
          "from a checkpoint — decided on these numbers, not guessed.")
else:
    print("  trainer did not complete; read its output above before projecting anything")

Path("/kaggle/working/claude_pack_train.json").write_text(json.dumps({
    "rc": rc, "elapsed_s": elapsed, "epochs": EPOCHS,
    "n_train": N_TRAIN_SUBSET, "n_test": N_TEST_SUBSET,
    "batch_size": BATCH_SIZE, "window_size": WINDOW_SIZE,
    "unet_layers": UNET_LAYERS, "folds": folds,
    "tail": lines[-60:],
}, indent=2))
print("\nwrote claude_pack_train.json")
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
