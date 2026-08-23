"""Build notebooks/claude_cluster_probe.ipynb — read the public support pack.

`notes/15` §3 established what the 0.913-0.916 cluster is: inference over weights published
as a public Kaggle Dataset, not a trained-in-notebook pipeline. The source notebook it was
read from (scriptVersionId 333208869) now 404s, and `/datasets/list/files/...` 404s too, so
the reliable way to learn the pack's shape is to mount it and look.

This is a READ-ONLY probe. It trains nothing, submits nothing, and writes only a JSON
description of what it found. It exists so the next notebook can be written against the
real file layout and the real state_dict rather than against a remembered config block.

Rules position (checked 2026-08-23): Competition Rules §6.b -- "The use of external data
and models is acceptable unless specifically prohibited by the Host" -- and §6.a requires
external data be "publicly available and equally accessible to use by all Participants ...
at no cost". The pack is a public Kaggle Dataset under CC0: Public Domain, already in use
by ~514 teams. The one open item is the Competition-Specific Rules section, which the owner
is confirming; if it restricts external models this notebook and everything downstream of
it are void.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_cluster_probe.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Probe the public support pack

**Read-only.** Trains nothing, submits nothing, changes nothing. It mounts
`pilkwang/biohub-tracking-support-pack-50ep-v1` and reports what is actually in it, so the
reproduction notebook can be written against the real artefact rather than a remembered
config block.

## Why this is needed

`notes/15` §3 read the cluster's source and recorded its config:

```
METHOD           = "unet_transformer"
WEIGHTS_RELATIVE = "weights/unet_transformer/split_0/edge_predictor_best.pth"
DET_THRESHOLD    = 0.985      # UNet probability, not intensity
USE_ILP          = 1
ILP_DIVISION_WEIGHT = 1.0
```

That source now returns 404, and the dataset file-listing endpoint does too. Rather than
build on a five-day-old transcription, this reads the pack directly and prints:

- the directory tree (bounded, so a deep tree cannot cost minutes)
- every `.pth` / `.pt` checkpoint, with its `state_dict` **keys and tensor shapes** — which
  reconstructs the architecture without needing their code
- any config, JSON, YAML or Python file, printed in full if small
- the wheels directory, if the offline-install pattern is still there

## What it does not do

It does not run inference and it does not score anything. Loading a 355 MB pack and
guessing at an inference path in the same notebook would confound "can we read it" with
"can we run it", and the first question has to be answered first.
""")

code(r"""
import os, sys, json, time
from pathlib import Path

t0 = time.time()
ROOT = Path("/kaggle/input")
PACK_HINTS = ("support-pack", "support_pack", "tracking-support")

# Bounded recursive search, not a fixed depth: this account's kernels mount datasets at
# /kaggle/input/datasets/<owner>/<slug>/, not /kaggle/input/<slug>/, and a two-level scan
# finds only the owner directory. Depth 5 covers both layouts without assuming either.
def find_pack(max_depth=5):
    cands, stack = [], [(ROOT, 0)]
    while stack:
        d, depth = stack.pop(0)
        try:
            kids = [e for e in sorted(os.scandir(d), key=lambda x: x.name) if e.is_dir()]
        except (PermissionError, OSError, FileNotFoundError):
            continue
        for e in kids:
            if any(h in e.name.lower() for h in PACK_HINTS):
                cands.append(Path(e.path))
            elif depth < max_depth:
                stack.append((Path(e.path), depth + 1))
    return cands

packs = find_pack() if ROOT.is_dir() else []
print("candidate pack roots:", [str(p) for p in packs] or "NONE")
if not packs:
    print("\nnothing matched; here is the whole mount, bounded:")
    seen = 0
    stack = [(ROOT, 0)]
    while stack and seen < 200:
        d, depth = stack.pop(0)
        try:
            kids = sorted(os.scandir(d), key=lambda x: x.name)
        except (PermissionError, OSError, FileNotFoundError):
            continue
        for e in kids[:20]:
            print("  " * depth + "  " + e.name + ("/" if e.is_dir() else ""))
            seen += 1
            if e.is_dir() and depth < 4:
                stack.append((Path(e.path), depth + 1))
    raise SystemExit("Support pack not mounted. Add the dataset as an input.")
PACK = packs[0]
print(f"using {PACK}")
""")

code(r"""
# Bounded walk. A recursive glob over a mount can take minutes when a directory holds
# thousands of small files (notes/17 §4); this caps both depth and per-directory listing.
MAX_DEPTH, MAX_PER_DIR = 6, 40

def walk(root, max_depth=MAX_DEPTH):
    out = []
    stack = [(Path(root), 0)]
    while stack:
        d, depth = stack.pop(0)
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except (PermissionError, OSError, FileNotFoundError) as e:
            out.append((d, None, f"<unreadable: {e}>")); continue
        files = [e for e in entries if e.is_file()]
        dirs = [e for e in entries if e.is_dir()]
        for e in files[:MAX_PER_DIR]:
            try: sz = e.stat().st_size
            except OSError: sz = -1
            out.append((Path(e.path), sz, ""))
        if len(files) > MAX_PER_DIR:
            out.append((d / f"... {len(files)-MAX_PER_DIR} more files", None, ""))
        if depth < max_depth:
            stack += [(Path(e.path), depth + 1) for e in dirs]
    return out

rows = walk(PACK)
print(f"{len(rows)} entries (depth<={MAX_DEPTH}, <={MAX_PER_DIR} files/dir), "
      f"{time.time()-t0:.1f}s\n")
total = 0
for p, sz, note in rows:
    rel = str(p.relative_to(PACK)) if str(p).startswith(str(PACK)) else str(p)
    if sz is None:
        print(f"  {rel} {note}")
    else:
        total += max(sz, 0)
        print(f"  {rel:<70} {sz/1e6:>9.2f} MB")
print(f"\nlisted total {total/1e6:.1f} MB")
""")

code(r"""
# Every checkpoint, described by its state_dict rather than by their code. Keys and shapes
# fully determine the architecture, which is what the reproduction notebook needs.
import torch

ckpt_paths = [p for p, sz, _ in rows if sz and str(p).endswith((".pth", ".pt", ".ckpt"))]
print(f"{len(ckpt_paths)} checkpoint file(s)\n")

DESC = {}
for cp in ckpt_paths:
    rel = str(cp.relative_to(PACK))
    print("=" * 78)
    print(rel)
    try:
        obj = torch.load(cp, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  !! could not load: {type(e).__name__}: {str(e)[:200]}")
        continue
    if isinstance(obj, dict):
        top = list(obj.keys())
        print(f"  top-level keys ({len(top)}): {top[:12]}{' ...' if len(top) > 12 else ''}")
        # Non-tensor entries are usually the training config -- the most useful thing here.
        for k, v in obj.items():
            if not isinstance(v, (dict,)) and not hasattr(v, "shape"):
                print(f"    {k} = {repr(v)[:200]}")
            elif isinstance(v, dict) and not any(hasattr(x, 'shape') for x in v.values()):
                print(f"    {k} (config dict) = {json.dumps(v, default=str)[:400]}")
        sd = None
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]; print(f"  -> state_dict under {key!r}"); break
        if sd is None and all(hasattr(v, "shape") for v in obj.values() if v is not None):
            sd = obj; print("  -> the dict IS the state_dict")
    else:
        sd = getattr(obj, "state_dict", lambda: None)()
        print(f"  object of type {type(obj)}")
    if sd:
        items = [(k, tuple(v.shape)) for k, v in sd.items() if hasattr(v, "shape")]
        n_par = sum(int(torch.tensor(s).prod()) if s else 1 for _, s in items)
        print(f"  {len(items)} tensors, {n_par:,} parameters")
        for k, s in items[:60]:
            print(f"    {k:<58} {s}")
        if len(items) > 60:
            print(f"    ... {len(items)-60} more")
        DESC[rel] = {"n_tensors": len(items), "n_params": int(n_par),
                     "keys": [k for k, _ in items]}
""")

code(r"""
# Config / source files, printed in full when small. These carry the thresholds and the
# linking settings, which are as load-bearing as the weights.
TEXT_EXT = (".json", ".yaml", ".yml", ".py", ".txt", ".cfg", ".ini", ".toml", ".md")
texts = [p for p, sz, _ in rows if sz and str(p).endswith(TEXT_EXT) and sz < 200_000]
print(f"{len(texts)} small text/config file(s)\n")
for tp in texts[:25]:
    rel = str(tp.relative_to(PACK))
    print("=" * 78)
    print(rel)
    try:
        body = tp.read_text(errors="replace")
    except Exception as e:
        print(f"  !! {e}"); continue
    print(body[:4000])
    if len(body) > 4000:
        print(f"  ... [{len(body)-4000} more chars]")

wheels = sorted({str(Path(p).parent) for p, sz, _ in rows if sz and str(p).endswith(".whl")})
print("\nwheel directories:", wheels or "NONE")
""")

code(r"""
out = {"pack": str(PACK),
       "n_entries": len(rows),
       "total_mb": round(total / 1e6, 1),
       "checkpoints": DESC,
       "wheel_dirs": wheels,
       "text_files": [str(Path(p).relative_to(PACK)) for p in texts]}
Path("/kaggle/working/claude_cluster_probe.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"}, indent=2))
print(f"\ncheckpoints described: {list(DESC)}")
print(f"total elapsed {time.time()-t0:.0f}s")
""")

code(r"""
# Dump the sources the reproduction needs, in full, to the log -- kernel OUTPUT files are
# not downloadable from this container (kaggleusercontent.com is denied by egress policy),
# so the log is the only channel back.
WANT = [
    "weights/unet_transformer/split_0/config.json",
    "ARTIFACT_MANIFEST.json",
    "repo/src/biohub_tracking/models/temporal_unet.py",
    "repo/src/biohub_tracking/models/simple_node_transformer.py",
]
for rel in WANT:
    p = PACK / rel
    print("\n" + "#" * 78)
    print("### FILE:", rel)
    print("#" * 78)
    if not p.exists():
        print("  !! not present"); continue
    print(p.read_text(errors="replace"))
""")

code(r"""
# The inference script, separately -- it is the biggest and the one that decides where
# density is controlled, which is exactly where our budget calibration would be injected.
p = PACK / "repo/scripts/predict_unet_transformer.py"
print("### FILE: repo/scripts/predict_unet_transformer.py")
print("#" * 78)
print(p.read_text(errors="replace") if p.exists() else "!! not present")
""")

code(r"""
# CONTAMINATION CHECK. Their model trained on some subset of the same 199 competition
# training datasets, and predict() reads the membership from `data_dir/dataset_splits.json`
# -- which is NOT in the pack. Until it is known which datasets were in split_0's TRAIN
# set, any score measured on competition train data is suspect: a model scores high on
# data it was fitted to, and that number says nothing about the leaderboard.
import os
from pathlib import Path
comp = None
stack = [(Path("/kaggle/input"), 0)]
while stack and comp is None:
    d, depth = stack.pop(0)
    try:
        kids = list(os.scandir(d))
    except (PermissionError, OSError):
        continue
    if any(e.name == "train" for e in kids) and any(e.name == "test" for e in kids):
        comp = d; break
    if depth < 5:
        stack += [(Path(e.path), depth + 1) for e in kids
                  if e.is_dir() and not e.name.endswith((".zarr", ".geff"))]
print("competition root:", comp)
if comp:
    print("top-level entries:")
    for e in sorted(os.scandir(comp), key=lambda x: x.name):
        print("   ", e.name + ("/" if e.is_dir() else ""))
    sp = comp / "dataset_splits.json"
    print(f"\ndataset_splits.json present in competition data: {sp.exists()}")
    if sp.exists():
        print(sp.read_text()[:3000])

# Their own split-generation logic, which is what would let the held-out fold be
# reconstructed exactly rather than guessed.
for rel in ("repo/scripts/dataspec.py", "repo/scripts/evaluate.py"):
    f = PACK / rel
    print("\n" + "#" * 78)
    print("### FILE:", rel)
    print("#" * 78)
    print(f.read_text(errors="replace") if f.exists() else "!! not present")
""")

code(r"""
# Only the split-making part of the training script -- the whole file is 50 KB.
import re
tp = PACK / "repo/scripts/train_unet_transformer.py"
if tp.exists():
    src = tp.read_text(errors="replace")
    print(f"train_unet_transformer.py: {len(src):,} chars")
    hits = [m.start() for m in re.finditer(r"split|fold|KFold|shuffle|seed|random", src)]
    shown, last = 0, -1
    for h in hits:
        a = src.rfind("\n", 0, max(0, h - 400)) + 1
        b = src.find("\n", h + 400)
        if a <= last:
            continue
        last = b
        print("-" * 70)
        print(src[a:b if b > 0 else len(src)])
        shown += 1
        if shown >= 12:
            break
else:
    print("!! train_unet_transformer.py not present")
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
