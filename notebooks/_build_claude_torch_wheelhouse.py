"""Build notebooks/claude_torch_wheelhouse.ipynb — torch 2.5.1+cu121 wheels, offline-ready.

A scored rerun has **internet off**, and Kaggle's free GPU is a P100 (sm_60) whose kernels
the image's torch 2.10+cu128 does not ship (it builds sm_70+). Measured directly:

    Tesla P100-PCIE-16GB, compute_cap 6.0
    torch 2.10.0+cu128 | arch list ['sm_70', 'sm_75', 'sm_80', ...]
    IMAGE TORCH BROKEN: CUDA error: no kernel image is available for execution on the device

`machineShape` is accepted by `kernels/push` and then **ignored** — asking for a T4 still
produced a P100 — so the accelerator cannot be chosen. Every previous run fixed this with
`pip install torch==2.5.1` over the network, which a scored rerun cannot do.

So the wheels are downloaded here, with internet ON, and this kernel's OUTPUT is attached
to the submission notebook as a `kernelDataSource`. That avoids needing to download them
into the dev container at all — `kaggleusercontent.com` is denied by its egress policy.

CPU is not an alternative: the pack's model is 2.1 M parameters with per-voxel temporal
attention, measured at 34.5 s/dataset on the P100. A 10-30x CPU penalty puts a
~200-dataset rerun at 19-58 h against a 12 h ceiling.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_torch_wheelhouse.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# torch 2.5.1+cu121 wheels for an offline P100

**Run this with internet ON.** Its output is attached to the submission notebook as a
kernel data source, so the wheels never need to leave Kaggle.

## Why it is needed

| fact | measured |
|---|---|
| Kaggle free GPU | **Tesla P100, sm_60** |
| image torch | 2.10.0+cu128, builds **sm_70+** only |
| result | `CUDA error: no kernel image is available for execution on the device` |
| `machineShape="nvidiaTeslaT4"` | **accepted and ignored** — still a P100 |
| scored rerun | **internet off**, so `pip install torch==2.5.1` is unavailable |

Downloading the wheels *on Kaggle* is what guarantees they match its Python and platform.

## Why not CPU

The pack's model is 2.1 M parameters with per-voxel attention across time, measured at
**34.5 s/dataset** on the P100. A 10–30× CPU penalty puts a ~200-dataset scored rerun at
19–58 h against a 12 h ceiling. Not viable.
""")

code(r"""
import subprocess, sys, time, os
from pathlib import Path

WHEELS = Path("/kaggle/working/wheels")
WHEELS.mkdir(parents=True, exist_ok=True)

def sh(*a, **kw):
    try:
        return subprocess.run(a, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(a, 127, "", str(e))

print("python", sys.version.split()[0])
print(sh("nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv").stdout.strip())

# `pip download` rather than `pip install`: the point is to collect artefacts, not to
# change this kernel's environment. --only-binary=:all: refuses sdists, which would be
# useless offline (no compiler, no network to fetch build deps).
t0 = time.time()
r = sh(sys.executable, "-m", "pip", "download",
       "torch==2.5.1",
       "--index-url", "https://download.pytorch.org/whl/cu121",
       "--only-binary=:all:",
       "-d", str(WHEELS))
print(f"pip download rc={r.returncode} ({time.time()-t0:.0f}s)")
if r.returncode != 0:
    print(r.stdout[-3000:]); print(r.stderr[-3000:])
    raise SystemExit("torch download failed — the submission cannot run on a P100 without "
                     "these wheels, so this must be fixed rather than worked around.")
""")

code(r"""
files = sorted(WHEELS.glob("*"))
total = sum(f.stat().st_size for f in files)
print(f"{len(files)} files, {total/1e9:.2f} GB\n")
for f in files:
    print(f"  {f.name:<72} {f.stat().st_size/1e6:>9.1f} MB")

# Prove the set is self-sufficient BEFORE relying on it in a scored rerun, where a missing
# transitive dependency is unrecoverable. --no-index means pip may not touch the network;
# --dry-run means this kernel's own torch is left alone.
print("\nresolving offline (--no-index, --dry-run) ...")
r = sh(sys.executable, "-m", "pip", "install", "--no-index",
       f"--find-links={WHEELS}", "--dry-run", "torch==2.5.1")
print(f"  rc={r.returncode}")
print((r.stdout or r.stderr)[-2500:])
if r.returncode != 0:
    raise SystemExit("the wheel set does not resolve offline — a scored rerun would fail "
                     "the same way, with no way to recover.")
print("\nOK: torch==2.5.1 resolves from these wheels with no network.")
""")

code(r"""
import json
manifest = {
    "purpose": "torch 2.5.1+cu121 wheels for an offline P100 (sm_60) submission rerun",
    "why": ("Kaggle's free GPU is a P100 (sm_60); the image torch 2.10+cu128 builds sm_70+ "
            "only and dies with 'no kernel image is available'. machineShape is ignored, "
            "so the accelerator cannot be chosen, and a scored rerun has no internet."),
    "n_files": len(files),
    "total_gb": round(total / 1e9, 3),
    "files": [f.name for f in files],
    "install": "pip install --no-index --find-links=<this dir>/wheels torch==2.5.1",
}
Path("/kaggle/working/torch_wheelhouse_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
print("\nAttach THIS KERNEL's output as a data source in the submission notebook.")
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
