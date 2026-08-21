"""Build notebooks/10_wheelhouse.ipynb."""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/10_wheelhouse.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Wheelhouse — make `zarr` installable with the internet off

**Run this once, with internet ON.** Then save its output as a Kaggle Dataset and attach
that Dataset to `09_submission.ipynb`.

`zarr` is not in the Kaggle image. Measured 2026-08-21 on python 3.12.13 / numpy 2.0.2 /
scipy 1.16.3: `import zarr` fails, and a scored rerun has no internet, so `pip install
zarr` cannot fix it. The standard answer is a *wheelhouse*: download the `.whl` files
here, ship them as a Dataset, and install with `--no-index --find-links=…`, which reads
only local files.

The wheels must match the machine that will use them, which is why this runs on Kaggle
rather than on a laptop — same python, same platform tag, same numpy ABI.

This notebook does three things:

1. downloads `zarr` and everything it needs into `/kaggle/working/wheels`
2. **verifies the offline install actually works**, by installing from those files with
   the network disabled via `--no-index`
3. opens one real competition `.zarr` with the result, so the check is "can we read the
   data", not merely "did the import succeed"

Step 3 is the point. A wheelhouse that installs but cannot open an OME-Zarr store would
fail at hour three of a submission instead of here.
""")

code(r"""
import subprocess, sys, os, shutil
from pathlib import Path

WHEELS = Path("/kaggle/working/wheels")
if WHEELS.exists():
    shutil.rmtree(WHEELS)
WHEELS.mkdir(parents=True)

print(f"python {sys.version.split()[0]}")
r = subprocess.run([sys.executable, "-m", "pip", "download", "zarr", "-d", str(WHEELS)],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
if r.returncode != 0:
    print(r.stderr[-3000:])
    raise SystemExit("pip download failed — is internet enabled for this notebook? "
                     "Settings -> Internet -> On.")

got = sorted(WHEELS.glob("*"))
print(f"\n{len(got)} files in {WHEELS}:")
for f in got:
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
sdists = [f for f in got if not f.name.endswith(".whl")]
if sdists:
    # A source distribution means pip found no matching wheel and would have to BUILD it
    # at install time. That needs a compiler and often the network, so it defeats the
    # purpose. Name them rather than let 09 discover it during a scored rerun.
    print(f"\n!! not wheels, would need building at install time: {[f.name for f in sdists]}")
""")

md("""## Verify the offline install

`--no-index` tells pip to ignore PyPI entirely, so this is the same code path a scored
rerun takes. If it works here it works there.
""")

code(r"""
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    f"--find-links={WHEELS}", "zarr"],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
if r.returncode != 0:
    print(r.stderr[-3000:])
    raise SystemExit("offline install failed — 09 would fail the same way.")

# Fresh interpreter: this notebook may already have zarr imported from the download step,
# which would make an in-process import prove nothing about the installed package.
probe = ("import zarr, numcodecs, numpy; "
         "print('zarr', zarr.__version__, '| numcodecs', numcodecs.__version__, "
         "'| numpy', numpy.__version__)")
r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-2000:])
if r.returncode != 0:
    raise SystemExit("zarr installed but does not import — the wheelhouse is not usable.")
""")

md("""## Verify it can open real competition data

The install working is not the same as the data being readable. This opens one of the
actual `.zarr` stores and pulls a frame.
""")

code(r"""
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

COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Could not find the competition data — add it as an input.")

sample = sorted((COMP / "train").glob("*.zarr"))[0]
check = (
    "import zarr, numpy as np\n"
    f"g = zarr.open_group(r'{sample}', mode='r')\n"
    "a = g['0']\n"
    "print('opened', a.shape, a.dtype)\n"
    "f = np.asarray(a[0])\n"
    "print('frame 0:', f.shape, 'min', float(f.min()), 'max', float(f.max()))\n"
    "print('attrs keys:', sorted(dict(g.attrs))[:6])\n"
)
r = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
print(f"reading {sample.name}:")
print(r.stdout or r.stderr[-3000:])
if r.returncode != 0:
    raise SystemExit("the wheelhouse zarr cannot read the competition data.")
print("WHEELHOUSE OK — install works offline and reads real data.")
""")

md("""## Save it

1. **Save Version** (Save & Run All). Wait for it to finish.
2. On the finished version, open the **Output** tab → **New Dataset** → name it
   something like `biohub-wheels`.
3. In `09_submission.ipynb`, **Add Input** → that dataset.

Cell 1 of `09` scans the attached inputs for a directory of `.whl` files and installs
from it. Nothing in `09` needs editing.

One caution: rebuild this wheelhouse if Kaggle's image changes python version. Wheels are
tagged to a specific python (`cp312`), and a `cp312` wheel will not install on `cp313`.
`09` will say so in its first cell rather than failing later.
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
