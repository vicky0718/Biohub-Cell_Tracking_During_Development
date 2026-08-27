"""Build notebooks/claude_deepcenter_veto.ipynb — ask a second detector whether the nodes
the repair chain invents are actually there.

`notes/33`: the 0.927 public notebook attaches three model datasets where we attach one.
This tests the cheapest of the two we are missing,
`pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, which is never used to *find* cells.
It is only ever used to *reject* one: a gap-closed node must score above a threshold on
this model's heatmap before the repair commits it.

The reason to test this one first is that it targets a cost this project has already
measured twice and accepted. `close_gaps` inserts a node at the midpoint of a plausible
gap **without looking at the image**, and both `notes/27` §1 and `notes/31` §3 recorded the
same signature: `fn_gap` and `fn_mislink` fall, `fn_detect` rises (+36 at the best ILP arm).
Some invented nodes land where no cell is. This asks.

Nearly free, like the two sweeps before it: the ILP re-solves from `claude_relink_sweep`'s
cached candidates, so there is no main-model inference. The only new compute is a small
UNet over the raw frames.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_deepcenter_veto.ipynb")
N_DATASETS = 12
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Does a second detector know which invented nodes are real?

```
0.752  classical champion
0.880  pack + ILP + gap-close + linefit-smooth
0.883  + disappearance 0.1 -> 0.5              <- best scored
0.923–0.927  the PUBLIC notebooks                notes/33
0.944  gold cutoff        0.926  bronze          2,792 teams, median 0.894
```

`notes/33` found that the public notebooks attach **three** model datasets where we attach
one, and that ~0.04 of the gap is sitting in weights that are already public. This tests
the first of the two missing models.

## The mechanism, and why this one first

`biohub-deepcenter-unet3d-center-prior-v1` is not part of the public notebook's detection
path. It never adds a node. It is an **add-only veto**: every gap-closed node must score
above `GAP_THRESHOLD` on its heatmap before the repair is allowed to commit it.

That targets a cost this project measured and accepted rather than fixed. `close_gaps`
puts a node at the midpoint of a plausible gap **without ever looking at the image**, and
the anatomy has twice shown what that buys and what it costs:

```
notes/31 §3, control+repair -> asym0.1_0.5+repair
  fn_mislink  411 -> 382  (-29)      fn_gap  177 -> 159  (-18)
  fn_detect   218 -> 254  (+36)   <- invented nodes that match nothing
```

## Pre-registered predictions

1. **The heatmap is aligned to our coordinate convention.** Annotated GT nodes score far
   higher than random in-volume points. This is load-bearing and everything else depends
   on it: the heatmap keeps full z and is pooled 4x in y/x, so a convention error scores
   the wrong voxel and the veto still "works" — it just vetoes at random.
   `probes/exec_deepcenter.py` pins the geometry on synthetic data; this checks it against
   the real checkpoint and real cells.
2. **`repair` with no veto reproduces `notes/31`'s arm** on this dataset subset, so the
   veto arms are being compared against the thing that actually scored 0.883.
3. **The veto rejects a real minority of candidates** — not ~0% (the threshold does
   nothing) and not ~100% (misaligned, or the threshold is absurd). Either extreme means
   the arms below are not measuring what they claim.
4. **The veto reduces `fn_detect`.** The mechanism claim, stated separately from the
   outcome, because they can come apart: the veto can work exactly as designed and still
   lose on total score by refusing gaps that were worth two edges each.
5. **At its best threshold the veto beats plain repair on score.** The outcome claim.

*Training data, contaminated for the pack's weights (`notes/24` §2). `notes/32` §2 has two
transfer measurements that do not agree once leaderboard rounding is propagated — read a
train gain as roughly face value, not through a multiplier.*
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
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
    return r.returncode == 0

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

PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir(), ["/kaggle/input"])
REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
CACHE = find_dir(lambda p: any(p.glob("cand_*.npz")), ["/kaggle/input"])
# The new one. Identified by its own weight layout rather than by dataset name, so a
# renamed or re-versioned mount still resolves.
DC = find_dir(lambda p: any(p.glob("full_frame_center/*.pt")), ["/kaggle/input"])
DC_ROOT = find_dir(lambda p: "deepcenter" in p.name.lower(), ["/kaggle/input"], max_depth=3)
for lbl, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
               ("cand cache", CACHE), ("deepcenter", DC), ("dc root", DC_ROOT)):
    print(f"  {lbl:<13} {v}")

print("\ndeepcenter dataset tree:")
root = DC_ROOT or DC
if root:
    for p in sorted(Path(root).rglob("*"))[:40]:
        if p.is_file():
            print(f"    {p.relative_to(root)}  {p.stat().st_size/1e6:.1f} MB")
if None in (PACK, REPO, COMP, CACHE) or root is None:
    raise SystemExit("missing mount — needs the pack, our repo, the competition data, "
                     "claude-relink-sweep as a kernel source, and the deepcenter dataset")
TRAIN = COMP / "train"
print(f"\n  cached instances  {len(list(CACHE.glob('cand_*.npz')))}")

CKPTS = sorted(Path(root).rglob("*.pt"))
print("  checkpoints:", [str(p.relative_to(root)) for p in CKPTS])
if not CKPTS:
    raise SystemExit("no .pt checkpoint under the deepcenter mount")

ok = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                 extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok else 'FAILED'}")

CELLMOT = Path("/kaggle/working/kaggle-cell-tracking-competition")
if not (CELLMOT / "src" / "tracking_cellmot").is_dir():
    r = sh("git", "clone", "--depth", "1",
           "https://github.com/royerlab/kaggle-cell-tracking-competition", str(CELLMOT))
    print(f"official scorer clone rc={r.returncode}")
os.environ["CELLMOT_REPO"] = str(CELLMOT)

probe = sh(sys.executable, "-c",
           "import numpy, zarr, torch, tracksdata; import tracksdata.solvers; "
           "print('numpy', numpy.__version__, '| torch', torch.__version__, "
           "'| cuda', torch.cuda.is_available())")
print(probe.stdout.strip() or probe.stderr.strip()[-800:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import in a fresh interpreter")
print(f"setup took {time.time()-T_START:.0f}s")
""")

md("""
## 1. Load the checkpoint, check the alignment, then run the arms

The alignment check comes before any arm. `load_state_dict` is strict, so a wrong module
tree fails at load; what it cannot catch is a *coordinate* convention error, which produces
a perfectly valid heatmap read at the wrong voxel. So the first thing the worker does is
score real annotated GT nodes against random in-volume points and print both.
""")

code(r"""
WORKER = WORK / "run_dc.py"
WORKER.write_text(f'''
import json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["CELLMOT_REPO"] = {str(CELLMOT)!r}
REPO = Path({str(REPO)!r}); TRAIN = Path({str(TRAIN)!r})
CACHE = Path({str(CACHE)!r}); WORK = Path({str(WORK)!r})
CKPTS = {[str(p) for p in CKPTS]!r}
N_DATASETS = {N_DATASETS}
T0 = time.time()

sys.path.insert(0, str(REPO))
import polars as pl
import tracksdata as td
import zarr
from harness import Harness
from harness.tracks import Tracks, read_geff, read_scale
from harness.purescore import summarise
from pipeline.anatomy import BUCKETS, edge_anatomy, summarise_anatomy
from pipeline.repair import close_gaps, linefit_smooth
from pipeline import deepcenter as dc
print("worker numpy", np.__version__, flush=True)

# ---- the second detector -------------------------------------------------------
BUNDLE, LOAD_ERRORS = None, []
for cp in CKPTS:
    try:
        BUNDLE = dc.load(cp)
        print(f"loaded {{cp}}  epoch={{BUNDLE['epoch']}} best_score={{BUNDLE['best_score']}} "
              f"pool={{BUNDLE['pool_factor']}} device={{BUNDLE['device']}}", flush=True)
        break
    except Exception as exc:
        LOAD_ERRORS.append(f"{{cp}}: {{type(exc).__name__}}: {{exc}}")
if BUNDLE is None:
    # Never fall back to "no veto" -- that silently turns every veto arm into its own
    # control and reports the two as different.
    raise SystemExit("no deepcenter checkpoint loaded:\\n" + "\\n".join(LOAD_ERRORS))
NPAR = sum(p.numel() for p in BUNDLE["model"].parameters())
print(f"  {{NPAR:,}} parameters", flush=True)

def open_volume(name):
    return zarr.open_group(str(TRAIN / f"{{name}}.zarr"), mode="r")["0"]

def repair_chain(g, sc, accept=None):
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038,
                   max_added_abs=1650, accept=accept)
    return linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)

def build_td(t, zyx, cand):
    # Rebuild the ILP's INPUT graph. A docstring here would terminate the outer f-string
    # that writes this file -- the trap that has now bitten six notebooks in this project.
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    g.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    ids = g.bulk_add_nodes([{{"t": int(tt), "z": float(z), "y": float(y), "x": float(x)}}
                            for tt, (z, y, x) in zip(t, zyx)])
    if len(cand):
        g.bulk_add_edges([{{"source_id": ids[int(s)], "target_id": ids[int(d)],
                           "edge_prob": float(p)}} for s, d, p in cand[:, :3]])
    return g, np.asarray(ids, np.int64)

def solve(g_td, ew, ap, dis, dv):
    if g_td.num_edges() == 0:
        return g_td
    solver = td.solvers.ILPSolver(
        edge_weight=ew * td.EdgeAttr("edge_prob"),
        appearance_weight=ap, disappearance_weight=dis, division_weight=dv)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return solver.solve(g_td)

BEST = (-1.0, 0.1, 0.5, 1.0)          # notes/32: the weights scoring 0.883
THRESHOLDS = [0.10, 0.25, 0.40, 0.60]  # 0.25 is the public notebook's own GAP_THRESHOLD
LABELS = ["norepair", "repair"] + [f"veto{{th}}" for th in THRESHOLDS]

names = sorted(p.stem[len("cand_"):] for p in CACHE.glob("cand_*.npz"))
names = [n for n in names if (TRAIN / f"{{n}}.geff").exists()][:N_DATASETS]
print(f"{{len(names)}} datasets, {{len(LABELS)}} arms", flush=True)

# ---- PREDICTION 1: is the heatmap aligned to our coordinates? -------------------
print("\\n" + "=" * 78, flush=True)
print("ALIGNMENT: annotated GT nodes vs random in-volume points", flush=True)
print("=" * 78, flush=True)
ALIGN = {{}}
rng = np.random.default_rng(0)
for name in names[:3]:
    arr = open_volume(name)
    gt = read_geff(TRAIN / f"{{name}}.geff")
    scorer = dc.FrameScorer(BUNDLE, lambda t, a=arr: np.asarray(a[int(t)]).astype(np.float32),
                            max_frames=4)
    frames = sorted(set(int(f) for f in gt.t))[:4]
    sel = np.isin(gt.t, frames)
    gt_t, gt_zyx = gt.t[sel], gt.zyx[sel]
    if len(gt_t) == 0:
        continue
    s_gt = scorer.score(gt_t, gt_zyx)
    shape = np.asarray(arr.shape[1:], float)
    rnd = rng.random((len(gt_t), 3)) * shape[None, :]
    s_rnd = scorer.score(gt_t, rnd)
    ALIGN[name] = {{"n": int(len(gt_t)),
                   "gt_median": float(np.median(s_gt)), "gt_mean": float(s_gt.mean()),
                   "rnd_median": float(np.median(s_rnd)), "rnd_mean": float(s_rnd.mean()),
                   "frac_gt_above_0.25": float((s_gt >= 0.25).mean()),
                   "frac_rnd_above_0.25": float((s_rnd >= 0.25).mean())}}
    a = ALIGN[name]
    print(f"  {{name}}  n={{a['n']:,}}  GT median {{a['gt_median']:.3f}} "
          f"(>=0.25: {{a['frac_gt_above_0.25']:.1%}})   random median {{a['rnd_median']:.3f}} "
          f"(>=0.25: {{a['frac_rnd_above_0.25']:.1%}})", flush=True)

# ---- the arms -------------------------------------------------------------------
h = Harness(data_dir=TRAIN, cache_dir=None)
ROWS = {{l: [] for l in LABELS}}
ANAT = {{l: [] for l in LABELS}}
NODES = {{l: 0 for l in LABELS}}
EDGES = {{l: 0 for l in LABELS}}
VETO = {{f"veto{{th}}": {{"proposed": 0, "kept": 0}} for th in THRESHOLDS}}
PER = {{}}

for name in names:
    t0 = time.time()
    z = np.load(CACHE / f"cand_{{name}}.npz")
    t, zyx, cand = z["t"], z["zyx"], z["cand"]
    sc = read_scale(TRAIN / f"{{name}}.zarr")
    gt = read_geff(TRAIN / f"{{name}}.geff")
    arr = open_volume(name)
    base_td, _ = build_td(t, zyx, cand)
    g_td = solve(base_td, *BEST)
    tr = Tracks.from_tracksdata(g_td)
    base = (tr.t, tr.zyx, tr.edges)
    print(f"\\n{{name}}  nodes={{len(t):,}} cand={{len(cand):,}} kept={{tr.n_edges:,}}", flush=True)

    # One scorer for ALL veto arms on this dataset, sized to hold every frame. A heatmap
    # is 64x64x64 float32 = 1 MB at pool 4, so 110 of them is ~110 MB and each frame is
    # read from zarr and forward-passed ONCE per dataset instead of once per arm.
    scorer = dc.FrameScorer(BUNDLE, lambda tt, a=arr: np.asarray(a[int(tt)]).astype(np.float32),
                            max_frames=110)
    for lbl in LABELS:
        if lbl == "norepair":
            g = base
        elif lbl == "repair":
            g = repair_chain(base, sc)
        else:
            th = float(lbl[len("veto"):])
            counter = {{"proposed": 0, "kept": 0}}
            def acc(t_mid, zyx_mid, th=th, counter=counter):
                keep = scorer.score(t_mid, zyx_mid) >= th
                counter["proposed"] += int(len(keep)); counter["kept"] += int(keep.sum())
                return keep
            g = repair_chain(base, sc, accept=acc)
            VETO[lbl]["proposed"] += counter["proposed"]
            VETO[lbl]["kept"] += counter["kept"]
        ROWS[lbl].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
        NODES[lbl] += int(len(g[0])); EDGES[lbl] += int(len(g[2]))
        a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
        ANAT[lbl].append(a)
        if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
            raise SystemExit(f"{{name}}/{{lbl}}: buckets do not sum")
        PER.setdefault(name, {{}})[lbl] = float(
            ROWS[lbl][-1].get("adj_edge_jaccard", float("nan")))
    # frames_computed should be ~the frame count, NOT that times the number of veto arms.
    # If it is 4x too high the cache is thrashing and the run will take 4x as long.
    print(f"    repair {{PER[name]['repair']:.4f}}  " + "  ".join(
        f"{{l}} {{PER[name][l]-PER[name]['repair']:+.4f}}" for l in LABELS if l.startswith("veto")) +
        f"   heatmaps={{scorer.frames_computed}}  {{time.time()-t0:.0f}}s", flush=True)

    out = {{"arms": LABELS, "datasets": [n for n in names if n in PER],
           "thresholds": THRESHOLDS, "alignment": ALIGN, "veto": VETO,
           "summary": {{l: summarise(ROWS[l]) for l in LABELS if ROWS[l]}},
           "anatomy": {{l: summarise_anatomy(ANAT[l]) for l in LABELS if ANAT[l]}},
           "nodes": NODES, "edges": EDGES, "per_dataset": PER,
           "checkpoint": str(BUNDLE["path"]), "n_params": int(NPAR)}}
    (WORK / "deepcenter_veto.json").write_text(json.dumps(out, indent=2, default=float))

print(f"\\nworker done in {{time.time()-T0:.0f}}s", flush=True)
''')

t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", str(WORKER)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print(f"\nworker exited {rc} after {time.time()-t0:.0f}s")
if rc != 0:
    raise SystemExit(f"worker failed ({rc})")
""")

md("""## 2. The five predictions""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "deepcenter_veto.json").read_text())
S, A, N, E = D["summary"], D["anatomy"], D["nodes"], D["edges"]
ARMS, DS, AL, V = D["arms"], D["datasets"], D["alignment"], D["veto"]
EXACT = S["repair"]["score"] == S["repair"]["score"]
key = "score" if EXACT else "edge_jaccard"
REF = "repair"
print(f"{len(DS)} datasets, {len(ARMS)} arms   checkpoint {D['checkpoint']}")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

print(f"{'arm':<12}{'score':>9}{'vs repair':>11}{'edge_J':>9}{'mislink':>9}"
      f"{'gap':>7}{'detect':>8}{'nodes':>10}{'edges':>10}")
print("-" * 85)
for a in ARMS:
    if a not in S:
        continue
    st, an = S[a], A[a]
    print(f"{a:<12}{st[key]:>9.4f}{st[key]-S[REF][key]:>+11.4f}{st['edge_jaccard']:>9.4f}"
          f"{an['fn_mislink']:>9,}{an['fn_gap']:>7,}{an['fn_detect']:>8,}"
          f"{N[a]:>10,}{E[a]:>10,}")

print()
print("=" * 85)
print("PREDICTION GRADING")
print("=" * 85)

# 1 ---------------------------------------------------------------------------------
print("\n1. the heatmap is aligned — GT nodes score far above random in-volume points")
if not AL:
    print("   NOT GRADED — no alignment samples recorded.")
else:
    ok1 = True
    for nm, a in AL.items():
        good = a["gt_median"] > 0.25 and a["gt_median"] > 3 * max(a["rnd_median"], 1e-3)
        ok1 &= good
        print(f"   {nm:<16} GT {a['gt_median']:.3f} vs random {a['rnd_median']:.3f}"
              f"   ratio {a['gt_median']/max(a['rnd_median'],1e-3):>6.1f}x  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"   ->  {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        print("   The heatmap does NOT light up on real cells at our coordinates. Either the")
        print("   z/y/x pooling convention is wrong (probes/exec_deepcenter.py pins it on")
        print("   synthetic data — re-check against the real volume shape) or this checkpoint")
        print("   is not the center-prior detector. NOTHING BELOW IS READABLE until this passes.")

# 2 ---------------------------------------------------------------------------------
print("\n2. `repair` with no veto reproduces the arm that scored 0.883")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    print(f"   repair = {S['repair']['score']:.4f} on {len(DS)} datasets")
    print(f"   notes/31 measured 0.8958 for the same chain on 24 datasets — a different")
    print(f"   subset, so this is a sanity range, not an equality check.")
    ok2 = 0.85 <= S["repair"]["score"] <= 0.94
    print(f"   ->  {'PASS — in range' if ok2 else 'FAIL — out of range, the base is not the 0.883 chain'}")
    print(f"   repair vs norepair: {S['repair'][key]-S['norepair'][key]:+.4f}"
          "   (the +0.0115-ish that notes/28 transferred to the LB)")

# 3 ---------------------------------------------------------------------------------
print("\n3. the veto rejects a real minority of candidates — neither ~0% nor ~100%")
ok3 = True
for lbl, v in V.items():
    p, k = v["proposed"], v["kept"]
    rej = (p - k) / p if p else float("nan")
    good = 0.01 < rej < 0.95
    ok3 &= good
    print(f"   {lbl:<10} proposed {p:>7,}  kept {k:>7,}  rejected {rej:>6.1%}"
          f"  {'ok' if good else '<-- degenerate'}")
print(f"   ->  {'PASS' if ok3 else 'FAIL'}")
if not ok3:
    print("   A threshold that rejects ~nothing is not being tested; one that rejects almost")
    print("   everything means the scores are not on the scale the public notebook assumes.")

# 4 ---------------------------------------------------------------------------------
print("\n4. the veto reduces fn_detect (the MECHANISM claim)")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    d0 = A["repair"]["fn_detect"]
    dn = A["norepair"]["fn_detect"]
    print(f"   norepair {dn:,} -> repair {d0:,}  (repair's own detection tax: {d0-dn:+,})")
    best_d, best_l = min((A[a]["fn_detect"], a) for a in ARMS if a.startswith("veto"))
    ok4 = best_d < d0
    print(f"   best veto: {best_l} at {best_d:,}  ({best_d-d0:+,} vs repair)"
          f"  ->  {'PASS' if ok4 else 'FAIL'}")
    if ok4:
        recovered = (d0 - best_d) / max(d0 - dn, 1)
        print(f"   The veto claws back {recovered:.0%} of the detection tax repair pays.")
    else:
        print("   The veto does not reduce the bucket it exists to reduce. Either the rejected")
        print("   nodes were the ones that DID match, or fn_detect is not dominated by invented")
        print("   nodes at all — check norepair's own fn_detect above before going further.")

# 5 ---------------------------------------------------------------------------------
print("\n5. at its best threshold the veto beats plain repair on score (the OUTCOME claim)")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    cands = [(S[a][key], a) for a in ARMS if a.startswith("veto") and a in S]
    bv, ba = max(cands)
    ok5 = bv > S[REF][key] + 1e-6
    for v, a in sorted(cands, reverse=True):
        print(f"   {a:<10} {v:.4f}  ({v-S[REF][key]:+.4f})")
    print(f"   ->  {'PASS' if ok5 else 'FAIL'}")
    if ok5:
        print(f"   {ba} is submittable on top of the 0.883 chain.")
    else:
        print("   The veto works mechanically (see 4) but costs more than it saves: each")
        print("   refused gap is two edges not recovered, and that outweighs the nodes it")
        print("   stops stranding. If 4 passed and 5 failed, the honest reading is that the")
        print("   gap-close radius is already tight enough that most midpoints are real.")

print()
print("=" * 85)
if EXACT:
    best = max(ARMS, key=lambda a: S[a][key] if a in S else float("-inf"))
    print(f"BEST ARM: {best} at {S[best][key]:.4f}  ({S[best][key]-S[REF][key]:+.4f} vs repair)")
    print(f"  remaining: mislink {A[best]['fn_mislink']:,}  gap {A[best]['fn_gap']:,}  "
          f"undetected {A[best]['fn_detect']:,}")
    if best == REF:
        print("  The veto is not worth a submission slot on this evidence. The SECOND missing")
        print("  model (temporal-unet3d-seed314159, the linker blend) is untouched by this")
        print("  result — it acts on edge probabilities, not on invented nodes.")
else:
    print("NO BEST ARM — score column is NaN.")
print("=" * 85)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
