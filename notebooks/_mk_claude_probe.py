"""Dump the competition's own node budget for every movie, on CPU, dependency-free.

    python notebooks/_mk_claude_probe.py

**Why this is worth a kernel.** The score is

    adj_edge_jaccard = edge_J * (1 - 0.1 * (T_pred - T_est) / T_est)

with no upper clamp and a signed ratio (`notes/77` section 7, confirmed against the shipped
`metrics.per_sample_metrics`). `T_est` is `estimated_number_of_nodes`, read out of the ground
truth `.geff` metadata. On the eight movies our local validator scores, our ratio runs from
**-0.29 to +0.36** -- a 65-point spread, worth 6.5 points of multiplier -- and three movies
give back 0.057 of `adj` between them purely by over-predicting.

Closing that spread needs `T_est` per movie at inference time, and the test movies ship no
ground truth. So the question this kernel answers is narrow and decides a whole axis:

  1. Is `estimated_number_of_nodes` present anywhere on the **test** side -- a sidecar, a
     zarr attribute, a metadata file? If yes, per-movie node calibration is a direct edit.
  2. If not, is it **predictable** from things a test movie does show -- frame count, volume,
     voxel scale? All 199 training movies carry both halves, so the regression is free.

**No dependencies.** `geff` is a zarr group and zarr metadata is JSON on disk. Reading
`zarr.json` / `.zattrs` with the standard library cannot fail on a version skew, and
`notes/75`'s polars saga is four rounds of evidence that importing the ecosystem to read a
number is the expensive way to do it.

This reads and prints. It runs no model and writes no submission.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

NB = r'''
import json, os
from pathlib import Path

# The competition does not mount at /kaggle/input/<slug>; on this image it lands under
# /kaggle/input/competitions/<slug>. v1 hardcoded the first and died on iterdir().
_CANDS = [Path("/kaggle/input/biohub-cell-tracking-during-development"),
          Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")]
_CANDS += sorted(Path("/kaggle/input/competitions").glob("*")) \
    if Path("/kaggle/input/competitions").exists() else []
COMP = next((c for c in _CANDS if c.is_dir()), _CANDS[0])
print("COMPETITION ROOT:", COMP, "exists:", COMP.exists(), flush=True)
for p in sorted(Path("/kaggle/input").rglob("*")):
    if p.is_dir() and len(p.relative_to("/kaggle/input").parts) <= 2:
        print("  mount:", p, flush=True)
if not COMP.is_dir():
    raise SystemExit("no competition mount found")

print("\n" + "=" * 78, flush=True)
print("TOP LEVEL", flush=True)
print("=" * 78, flush=True)
for p in sorted(COMP.iterdir()):
    kind = "dir " if p.is_dir() else "file"
    n = len(list(p.iterdir())) if p.is_dir() else p.stat().st_size
    print(f"  {kind} {p.name:<40} {n}", flush=True)

def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None

def zattrs(group):
    """zarr v2 (.zattrs / .zarray) and v3 (zarr.json) in one call."""
    g = Path(group)
    v3 = jload(g / "zarr.json")
    if v3 is not None:
        return v3.get("attributes", v3), v3
    v2 = jload(g / ".zattrs")
    return (v2 or {}), (jload(g / ".zarray") or {})

def arr_len(group):
    """Length of a zarr 1-D array from its metadata alone."""
    a, meta = zattrs(group)
    shape = (meta or {}).get("shape")
    if shape:
        return int(shape[0])
    return None

def walk(group, depth=0, maxdepth=3):
    g = Path(group)
    if depth > maxdepth or not g.is_dir():
        return
    for c in sorted(g.iterdir()):
        if c.name.startswith("c") and c.name[1:].replace(".", "").isdigit():
            continue                      # chunk file, not structure
        if c.is_dir():
            print("   " * depth + "  " + c.name + "/", flush=True)
            walk(c, depth + 1, maxdepth)

# ---------------------------------------------------------------- 1. the TEST side
print("\n" + "=" * 78, flush=True)
print("TEST SIDE -- is a node budget shipped with the movies we must predict?", flush=True)
print("=" * 78, flush=True)
test = COMP / "test"
if not test.exists():
    print("  no test/ directory", flush=True)
else:
    entries = sorted(test.iterdir())
    kinds = {}
    for p in entries:
        kinds[p.suffix or "(none)"] = kinds.get(p.suffix or "(none)", 0) + 1
    print("  entries by suffix:", kinds, flush=True)
    print("  first 10:", [p.name for p in entries[:10]], flush=True)
    for p in entries[:2]:
        print(f"\n  --- structure of {p.name} ---", flush=True)
        walk(p)
        att, meta = zattrs(p)
        print("   root attributes:", json.dumps(att)[:1500], flush=True)
        for sub in ("0", "raw", "image", "nodes", "edges"):
            s = p / sub
            if s.exists():
                a2, m2 = zattrs(s)
                print(f"   {sub}: shape={(m2 or {}).get('shape')} "
                      f"dtype={(m2 or {}).get('dtype') or (m2 or {}).get('data_type')} "
                      f"attrs={json.dumps(a2)[:400]}", flush=True)
    # anything at all outside the .zarr directories?
    extras = [p.name for p in entries if not p.name.endswith(".zarr")]
    print("\n  non-zarr entries in test/:", extras[:40], flush=True)

print("\n  --- any file in the whole mount mentioning a node estimate ---", flush=True)
hits = 0
for p in COMP.rglob("*"):
    if not p.is_file() or p.stat().st_size > 2_000_000:
        continue
    if p.suffix.lower() not in (".json", ".csv", ".txt", ".yaml", ".yml") \
       and p.name not in (".zattrs", ".zgroup", ".zarray", "zarr.json"):
        continue
    try:
        t = p.read_text(errors="replace")
    except Exception:
        continue
    if "estimated_number_of_nodes" in t and "/test/" in str(p):
        print("   TEST HIT:", p, t[:600], flush=True)
        hits += 1
        if hits > 5:
            break
print("   test-side node-estimate files found:", hits, flush=True)

# ---------------------------------------------------------------- 2. the TRAIN table
print("\n" + "=" * 78, flush=True)
print("TRAIN SIDE -- T_est, GT size and movie geometry for every movie", flush=True)
print("=" * 78, flush=True)
train = COMP / "train"
rows = []
if not train.exists():
    print("  no train/ directory", flush=True)
else:
    geffs = sorted(p for p in train.iterdir() if p.name.endswith(".geff"))
    print(f"  {len(geffs)} .geff files", flush=True)
    if geffs:
        print(f"\n  --- structure of {geffs[0].name} ---", flush=True)
        walk(geffs[0])
        att, _ = zattrs(geffs[0])
        print("   root attributes:", json.dumps(att)[:3000], flush=True)

    for g in geffs:
        stem = g.name[:-5]
        att, _ = zattrs(g)
        # geff puts its own metadata under "geff" (v3) or at the root (v2).
        meta = att.get("geff", att) if isinstance(att, dict) else {}
        extra = (meta or {}).get("extra") or {}
        t_est = extra.get("estimated_number_of_nodes")
        n_nodes = None
        for cand in (g / "nodes" / "ids", g / "nodes" / "props" / "t" / "values",
                     g / "nodes" / "ids" / "values"):
            n_nodes = n_nodes or arr_len(cand)
        n_edges = None
        for cand in (g / "edges" / "ids", g / "edges" / "ids" / "values"):
            n_edges = n_edges or arr_len(cand)
        if n_edges is not None and n_edges and isinstance(n_edges, int):
            pass
        z = train / f"{stem}.zarr"
        shape = scale = None
        if z.exists():
            for sub in ("", "0", "raw", "image"):
                a2, m2 = zattrs(z / sub if sub else z)
                if (m2 or {}).get("shape"):
                    shape = m2["shape"]
                    break
            az, _ = zattrs(z)
            scale = (az or {}).get("voxel_size_um") or (az or {}).get("scale") \
                or (az or {}).get("pixel_size_um")
        rows.append(dict(stem=stem, t_est=t_est, gt_nodes=n_nodes, gt_edges=n_edges,
                         shape=shape, scale=scale,
                         axes=(meta or {}).get("axes") and len((meta or {}).get("axes"))))

    print(f"\n  stem,t_est,gt_nodes,gt_edges,shape,scale", flush=True)
    for r in rows:
        print(f"  {r['stem']},{r['t_est']},{r['gt_nodes']},{r['gt_edges']},"
              f"\"{r['shape']}\",\"{r['scale']}\"", flush=True)
    have = [r for r in rows if r["t_est"] is not None]
    print(f"\n  movies with estimated_number_of_nodes: {len(have)}/{len(rows)}", flush=True)
    if have:
        vals = sorted(float(r["t_est"]) for r in have)
        print(f"  T_est  min={vals[0]:,.0f}  median={vals[len(vals)//2]:,.0f} "
              f"max={vals[-1]:,.0f}  sum={sum(vals):,.0f}", flush=True)
    Path("/kaggle/working/node_budget_table.json").write_text(json.dumps(rows, indent=1))
    print("  wrote node_budget_table.json", flush=True)

print("\nPROBE COMPLETE", flush=True)
'''


def build() -> int:
    compile(NB, "claude_probe", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": NB.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / "claude_probe.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    (HERE / "claude_probe_push.json").write_text(json.dumps({
        "slug": "claude-probe", "title": "Claude probe",
        "notebook": str(out), "dataset_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [], "enable_gpu": False, "enable_internet": False,
    }, indent=1))
    print(f"wrote {out.name}: {len(NB):,} chars, CPU only, competition mount")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
