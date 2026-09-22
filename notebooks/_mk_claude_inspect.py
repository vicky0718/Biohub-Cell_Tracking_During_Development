"""Mount a public dataset and describe it, on CPU, without touching GPU quota.

    python notebooks/_mk_claude_inspect.py

`bhpepper/biohub-synthetic-5fold-ensemble-v1` is 46 MB, undocumented (usability 0.4375),
and published by the only team sitting at exactly our 0.950 target. `/datasets/list/files`
404s for it, so the only way to see inside is to mount it.

This inspects and reports; it changes nothing and runs nothing. `notes/86` is why that
separation matters -- plugging foreign weights into this stack is the `ftune` failure mode,
and that decision should be made with the file list in hand rather than before it.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

NB = r'''
import hashlib, json, os
from pathlib import Path

ROOT = Path("/kaggle/input")
print("MOUNTS:", flush=True)
for p in sorted(ROOT.glob("*")):
    print("  ", p, flush=True)

print("\n" + "=" * 78, flush=True)
print("FILE TREE", flush=True)
print("=" * 78, flush=True)
files = []
for p in sorted(ROOT.rglob("*")):
    if p.is_file():
        files.append(p)
        rel = p.relative_to(ROOT)
        print(f"  {str(rel)[:74]:<76} {p.stat().st_size/1e6:>9.3f} MB", flush=True)
print(f"\n{len(files)} files", flush=True)

print("\n" + "=" * 78, flush=True)
print("TEXT / JSON CONTENTS", flush=True)
print("=" * 78, flush=True)
for p in files:
    if p.suffix.lower() in (".json", ".txt", ".md", ".yaml", ".yml", ".csv", ".cfg"):
        print(f"\n--- {p.relative_to(ROOT)} ---", flush=True)
        try:
            t = p.read_text(errors="replace")
            print(t[:4000], flush=True)
            if len(t) > 4000:
                print(f"   ... ({len(t):,} chars total)", flush=True)
        except Exception as e:
            print("   unreadable:", e, flush=True)

print("\n" + "=" * 78, flush=True)
print("CHECKPOINTS", flush=True)
print("=" * 78, flush=True)
import torch
for p in files:
    if p.suffix.lower() not in (".pt", ".pth", ".ckpt", ".bin", ".safetensors"):
        continue
    print(f"\n--- {p.relative_to(ROOT)}  ({p.stat().st_size/1e6:.2f} MB) ---", flush=True)
    print("   sha256:", hashlib.sha256(p.read_bytes()).hexdigest()[:32], flush=True)
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print("   torch.load failed:", str(e)[:160], flush=True)
        continue
    print("   type:", type(obj).__name__, flush=True)
    if isinstance(obj, dict):
        print("   top-level keys:", list(obj)[:20], flush=True)
        sd = obj
        for k in ("state_dict", "model", "model_state_dict", "weights"):
            if k in obj and isinstance(obj[k], dict):
                print(f"   -> nested state_dict under '{k}'", flush=True)
                sd = obj[k]
                break
        for k in ("epoch", "score", "best_score", "config", "cfg", "args",
                  "method", "fold", "split", "manifest", "arch"):
            if k in obj and not isinstance(obj[k], dict):
                print(f"   {k}: {str(obj[k])[:180]}", flush=True)
            elif k in obj:
                print(f"   {k}: {json.dumps(obj[k], default=str)[:400]}", flush=True)
        tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")}
        if tensors:
            tot = sum(v.numel() for v in tensors.values())
            print(f"   tensors: {len(tensors)}   params: {tot:,}", flush=True)
            names = list(tensors)
            print("   first 12 tensor names:", flush=True)
            for n in names[:12]:
                print(f"      {n:<58} {tuple(tensors[n].shape)}", flush=True)
            print("   last 6:", flush=True)
            for n in names[-6:]:
                print(f"      {n:<58} {tuple(tensors[n].shape)}", flush=True)
            prefixes = sorted({n.split(".")[0] for n in names})
            print("   top-level module prefixes:", prefixes[:20], flush=True)
print("\n" + "=" * 78, flush=True)
print("COMPATIBILITY: ours vs theirs", flush=True)
print("=" * 78, flush=True)
ours = [p for p in files if "support-pack" in str(p) and p.suffix in (".pth", ".pt")
        and "edge_predictor" in p.name]
theirs = [p for p in files if "5fold" in str(p) and p.name.endswith("swa.pth")]
if not ours or not theirs:
    print("   need BOTH the support pack and the ensemble mounted", flush=True)
    print("   ours  :", [str(x.name) for x in ours][:4], flush=True)
    print("   theirs:", [str(x.name) for x in theirs][:4], flush=True)
else:
    def sd(path):
        o = torch.load(path, map_location="cpu", weights_only=False)
        for k in ("state_dict", "model", "model_state_dict"):
            if isinstance(o, dict) and k in o and isinstance(o[k], dict):
                o = o[k]; break
        return {k: tuple(v.shape) for k, v in o.items() if hasattr(v, "shape")}
    A, B = sd(ours[0]), sd(theirs[0])
    print(f"   ours   {ours[0].name}: {len(A)} tensors, "
          f"{sum(__import__('math').prod(v) if v else 1 for v in A.values()):,} params", flush=True)
    print(f"   theirs {theirs[0].name}: {len(B)} tensors, "
          f"{sum(__import__('math').prod(v) if v else 1 for v in B.values()):,} params", flush=True)
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    common = sorted(set(A) & set(B))
    print(f"\n   common keys      : {len(common)}", flush=True)
    print(f"   only in OURS     : {len(only_a)}", flush=True)
    for k in only_a[:12]: print(f"      {k:<52} {A[k]}", flush=True)
    print(f"   only in THEIRS   : {len(only_b)}", flush=True)
    for k in only_b[:12]: print(f"      {k:<52} {B[k]}", flush=True)
    mismatch = [(k, A[k], B[k]) for k in common if A[k] != B[k]]
    print(f"\n   shape mismatches among common keys: {len(mismatch)}", flush=True)
    for k, a, b in mismatch[:12]: print(f"      {k:<44} ours {a}  theirs {b}", flush=True)
    print(f"\n   VERDICT: {'DROP-IN COMPATIBLE' if not only_a and not only_b and not mismatch else 'NOT drop-in'}", flush=True)
print("\nINSPECTION DONE", flush=True)
'''


def build(datasets: list[str]) -> int:
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": NB.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    compile(NB, "claude_inspect", "exec")
    out = HERE / "claude_inspect.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    (HERE / "claude_inspect_push.json").write_text(json.dumps({
        "slug": "claude-inspect", "title": "Claude inspect",
        "notebook": str(out), "dataset_sources": datasets,
        "competition_sources": [], "kernel_sources": [],
        "enable_gpu": False, "enable_internet": False,
    }, indent=1))
    print(f"wrote {out.name}: {len(NB):,} chars, mounting {datasets}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(build(sys.argv[1:] or [
        "bhpepper/biohub-synthetic-5fold-ensemble-v1",
        "hitoshisaito/biohub-synthetic16-cc0-subset",
        "giorgosi/biohub-divnet-v2",
    ]))
