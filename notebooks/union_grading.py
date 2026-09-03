md("""## 2. Grading — is there anything to ensemble?""")

code(r"""
import json, numpy as np
D = json.loads((WORK / "union.json").read_text())
R = D["rows"]
print(f"{len(R)} datasets")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(r["name"]) for r in R})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(r['name']) == e for r in R)}" for e in EMB))

# Pooled over GT nodes, not a mean of per-dataset recalls: the denominators span two
# orders of magnitude and averaging ratios across them is notes/47 §2's error.
def pooled(num, den="n_gt", rows=None):
    rows = rows if rows is not None else R
    d = sum(r[den] for r in rows)
    return sum(r[num] for r in rows) / d if d else float("nan")

tot_gt = sum(r["n_gt"] for r in R)
pk, sp = pooled("n_pack_matched"), pooled("n_spot_matched")
un, se = pooled("n_union_matched"), pooled("n_sel_matched")
resc, resc_s = sum(r["rescued"] for r in R), sum(r["rescued_sel"] for r in R)
add_f, add_s = sum(r["n_added_full"] for r in R), sum(r["n_added_sel"] for r in R)
n_pack = sum(r["n_pack"] for r in R)

print(f"\n{'set':<14}{'GT matched':>12}{'recall':>9}{'vs pack':>10}{'nodes added':>13}")
print("-" * 58)
print(f"{'pack':<14}{sum(r['n_pack_matched'] for r in R):>12,}{pk:>9.4f}{'':>10}{'':>13}")
print(f"{'spotiflow':<14}{sum(r['n_spot_matched'] for r in R):>12,}{sp:>9.4f}{'':>10}{'':>13}")
print(f"{'union':<14}{sum(r['n_union_matched'] for r in R):>12,}{un:>9.4f}"
      f"{un - pk:>+10.4f}{add_f:>13,}")
print(f"{'selective':<14}{sum(r['n_sel_matched'] for r in R):>12,}{se:>9.4f}"
      f"{se - pk:>+10.4f}{add_s:>13,}")
print(f"\nGT nodes: {tot_gt:,}   pack detections: {n_pack:,}")
print(f"rescued by union {resc:,}   by selective {resc_s:,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

print("\n1. reproduction: pack 0.996 and spotiflow 0.547 (notes/47)")
ok1 = abs(pk - 0.996) < 0.005 and abs(sp - 0.547) < 0.02
print(f"   pack {pk:.4f} (want 0.996 +-0.005)   spotiflow {sp:.4f} (want 0.547 +-0.02)"
      f"  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   Not the same measurement as notes/47. Nothing below compares to the record.")

print("\n2. THE CRUX — the union rescues GT nodes the pack misses (>0.002 recall)")
ok2 = (un - pk) > 0.002
print(f"   union {un:.4f} vs pack {pk:.4f}   {un - pk:+.4f}   ({resc:,} GT nodes)"
      f"  ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   Spotiflow's detections are effectively a SUBSET of ours. No weighting,")
    print("   veto, corrector or blend can extract what is not there, so detection-stage")
    print("   ensembling with this model is closed regardless of architecture.")

print("\n3. the rescue survives the 7um selectivity filter (>half of it)")
ok3 = resc > 0 and resc_s > 0.5 * resc
print(f"   selective rescues {resc_s:,} of {resc:,}"
      f"   ({resc_s / max(resc, 1):.1%})  ->  {'PASS' if ok3 else 'FAIL'}")
if not ok3:
    print("   The rescue only appears when EVERY spotiflow detection is added, i.e. it")
    print("   comes from re-matching within 7um rather than from finding new cells.")
    print("   That is a matching artifact, not complementarity.")

print("\n4. the rescue is affordable (<20 added nodes per rescued GT node)")
cost = add_s / max(resc_s, 1)
ok4 = resc_s > 0 and cost < 20
print(f"   selective: {add_s:,} nodes added for {resc_s:,} rescues = {cost:.1f} per rescue"
      f"  ->  {'PASS' if ok4 else 'FAIL'}")
# what it does to the budget we actually sit at (notes/52: ratio -0.129)
if n_pack:
    d_ratio = add_s / max(sum(r["n_total"] for r in R), 1)
    print(f"   node count +{add_s / n_pack:.1%}, moving ratio -0.129 -> {-0.129 + d_ratio:+.3f}"
          f"   multiplier {1 - 0.1 * (-0.129):.4f} -> {1 - 0.1 * (-0.129 + d_ratio):.4f}")
if not ok4:
    print("   The nodes cost more multiplier than the rescued edges can repay.")

print("\n5. it holds on BOTH embryos (notes/49 — the test set is a third pair)")
per = {}
for e in EMB:
    rows = [r for r in R if emb(r["name"]) == e]
    per[e] = (pooled("n_union_matched", rows=rows) - pooled("n_pack_matched", rows=rows),
              sum(r["rescued_sel"] for r in rows), len(rows))
print(f"   {'embryo':<8}{'n':>4}{'union - pack':>15}{'sel rescues':>13}")
for e, (d, rs, n) in per.items():
    print(f"   {e:<8}{n:>4}{d:>+15.4f}{rs:>13,}")
vals = [d for d, _, _ in per.values()]
ok5 = len(vals) > 1 and all(v > 0 for v in vals)
print(f"   both positive  ->  {'PASS' if ok5 else 'FAIL'}")
if not ok5:
    print("   The complementarity is embryo-specific, so it does not transfer to a third.")

print("\n" + "=" * 78)
n_ok = sum([ok1, ok2, ok3, ok4, ok5])
print(f"{n_ok}/5 predictions passed")
if not ok1:
    print("NOT COMPARABLE: reproduction failed; fix that before reading anything else.")
elif not ok2:
    print("CLOSED: nothing to ensemble. Spotiflow finds a subset of what we find, and no")
    print("architecture recovers information that is absent. notes/51's remaining")
    print("detection loss needs a better detector, not a second opinion from a worse one.")
elif ok2 and ok3 and ok4 and ok5:
    ceil = resc_s / max(tot_gt, 1)
    print(f"WORTH BUILDING: {resc_s:,} affordable rescues, {ceil:.2%} of GT nodes.")
    print(f"Ceiling from notes/51 is ~1.72% of GT edges; size the build against that.")
else:
    print("PARTIAL: there is complementarity but it fails an affordability or transfer")
    print("test. Read predictions 3-5 before designing anything.")
print("=" * 78)
""")
