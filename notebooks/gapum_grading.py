md("""## 2. Grading — does a wider gap radius pay?""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "gapum.json").read_text())
S, F, E = D["summary"], D["forks"], D["edges"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
BASE = "g5.75"                       # the shipped radius
REF_TOTAL, REF_DIVJ, REF_FORKS = 0.9188, 0.1154, 1443   # claude_divsweep's inc/g2sp6
print(f"{len(DS)} datasets | {len(ARMS)} arms")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(n) for n in DS})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(n) == e for n in DS)}" for e in EMB))

def rows(arm, names=None):
    return [PER[n][arm] for n in (names or DS) if n in PER and arm in PER[n]]

# Full-set figures come from purescore.summarise -- div_J micro-averaged, adj_edge
# weight-averaged. Recomputing would substitute unweighted means (notes/47).
def divJ(arm, names=None):
    if names is None:
        return S.get(arm, {}).get("division_jaccard", float("nan"))
    r = rows(arm, names)
    tp, fp, fn = (sum(x[k] for x in r) for k in ("dtp", "dfp", "dfn"))
    return tp / (fp + tp + fn) if (fp + tp + fn) > 0 else float("nan")

def adj(arm, names=None):
    if names is None:
        return S.get(arm, {}).get("adj_edge_jaccard", float("nan"))
    v = [x["adj"] for x in rows(arm, names) if x["adj"] == x["adj"]]
    return sum(v) / len(v) if v else float("nan")

def total(arm, names=None):
    if names is None and arm in S and S[arm].get("score") == S[arm].get("score"):
        return S[arm]["score"]
    return adj(arm, names) + 0.1 * divJ(arm, names)

def mean(arm, key, names=None):
    v = [x[key] for x in rows(arm, names) if x.get(key) == x.get(key)]
    return sum(v) / len(v) if v else float("nan")

print(f"\n{'arm':<10}{'total':>9}{'adj_edge':>10}{'edge_J':>9}{'div_J':>8}"
      f"{'nodes':>10}{'ratio':>9}{'added':>9}{'forks':>8}")
print("-" * 82)
b_nodes = mean(BASE, "nodes")
for a in ARMS:
    print(f"{a:<10}{total(a):>9.4f}{adj(a):>10.4f}"
          f"{S.get(a,{}).get('edge_jaccard',float('nan')):>9.4f}{divJ(a):>8.4f}"
          f"{mean(a,'nodes'):>10,.0f}{mean(a,'ratio'):>9.3f}"
          f"{mean(a,'nodes')-b_nodes:>+9,.0f}{F.get(a,0):>8,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

print(f"\n1. the shipped radius reproduces claude_divsweep's inc/g2sp6")
ok1 = (abs(total(BASE) - REF_TOTAL) < 0.002 and abs(divJ(BASE) - REF_DIVJ) < 0.010
       and abs(F.get(BASE, 0) - REF_FORKS) <= 20)
print(f"   {BASE}: total {total(BASE):.4f} (want {REF_TOTAL})  div_J {divJ(BASE):.4f} "
      f"(want {REF_DIVJ})  forks {F.get(BASE,0):,} (want {REF_FORKS:,})"
      f"  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   The chain has moved. Nothing below is comparable.")

print("\n2. a wider radius actually inserts more nodes (the cap does not bind first)")
adds = [(a, mean(a, "nodes") - b_nodes) for a in ARMS if a != BASE]
for a, d in adds:
    print(f"   {a:<10}{d:>+10,.0f} nodes vs {BASE}")
ok2 = any(d > 50 for _, d in adds)
print(f"   ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   close_gaps also carries max_added_frac=0.038 and max_added_abs=1650.")
    print("   Those bind before the radius does, so max_um was never the constraint and")
    print("   notes/59's geometric argument, upper bound or not, is moot.")

print(f"\n3. some radius beats {BASE} by more than 0.0015 (notes/44's floor)")
cand = [a for a in ARMS if a != BASE]
best = max(cand, key=lambda a: total(a) if total(a) == total(a) else -9) if cand else BASE
gain = total(best) - total(BASE)
ok3 = gain > 0.0015
print(f"   best {best} {total(best):.4f} vs {BASE} {total(BASE):.4f}   {gain:+.4f}"
      f"  ->  {'PASS' if ok3 else 'FAIL'}")
print(f"   decomposed:  adj_edge {adj(best)-adj(BASE):+.4f}"
      f"   0.1*div_J {0.1*(divJ(best)-divJ(BASE)):+.4f}"
      f"   nodes {mean(best,'nodes')-b_nodes:+,.0f}")
if not ok3:
    print("   notes/59 measured 5.75um rejecting up to 23.4% of real two-frame spans, but")
    print("   that was 2x a single-frame step -- an UPPER bound assuming straight-line")
    print("   motion. Read directly, the radius is not costing score, and the geometric")
    print("   argument does not survive contact with the metric.")

print(f"\n4. the best arm holds in sign on BOTH embryos (n is {len(EMB)}, not {len(DS)})")
if best == BASE or abs(total(best) - total(BASE)) < 1e-9:
    ok4 = False
    print("   NOT GRADED — no arm differs from the shipped radius")
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][best]["adj"] - PER[n][BASE]["adj"] for n in ns
             if best in PER.get(n, {}) and BASE in PER[n]]
        per[e] = (sum(d) / len(d) if d else float("nan"),
                  divJ(best, ns) - divJ(BASE, ns), len(d))
    print(f"   {'embryo':<8}{'n':>4}{'adj delta':>12}{'div_J delta':>14}{'total':>10}")
    for e, (da, dj, n) in per.items():
        print(f"   {e:<8}{n:>4}{da:>+12.4f}{dj:>+14.4f}{da + 0.1 * dj:>+10.4f}")
    t = [da + 0.1 * dj for da, dj, _ in per.values()]
    ok4 = len(t) > 1 and (all(x > 0 for x in t) or all(x < 0 for x in t))
    print(f"   signs agree  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   notes/49: the test set is a THIRD pair of embryos.")

print("\n" + "=" * 78)
print(f"{sum([ok1, ok2, ok3, ok4])}/4 predictions passed")
if ok1 and ok3 and ok4:
    print(f"SUBMITTABLE: close_gaps max_um {BASE[1:]} -> {best[1:]} gains {gain:+.4f}.")
elif ok1 and ok2 and not ok3:
    print("CLOSED: the radius is not the constraint. notes/59's 23.4% was an upper bound")
    print("from a straight-line assumption; measured directly it buys nothing.")
elif ok1 and not ok2:
    print("CLOSED: max_added_frac/max_added_abs bind before the radius does.")
else:
    print("NOT COMPARABLE: reproduction failed; fix that first.")
print("=" * 78)
""")
