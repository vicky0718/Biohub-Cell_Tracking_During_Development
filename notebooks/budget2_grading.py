md("""## 2. Grading — the budget, per embryo""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "budget2.json").read_text())
S, F, E = D["summary"], D["forks"], D["edges"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
NONE = "none"
REF_TOTAL, REF_DIVJ, REF_FORKS = 0.9188, 0.1154, 1443   # claude_divsweep's inc/g2sp6
print(f"{len(DS)} datasets | {len(ARMS)} arms")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(n) for n in DS})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(n) == e for n in DS)}" for e in EMB))

def rows(arm, names=None):
    return [PER[n][arm] for n in (names or DS) if n in PER and arm in PER[n]]

# Full-set figures come from purescore.summarise -- div_J micro-averaged, adj_edge
# weight-averaged by TP+FP+FN. Recomputing would substitute unweighted means for both
# (notes/47). Per-embryo subsets have no stored weight, so those are means, and are only
# ever used as DELTAS between two arms on the same datasets.
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

print(f"\n{'arm':<16}{'total':>9}{'adj_edge':>10}{'edge_J':>9}{'div_J':>8}"
      f"{'nodes':>10}{'ratio':>9}{'mult':>7}{'forks':>8}")
print("-" * 86)
for a in ARMS:
    r = mean(a, "ratio")
    print(f"{a:<16}{total(a):>9.4f}{adj(a):>10.4f}"
          f"{S.get(a,{}).get('edge_jaccard',float('nan')):>9.4f}{divJ(a):>8.4f}"
          f"{mean(a,'nodes'):>10,.0f}{r:>9.3f}{1 - 0.1 * r:>7.3f}{F.get(a,0):>8,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

# ---- 1. reproduction -------------------------------------------------------------
print("\n1. the `none` arm reproduces claude_divsweep's inc/g2sp6")
ok1 = (abs(total(NONE) - REF_TOTAL) < 0.002 and abs(divJ(NONE) - REF_DIVJ) < 0.010
       and abs(F.get(NONE, 0) - REF_FORKS) <= 20)
print(f"   total {total(NONE):.4f} (want {REF_TOTAL})   div_J {divJ(NONE):.4f} "
      f"(want {REF_DIVJ})   forks {F.get(NONE,0):,} (want {REF_FORKS:,})"
      f"  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   The chain has moved since claude_divsweep. Nothing below is comparable.")

# ---- 2. isolated is free ---------------------------------------------------------
print("\n2. `isolated` is non-negative on BOTH embryos (an edgeless node cannot be a TP)")
iso = "isolated"
if iso not in ARMS:
    ok2 = False; print("   NOT GRADED — arm missing")
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][iso]["adj"] - PER[n][NONE]["adj"] for n in ns
             if iso in PER.get(n, {}) and NONE in PER[n]]
        per[e] = sum(d) / len(d) if d else float("nan")
    ok2 = all(v > -0.0005 for v in per.values())
    print("   " + "   ".join(f"{e} {v:+.4f}" for e, v in per.items())
          + f"   (pooled {adj(iso) - adj(NONE):+.4f})  ->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   Dropping edgeless nodes LOSES score. That contradicts the metric as")
        print("   read in notes/45 and forum 739018, so the reading is wrong and")
        print("   predictions 3-5 cannot be interpreted.")

# ---- 3. THE CRUX -----------------------------------------------------------------
print(f"\n3. some budget arm beats `{NONE}` by more than 0.0015 (notes/44's floor)")
cand = [a for a in ARMS if a != NONE]
best = max(cand, key=lambda a: total(a) if total(a) == total(a) else -9) if cand else NONE
gain = total(best) - total(NONE)
ok3 = gain > 0.0015
print(f"   best {best} {total(best):.4f} vs {NONE} {total(NONE):.4f}"
      f"   {gain:+.4f}  ->  {'PASS' if ok3 else 'FAIL'}")
print(f"   decomposed:  adj_edge {adj(best) - adj(NONE):+.4f}"
      f"   0.1*div_J {0.1 * (divJ(best) - divJ(NONE)):+.4f}"
      f"   nodes {mean(best,'nodes') - mean(NONE,'nodes'):+,.0f}")
if not ok3:
    print("   Track-level ranking under a per-dataset cap is the THIRD selection rule,")
    print("   after uniform thinning (notes/46) and confidence thinning (notes/48,49).")
    print("   All three now fail. The node budget is not collectable by any rule we")
    print("   have, and the remaining gap is detection (notes/51: fn_detect 583).")

# ---- 4. does tightness carry anything? -------------------------------------------
print("\n4. `geometry` beats `length` at the same budget factor")
pairs = [(g, l) for g in ARMS if g.startswith("geometry")
         for l in ARMS if l == g.replace("geometry", "length")]
if not pairs:
    ok4 = False; print("   NOT GRADED — no matched pair")
else:
    wins = sum(total(g) > total(l) for g, l in pairs)
    for g, l in pairs:
        print(f"   {g:<16}{total(g):>9.4f}   vs {l:<16}{total(l):>9.4f}"
              f"   {total(g)-total(l):+.4f}")
    ok4 = wins > len(pairs) / 2
    print(f"   geometry wins {wins}/{len(pairs)}  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   Tightness carries nothing; the rule is just 'keep long tracks'.")

# ---- 5. notes/49's rule ----------------------------------------------------------
print(f"\n5. the best arm holds in sign on BOTH embryos (n is {len(EMB)}, not {len(DS)})")
# An arm tied with the control has nothing to transfer, and reporting that as "wins on one
# embryo, loses on the other" borrows notes/49's language for a run that had no effect at
# all. claude_budget2 hit exactly this: `isolated` was byte-identical to `none`, every
# per-embryo delta was +0.0000, and `all(x > 0)` rejected the zeros as a sign disagreement.
if best == NONE or abs(total(best) - total(NONE)) < 1e-9:
    ok5 = False
    print("   NOT GRADED — no arm differs from the control"
          + ("" if best == NONE else f" ({best} is tied with it)"))
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][best]["adj"] - PER[n][NONE]["adj"] for n in ns
             if best in PER.get(n, {}) and NONE in PER[n]]
        per[e] = (sum(d) / len(d) if d else float("nan"),
                  divJ(best, ns) - divJ(NONE, ns), len(d))
    print(f"   {'embryo':<8}{'n':>4}{'adj delta':>12}{'div_J delta':>14}{'total':>10}")
    for e, (da, dj, n) in per.items():
        print(f"   {e:<8}{n:>4}{da:>+12.4f}{dj:>+14.4f}{da + 0.1 * dj:>+10.4f}")
    tots = [da + 0.1 * dj for da, dj, _ in per.values()]
    ok5 = len(tots) > 1 and (all(x > 0 for x in tots) or all(x < 0 for x in tots))
    print(f"   signs agree  ->  {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        print("   Wins on one embryo, loses on the other. The test set is a THIRD pair")
        print("   (notes/07 §3). This is the shape that cost 0.901 -> 0.863 in notes/49.")

print("\n" + "=" * 78)
print(f"{sum([ok1, ok2, ok3, ok4, ok5])}/5 predictions passed")
if ok1 and ok3 and ok5:
    print(f"SUBMITTABLE: {best} gains {gain:+.4f} and holds on both embryos.")
elif ok1 and ok2 and not ok3:
    print("CLOSED: all three selection rules fail. The budget is not collectable and")
    print("the remaining gap is detection (notes/51).")
elif ok1 and ok3 and not ok5:
    print("MEASURED, NOT SUBMITTABLE: real on train, does not agree across embryos.")
else:
    print("NOT COMPARABLE: reproduction or the metric reading failed; fix that first.")
""")
