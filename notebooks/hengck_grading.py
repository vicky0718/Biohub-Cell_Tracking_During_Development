md("""## 2. Grading — do his two suggestions pay?""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "hengck.json").read_text())
S, F, E = D["summary"], D["forks"], D["edges"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
BASE = "base"
REF_TOTAL, REF_DIVJ, REF_FORKS = 0.9188, 0.1154, 1443   # claude_divsweep's inc/g2sp6
CAPS = [a for a in ARMS if a.startswith("cap")]
DIVS = [a for a in ARMS if a.startswith("div")]
print(f"{len(DS)} datasets | {len(ARMS)} arms  ({len(CAPS)} cap, {len(DIVS)} div)")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(n) for n in DS})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(n) == e for n in DS)}" for e in EMB))

def rows(arm, names=None):
    return [PER[n][arm] for n in (names or DS) if n in PER and arm in PER[n]]

# Full-set figures come from purescore.summarise: div_J micro-averaged, adj_edge
# weight-averaged. Recomputing would substitute unweighted means for both (notes/47).
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

b_e = E.get(BASE, 0)
print(f"\n{'arm':<10}{'total':>9}{'adj_edge':>10}{'edge_J':>9}{'div_J':>8}"
      f"{'dTP':>7}{'dFP':>7}{'forks':>8}{'edges':>10}{'d_edges':>9}")
print("-" * 87)
for a in ARMS:
    print(f"{a:<10}{total(a):>9.4f}{adj(a):>10.4f}"
          f"{S.get(a,{}).get('edge_jaccard',float('nan')):>9.4f}{divJ(a):>8.4f}"
          f"{sum(x['dtp'] for x in rows(a)):>7.0f}{sum(x['dfp'] for x in rows(a)):>7.0f}"
          f"{F.get(a,0):>8,}{E.get(a,0):>10,}{E.get(a,0)-b_e:>+9,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

print("\n1. `base` reproduces claude_divsweep's inc/g2sp6")
ok1 = (abs(total(BASE) - REF_TOTAL) < 0.002 and abs(divJ(BASE) - REF_DIVJ) < 0.010
       and abs(F.get(BASE, 0) - REF_FORKS) <= 20)
print(f"   total {total(BASE):.4f} (want {REF_TOTAL})  div_J {divJ(BASE):.4f} "
      f"(want {REF_DIVJ})  forks {F.get(BASE,0):,} (want {REF_FORKS:,})"
      f"  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   Nothing below is comparable.")

print("\n2. SUGGESTION 1 — long links are worth filtering (>0.0015 over base)")
best_cap = max(CAPS, key=lambda a: total(a) if total(a) == total(a) else -9) if CAPS else BASE
g_cap = total(best_cap) - total(BASE)
ok2 = bool(CAPS) and g_cap > 0.0015
for a in CAPS:
    print(f"   {a:<10}{total(a):>9.4f}{total(a)-total(BASE):>+9.4f}"
          f"   edges {E.get(a,0)-b_e:+,}")
print(f"   best {best_cap} {g_cap:+.4f}  ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   hengck23 saw long links as almost always wrong on the HOST BASELINE's raw")
    print("   linker output. Our chain already runs an ILP that pays a per-edge cost, so")
    print("   the long-and-wrong edges he was filtering may simply not survive our solve.")

print("\n3. MECHANISM — the broken gates were why insertion looked dead (notes/57)")
old = next((a for a in DIVS if a.endswith("old")), None)
p90 = next((a for a in DIVS if a.endswith("p90")), None)
if old and p90:
    f_old, f_p90 = F.get(old, 0) - F.get(BASE, 0), F.get(p90, 0) - F.get(BASE, 0)
    ok3 = f_p90 > 0 and f_old < 0.5 * f_p90
    print(f"   forks inserted:  {old} +{f_old:,}   {p90} +{f_p90:,}"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    print(f"   notes/57: max_um 4.5 rejects 88.7% and sister_max_um 6.8 rejects 86.8%")
    print(f"   of the 151 real GT divisions. This is that measurement, seen as forks.")
else:
    ok3 = False; print("   NOT GRADED — arms missing")

print("\n4. SUGGESTION 2 — corrected gates raise div_J above base")
best_div = max(DIVS, key=lambda a: divJ(a) if divJ(a) == divJ(a) else -9) if DIVS else BASE
d_div = divJ(best_div) - divJ(BASE)
ok4 = bool(DIVS) and d_div > 0.005
for a in DIVS:
    print(f"   {a:<10} div_J {divJ(a):.4f}{divJ(a)-divJ(BASE):>+9.4f}"
          f"   total {total(a):.4f}{total(a)-total(BASE):>+9.4f}   forks +{F.get(a,0)-F.get(BASE,0):,}")
print(f"   best {best_div} div_J {d_div:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
if ok4:
    print(f"   worth {0.1*d_div:+.4f} on the score; total moves {total(best_div)-total(BASE):+.4f}")
    print("   (a fork that adds div_J can still cost edge_J -- read the total, not div_J)")
else:
    print("   Insertion does not pay even with the gates set from ground truth. notes/25's")
    print("   probe is answered: the direction is closed on its merits, not on a bad gate.")

print(f"\n5. the best arm overall holds in sign on BOTH embryos (n is {len(EMB)}, not {len(DS)})")
cand = [a for a in ARMS if a != BASE]
best = max(cand, key=lambda a: total(a) if total(a) == total(a) else -9) if cand else BASE
if best == BASE or abs(total(best) - total(BASE)) < 1e-9:
    ok5 = False
    print("   NOT GRADED — no arm differs from base")
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][best]["adj"] - PER[n][BASE]["adj"] for n in ns
             if best in PER.get(n, {}) and BASE in PER[n]]
        per[e] = (sum(d) / len(d) if d else float("nan"),
                  divJ(best, ns) - divJ(BASE, ns), len(d))
    print(f"   best overall: {best}")
    print(f"   {'embryo':<8}{'n':>4}{'adj delta':>12}{'div_J delta':>14}{'total':>10}")
    for e, (da, dj, n) in per.items():
        print(f"   {e:<8}{n:>4}{da:>+12.4f}{dj:>+14.4f}{da + 0.1 * dj:>+10.4f}")
    t = [da + 0.1 * dj for da, dj, _ in per.values()]
    ok5 = len(t) > 1 and (all(x > 0 for x in t) or all(x < 0 for x in t))
    print(f"   signs agree  ->  {'PASS' if ok5 else 'FAIL'}")

print("\n" + "=" * 78)
print(f"{sum([ok1, ok2, ok3, ok4, ok5])}/5 predictions passed")
if ok1 and (ok2 or ok4) and ok5:
    print(f"SUBMITTABLE: {best} gains {total(best)-total(BASE):+.4f} and holds on both embryos.")
elif ok1 and ok3 and not ok4:
    print("CLOSED ON MERIT: notes/57's gate error was real (prediction 3), and fixing it")
    print("still does not make insertion pay. notes/25's unrun probe is now answered.")
elif ok1 and not ok2 and not ok4:
    print("BOTH SUGGESTIONS CLOSED on our chain. His observations were made on the host")
    print("baseline's raw output; our ILP + repair chain has already removed what he saw.")
else:
    print("SEE ABOVE — read prediction 1 before anything else.")
print("=" * 78)
""")
