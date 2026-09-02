md("""## 2. Grading — `div_J` against the chain we ship, per embryo""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "divsweep.json").read_text())
S, F, E = D["summary"], D["forks"], D["edges"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
POST = D["post"]; SOLVES = [g["label"] for g in D["grid"]]
REF, SHIP = "inc/g1s", "inc/g2sp6"     # notes/36's chain, and the one in the submission
NOTES36_DIVJ = 0.1154                  # what REF must reproduce
print(f"{len(DS)} datasets | {len(SOLVES)} solves x {len(POST)} post-chains = {len(ARMS)} arms")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(n) for n in DS})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(n) == e for n in DS)}" for e in EMB))

def rows(arm, names=None):
    return [PER[n][arm] for n in (names or DS) if n in PER and arm in PER[n]]

# On the FULL set every figure comes from purescore.summarise, which is the metric:
# div_J micro-averaged (counts pooled, then divided) and adj_edge_jaccard weight-averaged
# by TP+FP+FN. Recomputing here would silently substitute unweighted means for both --
# printing one next to the other is the aggregation mismatch notes/47 §2 was about.
# For an embryo SUBSET there is no per-dataset weight in the record, so those columns are
# pooled counts for div_J and an unweighted mean for adj, and are labelled as deltas only.
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

print(f"\n{'arm':<14}{'total':>9}{'adj_edge':>10}{'div_J':>9}{'0.1divJ':>9}"
      f"{'forks':>8}{'edges':>10}")
print("-" * 69)
for a in ARMS:
    print(f"{a:<14}{total(a):>9.4f}{adj(a):>10.4f}{divJ(a):>9.4f}"
          f"{0.1 * divJ(a):>9.4f}{F.get(a, 0):>8,}{E.get(a, 0):>10,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

# ---- 1. reproduction -------------------------------------------------------------
print("\n1. ctl has a dead division term, and inc/g1s reproduces notes/36's div_J 0.1154")
c, r = divJ("ctl/g1s"), divJ(REF)
ok1 = c < 0.02 and abs(r - NOTES36_DIVJ) < 0.010
print(f"   ctl/g1s div_J {c:.4f} (want <0.02)   {REF} div_J {r:.4f} "
      f"(want {NOTES36_DIVJ:.4f} +-0.010)  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   The cache or the solver has moved since notes/36. Nothing below is")
    print("   comparable to the record, and that is the finding.")

# ---- 2. THE CRUX -----------------------------------------------------------------
print(f"\n2. the shipped chain loses >0.020 of div_J vs {REF}, on IDENTICAL datasets")
drop = divJ(REF) - divJ(SHIP)
ok2 = drop > 0.020
print(f"   {REF} {divJ(REF):.4f}  ->  {SHIP} {divJ(SHIP):.4f}   drop {drop:+.4f}"
      f"  ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   notes/42 read div_J 0.0645 at n=12 against notes/36's 0.1154 at n=24 and")
    print("   attributed the gap to config. On one dataset sample the gap is not there:")
    print("   it was the sampling artifact notes/44 predicted (the 12 were an easy")
    print("   subset by +0.0116). The division term is CLOSED, notes/36 §concluded")
    print("   correctly, and the whole remaining gap is the edge term.")

# ---- 3. attribution --------------------------------------------------------------
print("\n3. one stage dominates the drop (>half), rather than three sharing it")
# each pair differs by exactly one stage
STAGES = [("max_gap 1->2", "inc/g1s", "inc/g2s"),
          ("+prune(6)", "inc/g2s", "inc/g2sp6"),
          ("+linefit_smooth", "inc/g2p6", "inc/g2sp6"),
          ("keep_div_comp OFF", "inc/g2sp6", "inc/g2sp6_nk")]
print(f"   {'stage':<20}{'div_J before':>13}{'after':>9}{'delta':>9}{'forks lost':>12}")
deltas = []
for lbl, a, b in STAGES:
    if a not in ARMS or b not in ARMS:
        continue
    d = divJ(b) - divJ(a)
    deltas.append((abs(d), lbl))
    print(f"   {lbl:<20}{divJ(a):>13.4f}{divJ(b):>9.4f}{d:>+9.4f}"
          f"{F.get(a, 0) - F.get(b, 0):>12,}")
if deltas and drop > 1e-9:
    worst, who = max(deltas)
    ok3 = worst > drop / 2
    print(f"   largest single stage: {who} at {worst:.4f}, total drop {drop:.4f}"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   No single stage owns it; the chain degrades div_J cumulatively and")
        print("   there is no one knob to turn.")
else:
    ok3 = False
    print("   NOT GRADED — no drop to attribute")

# ---- 4. is the recovery free? ----------------------------------------------------
print(f"\n4. some arm beats {SHIP} on TOTAL, not just on div_J")
best = max(ARMS, key=lambda a: total(a) if total(a) == total(a) else -9)
gain = total(best) - total(SHIP)
ok4 = best != SHIP and gain > 0.0015      # notes/44's measurable floor
print(f"   best arm {best} {total(best):.4f} vs {SHIP} {total(SHIP):.4f}"
      f"   {gain:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
print(f"   decomposed:  adj_edge {adj(best) - adj(SHIP):+.4f}"
      f"   0.1*div_J {0.1 * (divJ(best) - divJ(SHIP)):+.4f}")
if not ok4:
    print("   notes/36's trade holds: every fork recovered costs more edge score than")
    print("   the division term pays back. div_J is not free, and it is not the lever.")

# ---- 5. notes/49's rule ----------------------------------------------------------
print(f"\n5. the best arm holds on BOTH embryos (notes/49 -- n is 2, not {len(DS)})")
if best == SHIP:
    ok5 = False
    print("   NOT GRADED — the shipped chain is already best, nothing to transfer")
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][best]["adj"] - PER[n][SHIP]["adj"]
             for n in ns if best in PER.get(n, {}) and SHIP in PER[n]]
        dj = divJ(best, ns) - divJ(SHIP, ns)
        per[e] = (sum(d) / len(d) if d else float("nan"), dj, len(d))
    print(f"   {'embryo':<8}{'n':>4}{'adj delta':>12}{'div_J delta':>14}{'total':>10}")
    for e, (da, dj, n) in per.items():
        print(f"   {e:<8}{n:>4}{da:>+12.4f}{dj:>+14.4f}{da + 0.1 * dj:>+10.4f}")
    tots = [da + 0.1 * dj for da, dj, _ in per.values()]
    ok5 = len(tots) > 1 and (all(t > 0 for t in tots) or all(t < 0 for t in tots))
    print(f"   signs agree across embryos  ->  {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        print("   The arm wins on one embryo and loses on the other. The test set is a")
        print("   THIRD pair of embryos (notes/07 §3), so this does not transfer --")
        print("   this is exactly the shape that cost 0.901 -> 0.863 in notes/49.")

print("\n" + "=" * 78)
n_ok = sum([ok1, ok2, ok3, ok4, ok5])
print(f"{n_ok}/5 predictions passed")
if ok1 and ok2 and ok4 and ok5:
    print(f"SUBMITTABLE: {best} gains {gain:+.4f} and holds on both embryos.")
elif ok1 and not ok2:
    print("CLOSED: the div_J drop was a sampling artifact. The division term is done,")
    print("and notes/44's shortlist is now empty of cheap items.")
elif ok1 and ok2 and not ok4:
    print("PRICED, NOT SUBMITTABLE: the drop is real and attributable, but recovering it")
    print("costs more edge score than it returns. Records the trade; does not change the run.")
else:
    print("NOT COMPARABLE: reproduction failed; fix the cache before reading anything else.")
""")
