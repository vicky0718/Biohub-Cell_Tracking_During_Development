print("\n2. node count RISES more than 20% from the anchor to the loosest threshold")
# ROWS is in POOL_GRID order, so ROWS[0] is the 0.975 anchor and the rest descend.
ns = [(p, n) for p, _, n, _, _ in ROWS if n]
if len(ns) < 2:
    ok2 = False
    print("   NOT GRADED — fewer than two arms produced counts")
else:
    base_n = dict(ns).get(POOLS[0], 0)
    top_n = max(n for _, n in ns)
    rise = top_n / max(base_n, 1)
    ok2 = rise > 1.20
    print(f"   anchor {base_n:,} -> max {top_n:,}   x{rise:.3f}"
          f"  ->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   notes/44 measured 5.6% over [0.965, 0.99] and called the axis flat. The")
        print("   sigmoid is just as saturated BELOW 0.965, so the budget cannot be spent")
        print("   any more than it could be collected. That closes this axis in both")
        print("   directions in one run, and the 583 undetected endpoints are not")
        print("   threshold-reachable at all.")

print("\n3. fn_detect falls more than 15% — the extra nodes recover missed endpoints")
det = [(p, A.get(cell(p, m0, g0), {}).get("fn_detect", float("nan"))) for p in POOLS]
det = [(p, d) for p, d in det if d == d]
if len(det) < 2:
    ok3 = False
    print("   NOT GRADED — anatomy unavailable")
else:
    base_d = dict(det).get(POOLS[0], float("nan"))
    lo_p, lo_d = min(det, key=lambda x: x[1])
    ok3 = base_d == base_d and base_d > 0 and (base_d - lo_d) / base_d > 0.15
    print(f"   {'det':>10}{'fn_detect':>11}{'vs anchor':>11}")
    for p, d in det:
        print(f"   {p:>10}{d:>11,.0f}{d - base_d:>+11,.0f}")
    print(f"   best {lo_p} at {lo_d:,.0f}, anchor {base_d:,.0f}, "
          f"{(base_d - lo_d) / max(base_d, 1):.1%} lower"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   Loosening does not bring the missed endpoints back. They are detector")
        print("   CAPACITY misses, not threshold misses -- notes/51's reading -- and the")
        print("   direction that addresses them is a retrained detector, not a knob.")

print("\n4. some arm beats the anchor by more than 0.0015 (notes/44's floor)")
if not EXACT:
    ok4 = False
    print("   NOT GRADED — score is NaN")
else:
    gain = S[best][key] - inc
    ok4 = best != ANCHOR and gain > 0.0015
    print(f"   best {best} = {S[best][key]:.4f} vs {ANCHOR} {inc:.4f}"
          f"   {gain:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
    r4 = paired(best, ANCHOR) if best != ANCHOR else None
    if r4:
        m, sd, se, t, n = r4
        print(f"   paired over {n} datasets: mean {m:+.4f}  sd {sd:.4f}  t {t:.2f}"
              f"   {'RESOLVED' if abs(t) > 2.0 else 'not resolved'}")
    _bs, _as_ = S[best], S[ANCHOR]
    print(f"   decomposed:  edge_J {_bs.get('edge_jaccard', float('nan')) - _as_.get('edge_jaccard', float('nan')):+.4f}"
          f"   multiplier {(_bs.get('adj_edge_jaccard', 0) / max(_bs.get('edge_jaccard', 1), 1e-9)) - (_as_.get('adj_edge_jaccard', 0) / max(_as_.get('edge_jaccard', 1), 1e-9)):+.4f}"
          f"   nodes {N.get(best, 0) - N.get(ANCHOR, 0):+,}")
    if not ok4:
        print("   notes/52's break-even is refuted: a 13% rise in node count does not")
        print("   return the ~29% of fn_detect edges needed to clear the 0.0118 the")
        print("   multiplier costs. The threshold axis is then closed in BOTH directions")
        print("   and every remaining lever is on the detector itself.")

print(f"\n5. the best arm holds in sign on BOTH embryos (notes/49 — n is 2, not {NN})")
if best == ANCHOR:
    ok5 = False
    print("   NOT GRADED — the anchor is already best, nothing to transfer")
else:
    em = by_embryo(best, ANCHOR)
    print(f"   {'embryo':<8}{'mean delta':>13}")
    for e, v in em.items():
        print(f"   {e:<8}{v:>+13.4f}")
    ok5 = len(em) > 1 and (all(v > 0 for v in em.values())
                           or all(v < 0 for v in em.values()))
    if len(em) > 1 and all(abs(v) < 1e-9 for v in em.values()):
        ok5 = False
        print("   NOT GRADED — the arm is identical to the anchor on both embryos")
    else:
        print(f"   signs agree  ->  {'PASS' if ok5 else 'FAIL'}")
        if not ok5:
            print("   Wins on one embryo and loses on the other. The test set is a THIRD")
            print("   pair (notes/07 §3), so a pooled win across crops of two says nothing")
            print("   about it. This is the shape that cost 0.901 -> 0.863 in notes/49.")

print("\n" + "=" * 92)
_oks = [v for v in (globals().get("ok1"), ok2, ok3, ok4, ok5) if v is not None]
print(f"{sum(bool(v) for v in _oks)}/{len(_oks)} predictions passed")
if ok4 and ok5:
    print(f"SUBMITTABLE: {best} gains {S[best][key] - inc:+.4f} and holds on both embryos.")
elif ok2 and ok3 and not ok4:
    print("PRICED, NOT SUBMITTABLE: loosening does recover endpoints, but not enough to")
    print("pay for the multiplier it spends. The trade is now measured in both directions.")
elif not ok2:
    print("CLOSED BOTH WAYS: the threshold cannot move node count downward either.")
    print("Detection capacity is the only remaining lever (notes/51).")
elif ok4 and not ok5:
    print("MEASURED, NOT SUBMITTABLE: real on train, does not agree across embryos.")
else:
    print("SEE ABOVE — read predictions 2 and 3 before 4; they say whether the")
    print("mechanism moved at all.")
print("=" * 92)
''')
