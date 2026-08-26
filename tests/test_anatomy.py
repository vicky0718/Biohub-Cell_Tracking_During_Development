"""Tests for `pipeline/anatomy.py`.

The whole point of this module is to choose the next several days of work — gap closing
if `fn_gap` dominates, motion relink if `fn_mislink` does. So the tests build graphs where
the answer is **constructed rather than inferred**: a known gap, a known mislink of each
kind, and a known missed detection, each in its own track, well separated so node matching
cannot be ambiguous.

Two structural checks matter as much as the classifications. The buckets must sum to the
GT edge count, and `tp` must equal what `purescore.count_edges` independently reports —
the module is only useful if it decomposes *the metric*, not a lookalike.

    python tests/test_anatomy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.purescore import count_edges  # noqa: E402
from pipeline.anatomy import BUCKETS, edge_anatomy, summarise_anatomy  # noqa: E402

ISO = (1.0, 1.0, 1.0)
LANE = 30.0        # µm between tracks — far outside the 7 µm match radius
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class Graph:
    """Small builder so each test case reads as the situation it is testing."""

    def __init__(self):
        self.t, self.zyx, self.edges = [], [], []

    def node(self, t, lane, off=0.0):
        self.t.append(int(t))
        self.zyx.append([0.0, lane * LANE + off, 0.0])
        return len(self.t) - 1

    def line(self, lane, frames, off=0.0, link=True):
        ids = [self.node(f, lane, off) for f in frames]
        if link:
            self.edges += list(zip(ids[:-1], ids[1:]))
        return ids

    def link(self, a, b):
        self.edges.append((a, b))

    def out(self):
        return (np.asarray(self.t, np.int64), np.asarray(self.zyx, float),
                np.asarray(self.edges, np.int64).reshape(-1, 2))


def main() -> int:
    print("=" * 66)
    print("edge_anatomy — each bucket, built so the answer is known")
    print("=" * 66)

    # ---- ground truth: 5 lanes, 4 frames each, fully linked -------------------
    gt = Graph()
    for lane in range(5):
        gt.line(lane, range(4))
    gt_t, gt_zyx, gt_edges = gt.out()
    n_gt = len(gt_edges)                       # 5 lanes x 3 edges = 15

    # ---- a perfect prediction -------------------------------------------------
    a = edge_anatomy(gt_t, gt_zyx, gt_edges, gt_t, gt_zyx, gt_edges, scale=ISO)
    check("a perfect prediction is all TP",
          a["tp"] == n_gt and a["fn_gap"] == 0 and a["fn_mislink"] == 0
          and a["fn_detect"] == 0,
          f"tp={a['tp']}/{n_gt}, gap={a['fn_gap']}, mis={a['fn_mislink']}, "
          f"det={a['fn_detect']}")
    check("reachable is 0 when nothing is missing", a["reachable"] == 0.0)

    # ---- one of each failure, one per lane ------------------------------------
    p = Graph()
    p.line(0, range(4))                                   # lane 0: perfect -> 3 TP
    l1 = p.line(1, range(4), link=False)                  # lane 1: nodes, no edges
    p.link(l1[0], l1[1]); p.link(l1[2], l1[3])            #   -> hole at 1->2 = 1 gap
    l2 = p.line(2, range(4), link=False)                  # lane 2: source committed
    p.link(l2[0], l2[1]); p.link(l2[2], l2[3])
    decoy = p.node(2, 2, off=15.0)                        #   a node 15 µm off-lane,
    p.link(l2[1], decoy)                                  #   so it matches no GT node
    l3 = p.line(3, range(4), link=False)                  # lane 3: target committed
    p.link(l3[0], l3[1]); p.link(l3[2], l3[3])
    thief = p.node(1, 3, off=15.0)
    p.link(thief, l3[2])                                  #   -> l3[2] already has a parent
    p.line(4, [0, 1, 3])                                  # lane 4: frame 2 never detected
    p_t, p_zyx, p_edges = p.out()

    a = edge_anatomy(p_t, p_zyx, p_edges, gt_t, gt_zyx, gt_edges, scale=ISO)
    for k in BUCKETS:
        print(f"      {k:<14} {a[k]}")
    print(f"      {'source_busy':<14} {a['source_busy']}")
    print(f"      {'target_busy':<14} {a['target_busy']}")

    check("the constructed gap lands in fn_gap", a["fn_gap"] == 1,
          f"fn_gap={a['fn_gap']}, wanted 1 (lane 1, both ends free and unlinked)")
    check("a committed SOURCE lands in fn_mislink", a["source_busy"] == 1,
          f"source_busy={a['source_busy']}, wanted 1 (lane 2 links to a decoy)")
    check("a committed TARGET lands in fn_mislink", a["target_busy"] == 1,
          f"target_busy={a['target_busy']}, wanted 1 (lane 3's child has another parent)")
    check("both mislink kinds are counted once each", a["fn_mislink"] == 2,
          f"fn_mislink={a['fn_mislink']}, wanted 2")
    check("a missing detection lands in fn_detect", a["fn_detect"] == 2,
          f"fn_detect={a['fn_detect']}, wanted 2 (lane 4 lost frame 2, so 1->2 and 2->3)")
    # Each broken lane loses only its MIDDLE edge; the two flanking edges still score.
    # lane 0 perfect (3) + gap lane (2) + source-busy (2) + target-busy (2) + lost
    # frame 2, which kills both edges touching it (1).
    check("only the broken edge in each lane is lost", a["tp"] == 3 + 2 + 2 + 2 + 1,
          f"tp={a['tp']}, wanted 10")

    # ---- the two structural checks -------------------------------------------
    total = sum(a[k] for k in BUCKETS)
    check("the buckets sum to the GT edge count", total == n_gt,
          f"{total} vs {n_gt} — a leak here means an edge was classified twice or not "
          "at all, and every fraction downstream is wrong")

    counts = count_edges(p_t, p_zyx, p_edges, gt_t, gt_zyx, gt_edges, scale=ISO)
    check("tp agrees with the metric's own count", a["tp"] == counts.tp,
          f"anatomy {a['tp']} vs count_edges {counts.tp}")
    check("the FN buckets agree with the metric's FN",
          n_gt - a["tp"] == counts.fn,
          f"anatomy {n_gt - a['tp']} vs count_edges {counts.fn}")
    check("edge_fp is carried through for reference", a["edge_fp"] == counts.fp,
          f"{a['edge_fp']} vs {counts.fp}")

    # ---- non-consecutive GT edges --------------------------------------------
    g2 = Graph()
    ids = g2.line(0, [0, 1, 3])         # the 1->3 edge cannot be scored by anyone
    g2_t, g2_zyx, g2_edges = g2.out()
    a = edge_anatomy(g2_t, g2_zyx, g2_edges, g2_t, g2_zyx, g2_edges, scale=ISO)
    check("a non-consecutive GT edge is quarantined, not blamed on the model",
          a["fn_nonconsec"] == 1 and a["tp"] == 1,
          f"nonconsec={a['fn_nonconsec']}, tp={a['tp']} of {len(g2_edges)}")
    check("buckets still sum with a non-consecutive edge present",
          sum(a[k] for k in BUCKETS) == len(g2_edges))

    # ---- the scorer's silent repairs must be respected ------------------------
    # A 3-fork: the scorer truncates out-degree to the two lowest edge ids, so the third
    # link is not a link the metric ever saw. Classifying against raw edges would call
    # the truncated GT edge a TP.
    g3 = Graph()
    root = g3.node(0, 0)
    kids = [g3.node(1, 0, off=o) for o in (0.0, 1.0, 2.0)]
    for k in kids:
        g3.link(root, k)
    g3_t, g3_zyx, g3_edges = g3.out()
    a = edge_anatomy(g3_t, g3_zyx, g3_edges, g3_t, g3_zyx, g3_edges, scale=ISO)
    c = count_edges(g3_t, g3_zyx, g3_edges, g3_t, g3_zyx, g3_edges, scale=ISO)
    check("out-degree truncation is respected, not counted as linked",
          a["tp"] == c.tp,
          f"anatomy {a['tp']} vs metric {c.tp} of 3 GT edges — the third fork branch is "
          "truncated by the scorer and must not be scored as a TP here either")

    # ---- degenerate inputs ----------------------------------------------------
    e0 = np.zeros((0, 2), np.int64)
    a = edge_anatomy(gt_t, gt_zyx, e0, gt_t, gt_zyx, gt_edges, scale=ISO)
    check("a prediction with no edges is all gap, not all detect",
          a["fn_gap"] == n_gt and a["tp"] == 0,
          f"gap={a['fn_gap']}, detect={a['fn_detect']} — every node matched, nothing linked")
    a = edge_anatomy(np.zeros(0, np.int64), np.zeros((0, 3)), e0,
                     gt_t, gt_zyx, gt_edges, scale=ISO)
    check("an empty prediction is all fn_detect",
          a["fn_detect"] == n_gt, f"detect={a['fn_detect']}")
    a = edge_anatomy(gt_t, gt_zyx, gt_edges, gt_t, gt_zyx, e0, scale=ISO)
    check("empty ground truth returns cleanly", a["n_gt_edges"] == 0)

    # ---- pooling --------------------------------------------------------------
    print()
    print("=" * 66)
    print("summarise_anatomy")
    print("=" * 66)
    rows = [edge_anatomy(p_t, p_zyx, p_edges, gt_t, gt_zyx, gt_edges, scale=ISO),
            edge_anatomy(gt_t, gt_zyx, gt_edges, gt_t, gt_zyx, gt_edges, scale=ISO)]
    s = summarise_anatomy(rows)
    check("pooling sums counts rather than averaging fractions",
          s["n_gt_edges"] == 2 * n_gt and s["tp"] == rows[0]["tp"] + rows[1]["tp"],
          f"n_gt_edges={s['n_gt_edges']}, tp={s['tp']}")
    check("the pooled buckets still sum", sum(s[k] for k in BUCKETS) == s["n_gt_edges"])
    check("pooled edge_jaccard matches the metric's micro-average",
          abs(s["edge_jaccard"]
              - s["edge_tp"] / (s["edge_tp"] + s["edge_fp"] + s["edge_fn"])) < 1e-12)

    print()
    print("=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all anatomy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
