"""Prove `pipeline/deepcenter.py` scores the place it claims to, before a GPU is spent.

Two things can go silently wrong with a veto and both produce a plausible-looking number:

  1. **The scoring geometry.** The heatmap keeps full z resolution and is pooled by 4 in y
     and x. Divide the wrong axis, or divide all three, and every score is read from the
     wrong voxel -- the veto still "works", it just vetoes at random. So the central test
     puts a blob at a known voxel and checks the score peaks THERE and is low elsewhere,
     with an asymmetric position that would fail if z were pooled too.

  2. **The candidate filter.** `close_gaps(accept=...)` must drop rejected pairs *before*
     the greedy assignment, or a rejected pair burns an endpoint a good pair needed. The
     test builds exactly that contention: a short bad pair that outranks a longer good one
     on the same tail.

Runs on CPU with a randomly-initialised module -- it tests the plumbing and the geometry,
which is what can be tested without the published weights. The checkpoint itself is
verified in the notebook by `load_state_dict`, which is strict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline import deepcenter as dc
from pipeline.repair import close_gaps

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    print("=" * 76)
    print("scoring geometry — the thing that fails silently")
    print("=" * 76)

    # A heatmap in POOLED space: full z, y/x already divided by 4.
    nz, ny, nx, pf = 16, 32, 32, 4
    hm = np.zeros((nz, ny, nx), np.float32)
    # Deliberately asymmetric in every axis, and z far from y/x so a wrongly-pooled z
    # cannot coincidentally land on the peak.
    zt, yt, xt = 12, 5, 20
    hm[zt, yt, xt] = 1.0

    # The voxel-space point that SHOULD map to it: z as-is, y/x multiplied back up.
    pt = np.array([[zt, yt * pf, xt * pf]], float)
    s = dc.score_points(hm, pt, pf, win_z=0, win_yx=0)
    check("an exact hit scores 1.0", abs(float(s[0]) - 1.0) < 1e-6, f"{float(s[0]):.4f}")

    # If z were also divided by pool_factor this would be the point that hits instead.
    wrong = np.array([[zt * pf, yt * pf, xt * pf]], float)
    sw = dc.score_points(hm, wrong, pf, win_z=0, win_yx=0)
    check("z is NOT pooled (the z*4 point misses)", float(sw[0]) == 0.0,
          f"{float(sw[0]):.4f} — nonzero here means z is being divided as well")

    # If y/x were NOT divided, this would hit instead.
    wrong2 = np.array([[zt, yt, xt]], float)
    s2 = dc.score_points(hm, wrong2, pf, win_z=0, win_yx=0)
    check("y and x ARE pooled (the undivided point misses)", float(s2[0]) == 0.0,
          f"{float(s2[0]):.4f} — nonzero means y/x are not being divided")

    # The window is a max over a box, so a near miss inside the window still scores.
    near = np.array([[zt + 1, (yt + 2) * pf, (xt - 2) * pf]], float)
    check("a point inside the window still scores",
          float(dc.score_points(hm, near, pf)[0]) == 1.0)
    far = np.array([[zt + 3, (yt + 6) * pf, xt * pf]], float)
    check("a point outside the window does not",
          float(dc.score_points(hm, far, pf)[0]) == 0.0)

    # Out-of-volume points clamp rather than crash or read zero.
    oob = np.array([[nz + 5, -20.0, (nx + 9) * pf]], float)
    so = dc.score_points(hm, oob, pf)
    check("out-of-volume points clamp instead of crashing",
          so.shape == (1,) and np.isfinite(so[0]))

    check("empty input returns an empty array",
          dc.score_points(hm, np.zeros((0, 3)), pf).shape == (0,))
    try:
        dc.score_points(hm, np.zeros((4, 2)), pf)
        check("a wrong-width point array raises", False, "it did not")
    except ValueError:
        check("a wrong-width point array raises", True)

    print()
    print("=" * 76)
    print("batched scoring across frames")
    print("=" * 76)

    vols = {t: np.zeros((nz, ny * pf, nx * pf), np.float32) for t in range(3)}
    hms = {}
    for t in range(3):
        h = np.zeros((nz, ny, nx), np.float32)
        h[t + 2, t + 3, t + 4] = 1.0                 # a different peak per frame
        hms[t] = h

    class FakeBundle(dict):
        pass

    sc = dc.FrameScorer.__new__(dc.FrameScorer)
    sc.bundle = {"pool_factor": pf}
    sc.read_frame = lambda t: vols[t]
    sc.max_frames = 8
    sc.cache = dict(hms)                              # pre-seed; no model needed
    sc.frames_computed = 0

    ts = np.array([0, 1, 2, 0], np.int64)
    pts = np.array([[2, 3 * pf, 4 * pf],              # frame 0 peak
                    [3, 4 * pf, 5 * pf],              # frame 1 peak
                    [4, 5 * pf, 6 * pf],              # frame 2 peak
                    [9, 9 * pf, 9 * pf]], float)      # frame 0, nowhere
    got = sc.score(ts, pts)
    check("each point is scored against ITS OWN frame",
          np.allclose(got[:3], 1.0) and got[3] == 0.0, f"{np.round(got, 3).tolist()}")

    acc = sc.accept(0.5)
    check("accept() thresholds correctly",
          acc(ts, pts).tolist() == [True, True, True, False])

    print()
    print("=" * 76)
    print("close_gaps(accept=...) — filtering must happen BEFORE the greedy assignment")
    print("=" * 76)

    # One tail at frame 0. Two heads at frame 2: a NEAR one the veto will reject, and a
    # FARTHER one the veto accepts. Distance ranking puts the bad pair first, so if the
    # veto were applied after assignment the tail would already be spent and the good
    # pair would be lost. This is the contention that makes the ordering load-bearing.
    scale = (1.0, 1.0, 1.0)
    t = np.array([0, 2, 2], np.int64)
    zyx = np.array([[10.0, 10.0, 10.0],       # 0: tail
                    [10.0, 10.0, 12.0],       # 1: near head  -> midpoint (10,10,11)
                    [10.0, 10.0, 16.0]], float)  # 2: far head -> midpoint (10,10,13)
    edges = np.zeros((0, 2), np.int64)

    base = close_gaps(t, zyx, edges, scale=scale, max_um=8.0, max_added_frac=1.0)
    check("without a veto the NEAR head wins", len(base[0]) == 4
          and abs(base[1][3][2] - 11.0) < 1e-9, f"midpoint x={base[1][3][2]}")

    def veto_near(t_mid, zyx_mid):
        # reject the midpoint at x=11, accept the one at x=13
        return np.abs(np.asarray(zyx_mid)[:, 2] - 11.0) > 0.5

    out = close_gaps(t, zyx, edges, scale=scale, max_um=8.0, max_added_frac=1.0,
                     accept=veto_near)
    check("with the veto the FAR head is used instead", len(out[0]) == 4
          and abs(out[1][3][2] - 13.0) < 1e-9,
          f"midpoint x={out[1][3][2] if len(out[0]) == 4 else 'no node inserted'} — "
          "x=11 means the veto ran too late, 'no node' means the tail was consumed anyway")
    check("the inserted node still brings exactly two edges",
          len(out[2]) == 2 and set(map(tuple, out[2].tolist())) == {(0, 3), (3, 2)})

    none_pass = close_gaps(t, zyx, edges, scale=scale, max_um=8.0, max_added_frac=1.0,
                           accept=lambda tm, zm: np.zeros(len(tm), bool))
    check("a veto that rejects everything inserts nothing",
          len(none_pass[0]) == 3 and len(none_pass[2]) == 0)

    all_pass = close_gaps(t, zyx, edges, scale=scale, max_um=8.0, max_added_frac=1.0,
                          accept=lambda tm, zm: np.ones(len(tm), bool))
    check("a veto that accepts everything matches the no-veto result",
          len(all_pass[0]) == len(base[0]) and np.allclose(all_pass[1], base[1]))

    try:
        close_gaps(t, zyx, edges, scale=scale, max_um=8.0, max_added_frac=1.0,
                   accept=lambda tm, zm: np.ones(len(tm) + 1, bool))
        check("a wrong-length veto raises rather than mis-filtering", False, "it did not")
    except ValueError:
        check("a wrong-length veto raises rather than mis-filtering", True)

    print()
    print("=" * 76)
    print("the module itself")
    print("=" * 76)
    try:
        import torch
        m = dc._build_module(24)
        n_par = sum(p.numel() for p in m.parameters())
        x = torch.zeros(1, 1, 16, 32, 32)
        with torch.no_grad():
            y = m(x)
        check("the module runs and preserves shape", tuple(y.shape) == (1, 1, 16, 32, 32),
              f"{tuple(y.shape)}, {n_par:,} params")
        keys = set(m.state_dict())
        expect = {"enc1.block.0.weight", "bottleneck.block.0.weight", "up3.weight",
                  "dec1.block.4.weight", "head.weight", "head.bias"}
        check("the state_dict tree matches the published layout",
              expect <= keys, f"missing {sorted(expect - keys)}")
        v = np.random.default_rng(0).random((16, 128, 128)).astype(np.float32)
        p = dc.pool_xy(v, 4)
        check("pool_xy pools y and x only", p.shape == (16, 32, 32), f"{p.shape}")
        nb = {"norm_lo_pct": 50.0, "norm_hi_pct": 99.5,
              "norm_clip_lo": -0.5, "norm_clip_hi": 6.0}
        z = dc.normalize(p, nb)
        check("normalize clips to the published range",
              z.min() >= -0.5 - 1e-6 and z.max() <= 6.0 + 1e-6,
              f"[{z.min():.3f}, {z.max():.3f}]")
        check("a flat volume normalizes to zeros rather than NaN",
              np.all(dc.normalize(np.ones((4, 8, 8), np.float32), nb) == 0))
    except ImportError:
        print("  (torch unavailable — module tests skipped, geometry tests above stand)")

    print()
    print("=" * 76)
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("pipeline/deepcenter.py scores where it claims to")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
