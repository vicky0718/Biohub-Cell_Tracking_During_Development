"""Tests for the classical pipeline against synthetic 3D+time microscopy.

Builds anisotropic volumes with Gaussian "nuclei" at known positions, writes them as
OME-Zarr, and checks that the pipeline recovers those positions and links them.
Runs without the competition data.

    CELLMOT_REPO=/path/to/kaggle-cell-tracking-competition python tests/test_pipeline.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.classical import (  # noqa: E402
    Config, _footprint, detect_frame, estimated_total_nodes, link_all, predict_dataset,
    refine_centroids,
)

SCALE = (1.625, 0.40625, 0.40625)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def synth_volume(centres, shape=(24, 128, 128), radius_um=4.0, scale=SCALE, noise=0.02, seed=0):
    """Gaussian blobs at `centres` (voxel coords), blurred anisotropically in physical space."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    vol = rng.normal(0, noise, shape).astype(np.float32).clip(0)
    sig = np.array([radius_um / s for s in scale])  # blob sigma in voxels, per axis
    for cz, cy, cx in centres:
        d2 = (((zz - cz) / sig[0]) ** 2 + ((yy - cy) / sig[1]) ** 2 + ((xx - cx) / sig[2]) ** 2)
        vol += np.exp(-0.5 * d2).astype(np.float32)
    return np.clip(vol, 0, None)


def write_zarr(path: Path, movie: np.ndarray, scale=SCALE):
    import zarr

    if path.exists():
        shutil.rmtree(path)
    g = zarr.open_group(str(path), mode="w")
    g.create_array("0", shape=movie.shape, dtype=movie.dtype, chunks=(1,) + movie.shape[1:])
    g["0"][:] = movie
    g.attrs["multiscales"] = [{
        "datasets": [{"path": "0", "coordinateTransformations":
                      [{"type": "scale", "scale": [1.0, *scale]}]}]
    }]
    g.attrs["image_statistics"] = {"quantiles": {"0.001": float(np.quantile(movie, 0.001)),
                                                 "0.999": float(np.quantile(movie, 0.999))}}
    return path


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))

    print("\n[footprint is anisotropic]")
    fp = _footprint(5.0, SCALE)
    check("5um window spans fewer voxels in Z than XY", fp[0] < fp[1],
          f"footprint={fp} (Z, Y, X)")
    check("all footprint dims are odd", all(k % 2 == 1 for k in fp), str(fp))

    print("\n[detection recovers known nuclei]")
    shape = (24, 128, 128)
    rng = np.random.default_rng(7)
    # Keep nuclei well apart so detection is unambiguous: >= 12 um in XY.
    centres = []
    while len(centres) < 12:
        c = np.array([rng.uniform(5, shape[0] - 5), rng.uniform(12, shape[1] - 12),
                      rng.uniform(12, shape[2] - 12)])
        phys = c * np.array(SCALE)
        if all(np.linalg.norm(phys - np.array(o) * np.array(SCALE)) > 12 for o in centres):
            centres.append(c)
    centres = np.array(centres)

    vol = synth_volume(centres, shape=shape)
    vol = vol / vol.max()
    cfg = Config(det_threshold=0.3, min_separation_um=5.0, downsample=(1, 1, 1))
    got, scores = detect_frame(vol, SCALE, cfg)
    check("finds every nucleus", len(got) >= len(centres), f"{len(got)} found / {len(centres)} true")

    # each true centre should have a detection within the 7um match radius
    if len(got):
        gp = got * np.array(SCALE)
        cp = centres * np.array(SCALE)
        d = np.linalg.norm(cp[:, None, :] - gp[None, :, :], axis=2).min(axis=1)
        check("every nucleus matched within 7um", bool((d < 7.0).all()),
              f"worst = {d.max():.2f}um")

        ref = refine_centroids(vol, got, SCALE)
        gp2 = ref * np.array(SCALE)
        d2 = np.linalg.norm(cp[:, None, :] - gp2[None, :, :], axis=2).min(axis=1)
        check("centroid refinement does not make Z worse", d2.mean() <= d.mean() + 1e-6,
              f"mean err {d.mean():.3f}um -> {d2.mean():.3f}um")

    print("\n[threshold controls the detection count — the knob for experiment #1]")
    counts = {}
    for th in (0.6, 0.3, 0.1, 0.02):
        c, _ = detect_frame(vol, SCALE, Config(det_threshold=th, min_separation_um=5.0))
        counts[th] = len(c)
    print(f"    threshold -> detections: {counts}")
    check("lowering the threshold never loses detections",
          all(counts[a] <= counts[b] for a, b in [(0.6, 0.3), (0.3, 0.1), (0.1, 0.02)]),
          str(counts))

    print("\n[max_per_frame caps the node budget]")
    capped, _ = detect_frame(vol, SCALE, Config(det_threshold=0.02, min_separation_um=5.0,
                                                max_per_frame=5))
    check("cap is respected", len(capped) == 5, f"{len(capped)} detections")

    print("\n[end to end on a synthetic movie]")
    T = 6
    vel = rng.normal(0, 0.4, size=(len(centres), 3)) * np.array([0.2, 1.0, 1.0])
    frames, truth = [], []
    for t in range(T):
        pos = centres + vel * t
        pos[:, 0] = np.clip(pos[:, 0], 4, shape[0] - 5)
        pos[:, 1] = np.clip(pos[:, 1], 10, shape[1] - 11)
        pos[:, 2] = np.clip(pos[:, 2], 10, shape[2] - 11)
        truth.append(pos.copy())
        f = synth_volume(pos, shape=shape, seed=t)
        frames.append(f / f.max())
    movie = np.stack(frames).astype(np.float32)

    ds = write_zarr(tmp / "synth.zarr", movie)
    cfg = Config(det_threshold=0.3, min_separation_um=5.0, link_radius_um=8.0,
                 downsample=(1, 1, 1))
    graph = predict_dataset(ds, cfg, verbose=False)
    check("pipeline produces nodes", graph.n_nodes > 0, f"{graph.n_nodes} nodes")
    check("pipeline produces edges", graph.n_edges > 0, f"{graph.n_edges} edges")
    expected = len(centres) * (T - 1)
    check("edge count is in the right ballpark",
          0.7 * expected <= graph.n_edges <= 1.3 * expected,
          f"{graph.n_edges} edges vs ~{expected} expected")

    print("\n[the per-dataset node budget caps detections]")
    # recon §9: the budget varies 20.8x across datasets, so the cap has to come from
    # the dataset itself. With no budget anywhere, the pipeline must still run.
    cfg_free = Config(det_threshold=0.02, min_separation_um=5.0, downsample=(1, 1, 1),
                      budget_fill=None)
    n_free = predict_dataset(ds, cfg_free, verbose=False).n_nodes
    # est_total = 3 per frame over T frames -> cap of 3 detections per frame
    cfg_budget = Config(det_threshold=0.02, min_separation_um=5.0, downsample=(1, 1, 1),
                        budget_fill=1.0)
    n_budget = predict_dataset(ds, cfg_budget, verbose=False,
                               est_total_nodes=3.0 * T).n_nodes
    check("budget cap binds", n_budget == 3 * T, f"{n_budget} nodes, expected {3 * T}")
    check("budget cap is tighter than uncapped", n_budget < n_free,
          f"{n_budget} capped vs {n_free} uncapped")
    n_half = predict_dataset(ds, Config(det_threshold=0.02, min_separation_um=5.0,
                                        downsample=(1, 1, 1), budget_fill=0.5),
                             verbose=False, est_total_nodes=8.0 * T).n_nodes
    check("budget_fill scales the cap", n_half == 4 * T,
          f"{n_half} nodes at fill=0.5 of 8/frame, expected {4 * T}")
    check("a missing budget does not cap or crash",
          predict_dataset(ds, cfg_budget, verbose=False).n_nodes == n_free,
          "no .geff and no zarr attr -> uncapped")
    check("estimated_total_nodes returns None when absent",
          estimated_total_nodes(ds) is None, "synthetic zarr carries no budget")
    check("estimated_total_nodes reads the zarr attrs",
          estimated_total_nodes(ds, {"extra": {"estimated_number_of_nodes": 1234}}) == 1234.0,
          "nested lookup")

    print("\n[DoG detection recovers DIM nuclei that intensity thresholding misses]")
    from pipeline.classical import _ball_footprint, detect_frame_dog

    fpb = _ball_footprint(5.0, SCALE)
    check("ball footprint is not a box", fpb.sum() < fpb.size,
          f"{int(fpb.sum())} of {fpb.size} voxels inside the 5um sphere")
    check("ball footprint is anisotropic", fpb.shape[0] < fpb.shape[1], str(fpb.shape))

    # bright nuclei plus DIM ones at 12% amplitude sitting on a smooth bright gradient —
    # the case an absolute intensity cut cannot separate but a band-pass can.
    dim_shape = (24, 128, 128)
    bright = centres[:6]
    dim = centres[6:12]
    v = synth_volume(bright, shape=dim_shape, seed=1)
    v += 0.12 * synth_volume(dim, shape=dim_shape, noise=0.0, seed=2)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in dim_shape], indexing="ij")
    v += (0.55 * yy / dim_shape[1]).astype(np.float32)      # smooth background ramp
    v = v / v.max()

    cfg_i = Config(det_threshold=0.15, min_separation_um=4.0, downsample=(1, 1, 1))
    cfg_d = Config(detector="dog", min_separation_um=4.0, downsample=(1, 1, 1),
                   dog_scales=[(1.5, 4.0), (2.5, 6.0)])
    gi, _ = detect_frame(v, SCALE, cfg_i)
    gd, _ = detect_frame_dog(v, SCALE, cfg_d)

    def recall(got, truth, tol=7.0):
        if not len(got):
            return 0.0
        d = np.linalg.norm((np.array(truth) * SCALE)[:, None, :]
                           - (got * SCALE)[None, :, :], axis=2).min(axis=1)
        return float((d < tol).mean())

    # Compare at a MATCHED detection budget. Without that the intensity detector can
    # "win" recall by spraying peaks over the background ramp — 153 detections for 12
    # nuclei is not a detector, it is a shotgun. Equal budget, strongest peaks kept,
    # is the comparison that reflects what the node budget actually charges for.
    N = len(gd)
    gi_eq, _ = detect_frame(v, SCALE, Config(det_threshold=0.15, min_separation_um=4.0,
                                             downsample=(1, 1, 1), max_per_frame=N))
    ri, rd = recall(gi_eq, dim), recall(gd, dim)
    print(f"    at a matched budget of {N} detections — dim-nucleus recall: "
          f"intensity {ri:.2f} vs DoG {rd:.2f}")
    print(f"    (unmatched, intensity needs {len(gi)} detections to reach "
          f"{recall(gi, dim):.2f})")
    check("DoG finds more dim nuclei per detection spent", rd > ri, f"{rd:.2f} vs {ri:.2f}")
    check("DoG still finds the bright ones", recall(gd, bright) >= 0.8,
          f"bright recall {recall(gd, bright):.2f}")
    check("DoG is far more economical", len(gd) < len(gi) / 3,
          f"{len(gd)} vs {len(gi)} detections for the same volume")
    check("single-scale DoG also detects",
          len(detect_frame_dog(v, SCALE, Config(detector="dog", min_separation_um=4.0,
                                                downsample=(1,1,1)))[0]) > 0, "")

    print("\n[graph repair: gap closing and isolated pruning]")
    from pipeline.classical import close_gaps, prune_isolated

    # a track detected at t=0 and t=2 but MISSING at t=1. A direct 0->2 edge is dropped
    # by the scorer, so the repair has to insert a node instead.
    gc = np.array([[0., 10., 50., 50.], [2., 10., 52., 52.]])
    cfg_gap = Config(gap_close=1, link_radius_um=9.0)
    c2, e2 = close_gaps(gc, [], SCALE, cfg_gap)
    check("gap closing inserts a node", len(c2) == 3, f"{len(c2)} nodes")
    check("inserted node sits at t+1", len(c2) == 3 and c2[2, 0] == 1.0,
          f"t={c2[2, 0] if len(c2) > 2 else 'n/a'}")
    check("inserted node is at the midpoint", len(c2) == 3
          and abs(c2[2, 2] - 51.0) < 1e-9, f"y={c2[2, 2] if len(c2) > 2 else 'n/a'}")
    check("it makes TWO consecutive edges, not one bridge",
          len(e2) == 2 and all(abs(c2[d, 0] - c2[s, 0]) == 1 for s, d in e2),
          f"edges={e2}")
    check("gap closing is off by default",
          close_gaps(gc, [], SCALE, Config())[0].shape[0] == 2, "no nodes added")

    # too far apart to be the same cell across a 2-frame gap
    far = np.array([[0., 10., 50., 50.], [2., 10., 200., 200.]])
    check("gap closing respects the radius",
          len(close_gaps(far, [], SCALE, cfg_gap)[0]) == 2, "no bridge over 60um")

    # the cap must bind
    many = np.array([[float(t), 10., 50. + 20 * i, 50.] for i in range(40) for t in (0, 2)])
    c3, _ = close_gaps(many, [], SCALE, Config(gap_close=1, link_radius_um=9.0,
                                               gap_close_max_frac=0.05))
    check("gap_close_max_frac caps insertions",
          len(c3) - len(many) <= int(0.05 * len(many)) ,
          f"added {len(c3) - len(many)}, cap {int(0.05 * len(many))}")

    iso = np.array([[0., 1., 1., 1.], [1., 1., 1., 1.], [0., 9., 9., 9.]])
    c4, e4 = prune_isolated(iso, [(0, 1)])
    check("prune drops the unlinked node", len(c4) == 2, f"{len(c4)} nodes")
    check("prune reindexes edges correctly", e4 == [(0, 1)], f"{e4}")

    print("\n[downsampling maps coordinates back to original space]")
    cfg_ds = Config(det_threshold=0.3, min_separation_um=5.0, downsample=(1, 2, 2))
    g2 = predict_dataset(ds, cfg_ds, verbose=False)
    check("downsampled coords land in the original range",
          float(g2.zyx[:, 1].max()) > shape[1] / 2,
          f"max y = {float(g2.zyx[:, 1].max()):.1f} (volume is {shape[1]} voxels)")

    print("\n[scored end to end — purescore must equal the official scorer]")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness import purescore
    from pipeline.classical import build_graph

    # ground truth: the true positions, linked by identity
    gt_coords, gt_edges = [], []
    for t in range(T):
        for p in truth[t]:
            gt_coords.append([t, *p])
    n = len(centres)
    for t in range(T - 1):
        gt_edges += [(t * n + i, (t + 1) * n + i) for i in range(n)]
    gt = build_graph(np.array(gt_coords), gt_edges)

    pc = purescore.count_edges(graph.t, graph.zyx, graph.edges,
                               gt.t, gt.zyx, gt.edges, scale=SCALE)
    row = purescore.per_sample(pc, float(gt.n_nodes), gt.n_nodes)
    print(f"    purescore TP/FP/FN = {pc.tp}/{pc.fp}/{pc.fn}   "
          f"edge_J={row['edge_jaccard']:.4f}  adj={row['adj_edge_jaccard']:.4f}")
    check("scores well on clean synthetic data", row["edge_jaccard"] > 0.8,
          f"edge_jaccard={row['edge_jaccard']:.4f}")

    try:
        from harness.scorer import evaluate, node_recall, per_sample_metrics
        td_pred, td_gt = graph.to_tracksdata(), gt.to_tracksdata()
        er = evaluate(td_pred, td_gt, scale=SCALE, max_distance=7.0)
        off = per_sample_metrics(er, float(gt.n_nodes), node_recall(td_pred, td_gt))
        # The claim purescore rests on. If this ever fails, every number this project
        # reports from Kaggle is wrong — so it is a hard failure, not a warning.
        check("purescore TP/FP/FN == official",
              (pc.tp, pc.fp, pc.fn) == (er.edge_tp, er.edge_fp, er.edge_fn),
              f"pure {(pc.tp, pc.fp, pc.fn)} vs official "
              f"{(er.edge_tp, er.edge_fp, er.edge_fn)}")
        check("purescore adj_edge_jaccard == official",
              abs(row["adj_edge_jaccard"] - off["adj_edge_jaccard"]) < 1e-9,
              f"pure {row['adj_edge_jaccard']:.10f} vs official {off['adj_edge_jaccard']:.10f}")
    except ImportError as e:
        print(f"    (cross-check skipped — tracksdata unavailable here: {e})")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all pipeline tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
