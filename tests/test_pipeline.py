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
    Config, _footprint, detect_frame, link_all, predict_dataset, refine_centroids,
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
    check("pipeline produces nodes", graph.num_nodes() > 0, f"{graph.num_nodes()} nodes")
    check("pipeline produces edges", graph.num_edges() > 0, f"{graph.num_edges()} edges")
    expected = len(centres) * (T - 1)
    check("edge count is in the right ballpark",
          0.7 * expected <= graph.num_edges() <= 1.3 * expected,
          f"{graph.num_edges()} edges vs ~{expected} expected")

    print("\n[downsampling maps coordinates back to original space]")
    cfg_ds = Config(det_threshold=0.3, min_separation_um=5.0, downsample=(1, 2, 2))
    g2 = predict_dataset(ds, cfg_ds, verbose=False)
    na = g2.node_attrs()
    check("downsampled coords land in the original range",
          float(na["y"].max()) > shape[1] / 2,
          f"max y = {float(na['y'].max()):.1f} (volume is {shape[1]} voxels)")

    print("\n[scored with the official scorer]")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from harness.scorer import evaluate, per_sample_metrics, node_recall
        from pipeline.classical import build_graph

        # ground truth: the true positions, linked by identity
        gt_coords, gt_edges = [], []
        for t in range(T):
            for i, p in enumerate(truth[t]):
                gt_coords.append([t, *p])
        n = len(centres)
        for t in range(T - 1):
            for i in range(n):
                gt_edges.append((t * n + i, (t + 1) * n + i))
        gt = build_graph(np.array(gt_coords), gt_edges)

        er = evaluate(graph, gt, scale=SCALE, max_distance=7.0)
        rec = node_recall(graph, gt)
        row = per_sample_metrics(er, float(gt.num_nodes()), rec)
        print(f"    TP/FP/FN = {er.edge_tp}/{er.edge_fp}/{er.edge_fn}   "
              f"edge_J={row['edge_jaccard']:.4f}  adj={row['adj_edge_jaccard']:.4f}  "
              f"node_recall={rec:.3f}")
        check("scores well on clean synthetic data", row["edge_jaccard"] > 0.8,
              f"edge_jaccard={row['edge_jaccard']:.4f}")
    except ImportError as e:
        print(f"    (skipped — official scorer unavailable: {e})")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all pipeline tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
