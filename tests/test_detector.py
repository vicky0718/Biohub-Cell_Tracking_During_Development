"""Tests for the learned-detector support (`pipeline/detector.py`, `pipeline/unet.py`).

The numpy half runs anywhere. The torch half is skipped when torch is unavailable, so
this file stays useful on the submission image.

    python tests/test_detector.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.classical import Config, detect_frame_dog  # noqa: E402
from pipeline.detector import (  # noqa: E402
    TargetConfig, gaussian_heatmap, make_loss_mask, make_target, paired_recall,
    peaks_from_prob, recall_at_budget,
)

VOX = (1.625, 1.625, 1.625)   # isotropic grid that downsample=(1,4,4) produces
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def synth(centres, shape=(32, 32, 32), radius_um=3.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    vol = rng.normal(0, noise, shape).astype(np.float32).clip(0)
    sig = [radius_um / (2 * v) for v in VOX]
    for cz, cy, cx in centres:
        vol += np.exp(-0.5 * (((zz - cz) / sig[0]) ** 2 + ((yy - cy) / sig[1]) ** 2
                              + ((xx - cx) / sig[2]) ** 2)).astype(np.float32)
    return np.clip(vol, 0, None)


def main() -> int:
    rng = np.random.default_rng(3)
    shape = (32, 32, 32)
    centres = np.array([[8, 8, 8], [8, 22, 22], [22, 8, 22], [22, 22, 8], [16, 16, 16]],
                       float)
    vol = synth(centres, shape)
    vol = vol / vol.max()

    print("\n[training targets]")
    tgt = make_target(centres, shape, VOX, pos_radius_um=2.0)
    check("target is binary", set(np.unique(tgt)) <= {0, 1}, f"values {np.unique(tgt)}")
    check("every annotation lights up its own voxel",
          all(tgt[tuple(np.round(c).astype(int))] == 1 for c in centres),
          f"{int(tgt.sum())} positive voxels for {len(centres)} cells")
    check("positives are a small fraction of the volume",
          0 < tgt.mean() < 0.05, f"{tgt.mean():.4%} of voxels")
    bigger = make_target(centres, shape, VOX, pos_radius_um=4.0)
    check("a larger radius yields more positives", bigger.sum() > tgt.sum(),
          f"{int(tgt.sum())} at 2um -> {int(bigger.sum())} at 4um")

    # Centres outside the volume must be dropped, not clipped onto the face.
    out_of_range = np.array([[-5, -5, -5], [99, 99, 99], [16, 16, 16]], float)
    t2 = make_target(out_of_range, shape, VOX, pos_radius_um=2.0)
    check("out-of-range centres are dropped, not clipped to the face",
          t2[0, 0, 0] == 0 and t2[-1, -1, -1] == 0 and t2[16, 16, 16] == 1,
          f"{int(t2.sum())} positives from 1 in-range centre")
    check("no centres at all is empty, not an error",
          make_target(np.zeros((0, 3)), shape, VOX, 2.0).sum() == 0)

    print("\n[the loss mask keeps the ambiguous middle out of the loss]")
    cfg = TargetConfig(loss="masked")
    dog_resp, _ = detect_frame_dog(vol, VOX, Config(detector="dog",
                                                    dog_rel_threshold=0.005)), None
    # detect_frame_dog returns coords; the mask wants the RESPONSE map, so build one the
    # same way the detector does -- a DoG at the default scales.
    from scipy.ndimage import gaussian_filter
    resp = (gaussian_filter(vol, [1.5 / v for v in VOX])
            - gaussian_filter(vol, [4.0 / v for v in VOX]))

    mask = make_loss_mask(vol, resp, tgt, VOX, cfg)
    check("mask is binary", set(np.unique(mask)) <= {0, 1})
    check("every positive is inside the loss",
          bool((mask[tgt > 0] == 1).all()), f"{int(tgt.sum())} positives")
    check("some voxels are excluded", 0 < mask.mean() < 1.0,
          f"{mask.mean():.1%} of voxels carry loss")
    # The point of the mask: a real cell that is NOT annotated must not be a negative.
    unannotated = np.array([[16, 16, 16]], float)
    tgt_partial = make_target(centres[:4], shape, VOX, 2.0)   # 5th cell left unlabelled
    mask_partial = make_loss_mask(vol, resp, tgt_partial, VOX, cfg)
    at_cell = tuple(np.round(unannotated[0]).astype(int))
    check("an unannotated real cell is EXCLUDED, not labelled background",
          mask_partial[at_cell] == 0,
          f"mask at the unlabelled cell = {int(mask_partial[at_cell])}")
    check("naive mode masks nothing",
          make_loss_mask(vol, resp, tgt, VOX, TargetConfig(loss="naive")).all())

    print("\n[peaks_from_prob mirrors the DoG detector's final stage]")
    prob = gaussian_heatmap(centres, shape, VOX, sigma_um=2.0)
    coords, scores = peaks_from_prob(prob, VOX, min_separation_um=6.0)
    check("finds one peak per cell", len(coords) == len(centres),
          f"{len(coords)} peaks / {len(centres)} cells")
    check("scores come back sorted, best first",
          bool(np.all(np.diff(scores) <= 1e-9)), f"{np.round(scores, 3)}")
    d = np.linalg.norm((coords[:, None] - centres[None]) * np.array(VOX), axis=2).min(1)
    check("every peak lands within the metric's 7um match radius", bool((d < 7.0).all()),
          f"worst {d.max():.2f}um")
    capped, _ = peaks_from_prob(prob, VOX, 6.0, cap=2)
    check("cap keeps the strongest", len(capped) == 2, f"{len(capped)} peaks")
    wide, _ = peaks_from_prob(prob, VOX, min_separation_um=30.0)
    check("a wider separation yields no more peaks", len(wide) <= len(coords),
          f"{len(coords)} at 6um -> {len(wide)} at 30um")

    print("\n[recall uses the scorer's own bipartite matching]")
    gt_t = np.zeros(len(centres))
    pred_t = np.zeros(len(coords))
    r = recall_at_budget(pred_t, coords, gt_t, centres, VOX)
    check("perfect detections give recall 1.0", abs(r - 1.0) < 1e-9, f"{r:.4f}")
    half = recall_at_budget(pred_t[:2], coords[:2], gt_t, centres, VOX)
    check("half the detections give ~half the recall", abs(half - 0.4) < 1e-9, f"{half:.4f}")
    # Two GT nodes must not both claim one prediction -- that is what a greedy
    # "anything within 7um" check would get wrong.
    twin = np.array([[16, 16, 16], [16, 16, 17]], float)
    one = recall_at_budget(np.zeros(1), np.array([[16, 16, 16.5]]), np.zeros(2), twin, VOX)
    check("one prediction cannot match two nearby GT nodes", abs(one - 0.5) < 1e-9,
          f"recall {one:.4f} for 1 prediction vs 2 GT within 7um")

    print("\n[prob_fn swaps the detector inside the real pipeline]")
    import tempfile
    from pipeline.classical import predict_dataset
    tmp = Path(tempfile.mkdtemp(prefix="detector_test_"))
    T = 4
    movie = np.stack([synth(centres + t * np.array([0.0, 0.4, 0.4]), shape) for t in range(T)])
    movie = (movie / movie.max()).astype(np.float32)
    import zarr
    g = zarr.open_group(str(tmp / "m.zarr"), mode="w")
    g.create_array("0", shape=movie.shape, dtype=movie.dtype, chunks=(1,) + shape)
    g["0"][:] = movie
    g.attrs["multiscales"] = [{"datasets": [{"path": "0", "coordinateTransformations":
                               [{"type": "scale", "scale": [1.0, *VOX]}]}]}]
    g.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}

    calls = {"n": 0}

    def oracle_prob(vol):
        # A perfect detector: probability peaks exactly at the true centres. If the
        # pipeline really routes through prob_fn, this must beat the DoG arm outright.
        calls["n"] += 1
        return gaussian_heatmap(centres, vol.shape, VOX, sigma_um=2.0)

    cfg_p = Config(detector="dog", downsample=(1, 1, 1), min_separation_um=6.0,
                   dog_rel_threshold=0.005, link_radius_um=8.0)
    g_dog = predict_dataset(tmp / "m.zarr", cfg_p, verbose=False)
    g_unet = predict_dataset(tmp / "m.zarr", cfg_p, verbose=False, prob_fn=oracle_prob)
    check("prob_fn is actually called, once per frame", calls["n"] == T,
          f"{calls['n']} calls for {T} frames")
    check("the learned path produces a linked graph",
          g_unet.n_nodes > 0 and g_unet.n_edges > 0,
          f"{g_unet.n_nodes} nodes, {g_unet.n_edges} edges")
    check("an oracle detector emits exactly the true cells",
          g_unet.n_nodes == len(centres) * T,
          f"{g_unet.n_nodes} vs {len(centres) * T} expected")
    check("and differs from the DoG arm", g_unet.n_nodes != g_dog.n_nodes,
          f"unet {g_unet.n_nodes} vs dog {g_dog.n_nodes} nodes")
    # The budget path must route through prob_fn too, or the calibration would count DoG
    # peaks and then detect with the network -- the two would disagree silently.
    calls["n"] = 0
    cfg_a = Config(detector="dog", downsample=(1, 1, 1), min_separation_um=6.0,
                   dog_rel_threshold=0.005, adaptive_separation=True)
    predict_dataset(tmp / "m.zarr", cfg_a, verbose=False, prob_fn=oracle_prob,
                    est_total_nodes=float(len(centres) * T))
    check("adaptive calibration also routes through prob_fn", calls["n"] > T,
          f"{calls['n']} calls (> {T} means the calibration frames used it too)")

    print("\n[Config.temporal_radius stacks neighbouring frames as channels]")
    seen = []

    def shape_probe(x):
        seen.append(x.shape)
        return gaussian_heatmap(centres, x.shape[-3:], VOX, sigma_um=2.0)

    cfg_t = Config(detector="dog", downsample=(1, 1, 1), min_separation_um=6.0,
                   dog_rel_threshold=0.005, link_radius_um=8.0, temporal_radius=1)
    g_temp = predict_dataset(tmp / "m.zarr", cfg_t, verbose=False, prob_fn=shape_probe)
    check("prob_fn receives a (3, Z, Y, X) window at radius 1",
          all(s == (3, *shape) for s in seen), f"{sorted(set(seen))}")
    check("still one call per frame", len(seen) == T, f"{len(seen)} calls for {T} frames")
    check("the temporal arm still produces the true cells",
          g_temp.n_nodes == len(centres) * T,
          f"{g_temp.n_nodes} vs {len(centres) * T}")

    # Boundary CLAMPING, not zero padding. At t=0 the window is (0, 0, 1), so its first
    # two channels are identical -- and critically NOT a black frame the model never saw
    # in training. Checked on the real movie, frame by frame.
    got = []

    def window_probe(x):
        got.append(x.copy())
        return gaussian_heatmap(centres, x.shape[-3:], VOX, sigma_um=2.0)

    predict_dataset(tmp / "m.zarr", cfg_t, verbose=False, prob_fn=window_probe)
    check("t=0 clamps: channels 0 and 1 identical, and neither is empty",
          np.array_equal(got[0][0], got[0][1]) and got[0][0].max() > 0,
          f"max {got[0][0].max():.3f}")
    check(f"t={T-1} clamps at the other end",
          np.array_equal(got[-1][1], got[-1][2]) and got[-1][2].max() > 0)
    check("interior windows are three DIFFERENT frames",
          not np.array_equal(got[1][0], got[1][1])
          and not np.array_equal(got[1][1], got[1][2]))
    check("the window centre is frame t itself",
          all(np.array_equal(got[t][1], got[min(t + 1, T - 1)][0] if t + 1 < T else got[t][1])
              or t == T - 1 for t in range(T)))

    # temporal_radius must be inert for the classical detectors, or an arm could change
    # its detector and silently change what "frame t" means at the same time.
    a = predict_dataset(tmp / "m.zarr", cfg_p, verbose=False)
    b = predict_dataset(tmp / "m.zarr", cfg_t, verbose=False)
    check("temporal_radius is a no-op for DoG (it gets the window's centre)",
          a.n_nodes == b.n_nodes and np.allclose(a.zyx, b.zyx),
          f"{a.n_nodes} vs {b.n_nodes} nodes")

    print("\n[paired_recall sees the coherence that node recall is blind to]")
    # The notes/21 §2 scenario, reproduced exactly: two detectors with IDENTICAL node
    # recall and very different edge behaviour. If this metric cannot separate them it is
    # no better than the one it replaces.
    n_cells = 10
    gt_zyx2 = np.repeat(np.arange(n_cells)[:, None] * 8.0 + 4.0, 3, axis=1)
    gt_t2 = np.concatenate([np.zeros(n_cells), np.ones(n_cells)])
    gt_all = np.vstack([gt_zyx2, gt_zyx2])                 # cells do not move
    gt_e = np.column_stack([np.arange(n_cells), np.arange(n_cells) + n_cells])

    def arm(f0, f1):
        return (np.concatenate([np.zeros(len(f0)), np.ones(len(f1))]),
                np.vstack([gt_zyx2[f0], gt_zyx2[f1]]))

    keep = list(range(8))
    coh_t, coh_z = arm(keep, keep)                          # same 8 cells both frames
    inc_t, inc_z = arm(keep, list(range(2, 10)))            # 8 cells, only 6 in common

    coh = paired_recall(coh_t, coh_z, gt_t2, gt_all, gt_e, VOX)
    inc = paired_recall(inc_t, inc_z, gt_t2, gt_all, gt_e, VOX)
    check("both arms have IDENTICAL node recall", abs(coh["node"] - inc["node"]) < 1e-9,
          f"{coh['node']:.4f} both")
    check("but paired recall separates them",
          abs(coh["paired"] - 0.8) < 1e-9 and abs(inc["paired"] - 0.6) < 1e-9,
          f"coherent {coh['paired']:.2f} vs incoherent {inc['paired']:.2f} "
          f"at node recall {coh['node']:.2f}")
    check("the coherent arm sits at the top of the interval",
          abs(coh["position"] - 1.0) < 1e-9, f"position {coh['position']:+.3f}")
    check("the incoherent arm falls BELOW independence, as the UNet did",
          inc["position"] < 0, f"position {inc['position']:+.3f} "
          f"(paired {inc['paired']:.2f} < r^2 = {inc['independent']:.2f})")
    # A model that finds every cell every frame is the r == paired == 1 corner; position
    # is 0/0 there and must not come back as a number that ranks above a real result.
    per_t, per_z = arm(list(range(n_cells)), list(range(n_cells)))
    per = paired_recall(per_t, per_z, gt_t2, gt_all, gt_e, VOX)
    check("a perfect detector gives paired 1.0 and an undefined position",
          abs(per["paired"] - 1.0) < 1e-9 and np.isnan(per["position"]),
          f"paired {per['paired']:.2f}, position {per['position']}")
    # An edge leaving the evaluated window is unscoreable; charging the model for it would
    # make any frame subset look incoherent for a reason unrelated to the model.
    sub = paired_recall(coh_t, coh_z, gt_t2, gt_all, gt_e, VOX, frames=[0])
    check("frames= drops edges that leave the evaluated window", sub["n_edges"] == 0,
          f"{sub['n_edges']} edges kept when only frame 0 is evaluated")

    print("\n[Config.refine gates the intensity-weighted shift]")
    from pipeline.classical import Config as _C
    g_on = predict_dataset(tmp / "m.zarr", cfg_p, verbose=False, prob_fn=oracle_prob)
    cfg_off = _C(detector="dog", downsample=(1, 1, 1), min_separation_um=6.0,
                 dog_rel_threshold=0.005, link_radius_um=8.0, refine=False)
    g_off = predict_dataset(tmp / "m.zarr", cfg_off, verbose=False, prob_fn=oracle_prob)
    check("refine=True and refine=False give the same node COUNT",
          g_on.n_nodes == g_off.n_nodes, f"{g_on.n_nodes} vs {g_off.n_nodes}")
    moved = float(np.abs(g_on.zyx - g_off.zyx).max()) if g_on.n_nodes == g_off.n_nodes else -1
    check("but different coordinates — the shift is real and gated", moved > 1e-9,
          f"max coordinate shift {moved:.4f} voxels")
    # With an ORACLE probability map the peaks sit exactly on the cells, so any intensity
    # shift can only move them away. That is the whole hypothesis in miniature.
    # Compare PER FRAME: graph node order interleaves frames, and the oracle map is built
    # from `centres` at every t, so frame-0 truth is the reference throughout.
    def mean_err(g):
        d = []
        for t in np.unique(g.t):
            zz = g.zyx[g.t == t]
            if not len(zz):
                continue
            dist = np.linalg.norm((zz[:, None] - centres[None]) * np.array(VOX), axis=2)
            d.append(dist.min(axis=1).mean())
        return float(np.mean(d))
    d_on, d_off = mean_err(g_on), mean_err(g_off)
    check("on an oracle map, refinement moves peaks AWAY from truth", d_off < d_on,
          f"refined {d_on:.3f}um vs unrefined {d_off:.3f}um "
          f"({d_on - d_off:+.3f}um of pure damage)")

    shutil.rmtree(tmp, ignore_errors=True)
    try:
        import torch
    except ImportError:
        print("\n[torch half SKIPPED — torch unavailable]")
    else:
        from pipeline.unet import LOSSES, UNet3D, count_params, predict_volume
        print(f"\n[UNet3D forward/backward — torch {torch.__version__}]")
        dev = torch.device("cpu")
        model = UNet3D(base=8, depth=3).to(dev)
        check("parameter count is modest", count_params(model) < 2_000_000,
              f"{count_params(model):,} parameters")
        x = torch.randn(2, 1, 32, 32, 32)
        y = model(x)
        check("output shape matches input", y.shape == (2, 1, 32, 32, 32), str(tuple(y.shape)))

        # A non-power-of-two depth axis is the realistic case if downsample changes.
        odd = model(torch.randn(1, 1, 24, 32, 32))
        check("a non-power-of-two input still round-trips",
              odd.shape == (1, 1, 24, 32, 32), str(tuple(odd.shape)))

        t = torch.as_tensor(tgt, dtype=torch.float32)[None, None]
        m = torch.as_tensor(mask, dtype=torch.float32)[None, None]
        logits = model(torch.as_tensor(vol, dtype=torch.float32)[None, None])
        for name, fn in LOSSES.items():
            val = fn(logits, t, m, prior=1 / 28)
            check(f"{name} loss is finite and positive",
                  bool(torch.isfinite(val)) and float(val) >= 0, f"{float(val):.4f}")
            val.backward(retain_graph=True)
        check("gradients reached the first conv",
              model.down[0].net[0].weight.grad is not None
              and bool(torch.isfinite(model.down[0].net[0].weight.grad).all()))

        # The mask must actually change the gradient, or variant B is variant A.
        g_masked = torch.autograd.grad(LOSSES["masked"](logits, t, m), logits,
                                       retain_graph=True)[0]
        check("masked loss has zero gradient on excluded voxels",
              float(g_masked[m == 0].abs().max()) < 1e-12,
              f"max |grad| outside the mask = {float(g_masked[m == 0].abs().max()):.2e}")
        check("masked loss has nonzero gradient inside it",
              float(g_masked[m > 0].abs().max()) > 0)

        p = predict_volume(model, vol, dev, amp=False)
        check("predict_volume returns a probability map",
              p.shape == vol.shape and 0.0 <= p.min() and p.max() <= 1.0,
              f"range [{p.min():.3f}, {p.max():.3f}]")

        # A temporal model takes 3 channels in and still predicts ONE frame's centres.
        tmodel = UNet3D(base=8, depth=2, in_ch=3)
        stack = np.stack([vol, vol, vol])
        pt = predict_volume(tmodel, stack, dev, amp=False)
        check("predict_volume accepts a (3, Z, Y, X) window and returns (Z, Y, X)",
              pt.shape == vol.shape, f"{stack.shape} -> {pt.shape}")
        # The silent failure this guards: a 1-channel model handed a 3-frame stack would
        # otherwise read the window as a BATCH of three frames and return the wrong one.
        for bad, m_, label in ((stack, model, "1-channel model given a 3-frame window"),
                               (vol, tmodel, "3-channel model given a single frame")):
            try:
                predict_volume(m_, bad, dev, amp=False)
                ok_ = False
            except ValueError:
                ok_ = True
            check(f"rejects a {label}", ok_)

        print("\n[the network can actually learn this signal]")
        # 60 steps on one volume. Not a benchmark -- a wiring check. If the loss cannot
        # be driven down on a single trivial example, nothing downstream is worth running.
        # Seeded: without this the initial loss ranges 0.66-0.96 across runs and the
        # "halved" assertion fails by chance roughly one run in ten. A flaky gate is
        # worse than no gate -- it trains you to ignore it.
        torch.manual_seed(0)
        model = UNet3D(base=8, depth=2).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        xin = torch.as_tensor(vol, dtype=torch.float32)[None, None]
        first = last = None
        for step in range(60):
            opt.zero_grad()
            loss = LOSSES["masked"](model(xin), t, m)
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss)
            last = float(loss)
        check("masked loss falls with training", last < first * 0.5,
              f"{first:.4f} -> {last:.4f}")
        prob = predict_volume(model, vol, dev, amp=False)
        got, _ = peaks_from_prob(prob, VOX, min_separation_um=6.0, cap=len(centres))
        rec = recall_at_budget(np.zeros(len(got)), got, np.zeros(len(centres)),
                               centres, VOX)
        check("the overfit model recovers its own cells at matched budget", rec >= 0.8,
              f"recall {rec:.2f} with {len(got)} detections for {len(centres)} cells")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all detector tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
