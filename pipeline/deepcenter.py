"""A second detector used as an *add-only veto* on nodes the repair chain invents.

`notes/33`: the 0.927 public notebook attaches three model datasets where we attach one.
The second of them, `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, is not part of
its detection path at all -- it never adds a node. It only ever *rejects* one: every
gap-closed node and every geometric division candidate must score above a threshold on
this model's heatmap before the repair is allowed to commit it.

That is a direct answer to a cost this project measured twice and accepted. `close_gaps`
inserts a node at the midpoint of a plausible gap **without ever looking at the image**,
and `notes/27` §1 and `notes/31` §3 both recorded the consequence as `fn_detect` rising
(+36 at the best ILP arm) while `fn_gap` and `fn_mislink` fell. Some invented nodes land
where no cell is; the veto is a way to ask.

Architecture, preprocessing and scoring geometry are reproduced from that notebook's own
source so the published checkpoint loads -- `load_state_dict` is strict, so any deviation
in the module tree fails loudly rather than silently scoring noise.

Coordinates: `zyx` is in **voxels**, matching `Tracks`. The heatmap keeps full z
resolution and is pooled by `pool_factor` in y and x, so y and x are divided and z is not.
Mixing that up is silent -- it just scores the wrong place -- so `score_points` is the
only thing that knows it, and `probes/exec_deepcenter.py` pins it with a synthetic blob.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Reproduced from the public notebook. `base_channels` comes from the checkpoint's own
# config; the rest is fixed by the published weights.
POOL_FACTOR = 4
NORM_LO_PCT, NORM_HI_PCT = 50.0, 99.5
NORM_CLIP_LO, NORM_CLIP_HI = -0.5, 6.0
WIN_Z, WIN_YX = 1, 2
GAP_THRESHOLD = 0.25        # the notebook's BIOHUB_DEEPCENTER_GAP_THRESHOLD
DIV_THRESHOLD = 0.12        # the notebook's BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD


def _build_module(base_channels: int):
    import torch
    from torch import nn

    class ConvBlock3d(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            groups = min(8, cout)
            self.block = nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                nn.GroupNorm(groups, cout),
                nn.SiLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                nn.GroupNorm(groups, cout),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class DeepCenterUNet3D(nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
            super().__init__()
            c = int(base_channels)
            self.enc1 = ConvBlock3d(in_channels, c)
            self.down1 = nn.MaxPool3d(2, 2)
            self.enc2 = ConvBlock3d(c, c * 2)
            self.down2 = nn.MaxPool3d(2, 2)
            self.enc3 = ConvBlock3d(c * 2, c * 4)
            self.down3 = nn.MaxPool3d(2, 2)
            self.bottleneck = ConvBlock3d(c * 4, c * 8)
            self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
            self.dec3 = ConvBlock3d(c * 8, c * 4)
            self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
            self.dec2 = ConvBlock3d(c * 4, c * 2)
            self.up1 = nn.ConvTranspose3d(c * 2, c, 2, 2)
            self.dec1 = ConvBlock3d(c * 2, c)
            self.head = nn.Conv3d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    return DeepCenterUNet3D(base_channels=base_channels)


def load(checkpoint_path: str | Path, device: str | None = None) -> dict:
    """Load the published checkpoint. Raises rather than returning a degraded bundle.

    The public notebook falls back to "no veto" when the checkpoint will not load. Here
    that would silently turn a veto arm into its own control and report the two as
    different, so this raises instead.
    """
    import torch

    path = Path(checkpoint_path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"{path}: checkpoint has no model_state")
    cfg = dict(ckpt.get("config", {}))
    model = _build_module(int(cfg.get("base_channels", 24)))
    model.load_state_dict(ckpt["model_state"])       # strict: a wrong tree fails here
    model.to(dev).eval()
    return {
        "model": model, "device": dev, "path": path,
        "pool_factor": int(cfg.get("pool_factor", POOL_FACTOR)),
        "norm_lo_pct": float(cfg.get("norm_lo_pct", NORM_LO_PCT)),
        "norm_hi_pct": float(cfg.get("norm_hi_pct", NORM_HI_PCT)),
        "norm_clip_lo": float(cfg.get("norm_clip_lo", NORM_CLIP_LO)),
        "norm_clip_hi": float(cfg.get("norm_clip_hi", NORM_CLIP_HI)),
        "epoch": ckpt.get("epoch"), "best_score": ckpt.get("best_score"),
    }


def pool_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2, x2 = (y // factor) * factor, (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def normalize(volume: np.ndarray, bundle: dict) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, bundle["norm_lo_pct"]))
    hi = float(np.percentile(vol, bundle["norm_hi_pct"]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo),
                   bundle["norm_clip_lo"], bundle["norm_clip_hi"]).astype(np.float32)


def heatmap(volume: np.ndarray, bundle: dict) -> np.ndarray:
    """`(Z, Y//pool, X//pool)` of sigmoid probabilities for one frame."""
    import torch

    image = normalize(pool_xy(volume, bundle["pool_factor"]), bundle)
    with torch.no_grad():
        tensor = torch.from_numpy(image[None, None, ...]).to(
            device=bundle["device"], dtype=torch.float32)
        out = torch.sigmoid(bundle["model"](tensor))[0, 0]
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def score_points(hm: np.ndarray, zyx: np.ndarray, pool_factor: int,
                 win_z: int = WIN_Z, win_yx: int = WIN_YX) -> np.ndarray:
    """Max heatmap value in a window around each point. `zyx` in VOXELS, `(K, 3)`.

    Vectorised over points. The public notebook does this one point at a time inside the
    repair loop; a gap-close pass proposes thousands of candidates per dataset and this
    project has already been bitten once by a per-element lookup in an inner loop
    (`notes/29`, the relink probability table), so it is done as one gather here.

    y and x are divided by `pool_factor`; z is NOT -- the heatmap keeps full z resolution.
    """
    zyx = np.atleast_2d(np.asarray(zyx, dtype=np.float64))
    if zyx.size == 0:
        return np.zeros(0, np.float32)
    if zyx.shape[1] != 3:
        raise ValueError(f"expected (K, 3) zyx, got {zyx.shape}")
    pf = max(int(pool_factor), 1)
    nz, ny, nx = hm.shape

    z = np.rint(zyx[:, 0]).astype(np.int64)
    y = np.rint(zyx[:, 1] / pf).astype(np.int64)
    x = np.rint(zyx[:, 2] / pf).astype(np.int64)

    # One gather over the full window box, clipped at the volume edge. Points outside the
    # volume clamp to the nearest in-bounds voxel rather than scoring 0 -- a point one
    # voxel past the border is a rounding artifact, not evidence of absence.
    dz = np.arange(-win_z, win_z + 1)
    dy = np.arange(-win_yx, win_yx + 1)
    dx = np.arange(-win_yx, win_yx + 1)
    zz = np.clip(z[:, None, None, None] + dz[None, :, None, None], 0, nz - 1)
    yy = np.clip(y[:, None, None, None] + dy[None, None, :, None], 0, ny - 1)
    xx = np.clip(x[:, None, None, None] + dx[None, None, None, :], 0, nx - 1)
    return hm[zz, yy, xx].reshape(len(zyx), -1).max(axis=1).astype(np.float32)


class FrameScorer:
    """Scores points against per-frame heatmaps, computing each frame at most once.

    A repair pass touches frames in whatever order its candidate ranking produces, so the
    cache is keyed by frame and bounded; `max_frames` exists because a full 64x256x256
    heatmap is ~4 MB at pool 4 and 100 of them is not free.
    """

    def __init__(self, bundle: dict, read_frame, max_frames: int = 8) -> None:
        self.bundle = bundle
        self.read_frame = read_frame
        self.max_frames = max(1, int(max_frames))
        self.cache: dict[int, np.ndarray] = {}
        self.frames_computed = 0

    def heatmap(self, t: int) -> np.ndarray:
        t = int(t)
        hm = self.cache.get(t)
        if hm is None:
            hm = heatmap(self.read_frame(t), self.bundle)
            self.frames_computed += 1
            self.cache[t] = hm
            while len(self.cache) > self.max_frames:
                self.cache.pop(next(iter(self.cache)))
        return hm

    def score(self, t: np.ndarray, zyx: np.ndarray) -> np.ndarray:
        """Score `(K, 3)` points at their own frames `t`, batched per frame."""
        t = np.asarray(t, np.int64)
        zyx = np.atleast_2d(np.asarray(zyx, float))
        out = np.zeros(len(t), np.float32)
        for f in np.unique(t):
            sel = t == f
            out[sel] = score_points(self.heatmap(int(f)), zyx[sel],
                                    self.bundle["pool_factor"])
        return out

    def accept(self, threshold: float):
        """An `accept` callable for `pipeline.repair.close_gaps`."""
        def _accept(t_mid: np.ndarray, zyx_mid: np.ndarray) -> np.ndarray:
            return self.score(t_mid, zyx_mid) >= float(threshold)
        return _accept
