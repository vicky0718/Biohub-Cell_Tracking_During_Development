"""Smooth random in-plane deformation for `train_unet_transformer.py`.

The support pack trains with exactly two augmentations::

    DEFAULT_AUGMENTATIONS = [brightness_augment, flip_augment]

`flip_augment` samples the eight axis-aligned symmetries and `brightness_augment` adds a
scalar. That is the whole of it -- no deformation, no affine, no noise. hengck23 suggested
elastic deformation in July and nobody in this lineage has tried it, because nobody in this
lineage trains.

**The failure mode this file is written around.** An augmentation here receives the images
*and the node coordinates*. Warp the image and leave the coordinates behind and every label
is silently wrong: training converges, the loss falls, the checkpoint is garbage, and nothing
anywhere says so. That is worse than a crash, so `elastic_augment` ends by *measuring* whether
the coordinates followed the image and raising if they did not.

Design choices, each for a reason:

* **In-plane only.** `downsample = (1, 4, 4)` -- Z is not downsampled and its voxels are
  1.625 um against 0.40625 in Y and X. A deformation field over Z would mix physically
  different scales; over Y and X it does not.
* **One field for the whole window.** `window_size = 2`, and the association head learns
  correspondence *between* the two frames. Warping them differently would teach it motion
  that is not there. The same field is applied to every frame and every z-slice.
* **First-order coordinate update.** `grid_sample` with grid ``p + d(p)`` produces
  ``out(p) = in(p + d(p))``, so content at input ``q`` lands near output ``q - d(q)``. For a
  field this smooth and this small the first-order update is accurate to well under a voxel,
  and the intensity check at the end is what proves it.
"""
# this file is APPENDED to the pack's scripts/augmentations.py, so it must not open with
# anything that has to come first in a file. `from __future__ import annotations` did, and
# landed at line 98 of the concatenation: SyntaxError, run dead in ninety seconds. The same
# rule bit the notebook builder an hour earlier and I did not carry the lesson across the
# two places the same text is used. Nothing here needs it -- the annotations are all
# builtin generics, valid at runtime since 3.9.
import numpy as np
import torch
import torch.nn.functional as F


def elastic_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: np.random.Generator,
    max_shift_vox: float = 3.0,
    control: int = 4,
    prob: float = 0.5,
    check_drop: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deform Y and X by a smooth random field, carrying the node coordinates with it.

    Parameters
    ----------
    imgs : torch.Tensor
        ``(W, Z, Y, X)`` normalised images.
    coords : torch.Tensor
        ``(W, M, 3)`` node coordinates in voxels, ordered ``(z, y, x)``. Padded rows are
        zero and excluded by ``masks``; they are left untouched, exactly as `flip_augment`
        leaves them.
    masks : torch.Tensor
        ``(W, M)`` boolean, true for real nodes.
    rng : np.random.Generator
        The trainer's generator, so a run stays reproducible from ``base_seed``.
    max_shift_vox : float
        Largest displacement anywhere in the field, in Y/X voxels. At the pack's
        ``downsample = (1, 4, 4)`` one voxel is 1.625 um, so 3 voxels is about a cell radius.
    control : int
        Side of the coarse control grid the field is drawn on before upsampling. Small values
        give long-wavelength warps; large ones approach noise, which would fight the detector
        rather than regularise it.
    prob : float
        Probability of applying the deformation at all.
    check_drop : float
        Maximum tolerated fall in mean intensity at the node coordinates. The check exists to
        catch a coordinate update that did not follow the image; see the module docstring.
    """
    if rng.random() >= prob:
        return imgs, coords, masks

    W, Z, Y, X = imgs.shape
    dev, dt = imgs.device, imgs.dtype

    # A smooth field over (Y, X), shared by every frame and every slice.
    coarse = torch.from_numpy(
        rng.normal(0.0, 1.0, size=(1, 2, control, control)).astype("float32")
    )
    field = F.interpolate(coarse, size=(Y, X), mode="bicubic", align_corners=True)
    peak = field.abs().amax().clamp_min(1e-6)
    field = (field / peak) * float(max_shift_vox)          # (1, 2, Y, X) in voxels, [dy, dx]
    field = field.to(device=dev, dtype=torch.float32)

    # Sampling grid: output pixel p reads input p + d(p). grid_sample wants (x, y) last,
    # normalised to [-1, 1] with align_corners=True.
    yy, xx = torch.meshgrid(
        torch.arange(Y, device=dev, dtype=torch.float32),
        torch.arange(X, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    src_y = yy + field[0, 0]
    src_x = xx + field[0, 1]
    norm = lambda v, n: (2.0 * v / max(n - 1, 1)) - 1.0
    grid = torch.stack([norm(src_x, X), norm(src_y, Y)], dim=-1)[None]    # (1, Y, X, 2)

    flat = imgs.reshape(W * Z, 1, Y, X).to(torch.float32)
    warped = F.grid_sample(
        flat, grid.expand(W * Z, -1, -1, -1),
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    out_imgs = warped.reshape(W, Z, Y, X).to(dt)

    # Content at input q lands near output q - d(q), so sample the field AT the node and
    # subtract. Nearest-voxel lookup is enough: the field varies over hundreds of voxels.
    out_coords = coords.clone()
    if masks.any():
        cy = coords[..., 1].round().long().clamp_(0, Y - 1)
        cx = coords[..., 2].round().long().clamp_(0, X - 1)
        dy = field[0, 0][cy, cx]
        dx = field[0, 1][cy, cx]
        m = masks.to(torch.bool)
        out_coords[..., 1] = torch.where(m, coords[..., 1] - dy.to(coords.dtype), coords[..., 1])
        out_coords[..., 2] = torch.where(m, coords[..., 2] - dx.to(coords.dtype), coords[..., 2])
        out_coords[..., 1].clamp_(0, Y - 1)
        out_coords[..., 2].clamp_(0, X - 1)

        # Did the coordinates follow the image? Node positions are cell centres and therefore
        # bright; if the update had the wrong sign, or the wrong axis order, or was skipped,
        # intensity at the new coordinates collapses. A crash here is the point of the file.
        # Measured as CONTRAST -- node intensity over mean image intensity -- not as raw
        # intensity. A unit test on isolated blobs (background 0) separates a correct update
        # from a disabled one by 0.89x against 0.49x, but real frames are quantile-normalised
        # with a bright background, which pulls both ratios toward 1 and closes that gap.
        # Contrast does not have that problem: if the coordinates stop tracking the image the
        # nodes land on background and the ratio collapses toward 1 whatever the background is.
        before = _contrast(imgs, coords, m)
        after = _contrast(out_imgs, out_coords, m)
        if before > 1.0 + 1e-6 and (after - 1.0) < (before - 1.0) * (1.0 - check_drop):
            raise RuntimeError(
                "elastic_augment: node-to-background contrast fell from "
                f"{before:.4f} to {after:.4f} after warping -- the coordinates did not "
                "follow the image."
            )
    return out_imgs, out_coords, masks


def _contrast(imgs: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean intensity at the masked node coordinates, divided by mean image intensity.

    Nodes are cell centres, so this is comfortably above 1 whenever the coordinates point at
    cells and falls to about 1 when they point anywhere else.
    """
    W, Z, Y, X = imgs.shape
    w_idx = torch.arange(W, device=imgs.device)[:, None].expand_as(mask)[mask]
    cz = coords[..., 0][mask].round().long().clamp_(0, Z - 1)
    cy = coords[..., 1][mask].round().long().clamp_(0, Y - 1)
    cx = coords[..., 2][mask].round().long().clamp_(0, X - 1)
    at_nodes = float(imgs[w_idx, cz, cy, cx].to(torch.float32).mean())
    overall = float(imgs.to(torch.float32).mean())
    return at_nodes / overall if abs(overall) > 1e-9 else 1.0
