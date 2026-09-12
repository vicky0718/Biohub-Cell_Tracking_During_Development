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
    check_margin: float = 0.05,
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
    check_margin : float
        How much better the updated coordinates must score than the un-updated ones on the
        warped image. A skipped update scores exactly the same; a correct one is far ahead.
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

        # Did the coordinates follow the image? Compare the warped image sampled at the
        # UPDATED coordinates against the same warped image sampled at the ORIGINAL ones.
        #
        # The first version compared before-warp against after-warp and fired on real data at
        # contrast 108.3 -> 63.1, a false positive: at downsample (1, 4, 4) a cell is barely
        # a voxel across in Y and X, and bilinear resampling of a one-voxel peak loses ~40%
        # of its amplitude no matter how right the coordinates are. My synthetic test used
        # sigma=2 blobs and lost only 13%, which is exactly why it passed.
        #
        # Measuring both terms on the SAME warped image removes interpolation loss from the
        # comparison entirely. A skipped update makes the two identical by construction; a
        # wrong-signed one makes the updated coordinates worse than the originals. Only a
        # correct update is clearly better, and only when the field actually moved something,
        # which is why the check is gated on a real displacement.
        # Gated on the displacement the field INTENDED at the nodes, not on the one the
        # coordinates actually moved. Gating on the realised shift is circular: an update
        # that never happened leaves the shift at zero, the gate never opens, and the check
        # passes -- which is what the first rewrite did, silently, in six of six test cases.
        intended = float(torch.maximum(dy[m].abs().amax(), dx[m].abs().amax()))
        realised = float((out_coords - coords).abs().amax())
        if intended > 0.5:
            if realised < intended * 0.5:
                raise RuntimeError(
                    f"elastic_augment: the field moves nodes by up to {intended:.2f} voxels "
                    f"but the coordinates moved {realised:.2f} -- the update did not happen."
                )
            good = _contrast(out_imgs, out_coords, m)
            stale = _contrast(out_imgs, coords, m)
            if good < stale * (1.0 + check_margin):
                raise RuntimeError(
                    f"elastic_augment: after a {realised:.2f} voxel warp the updated "
                    f"coordinates score {good:.3f} against {stale:.3f} for the original "
                    "ones -- the coordinates did not follow the image."
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
