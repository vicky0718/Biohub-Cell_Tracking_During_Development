"""A small 3D UNet and the three candidate losses for 1/28 annotation.

Separate from `pipeline/detector.py` so that torch is imported only where it is needed:
training runs in a GPU notebook with internet on, while the submission path must stay
importable on whatever the scored rerun gives us.

Geometry (`notes/16` §5): at ``downsample=(1, 4, 4)`` a frame is **64 x 64 x 64 voxels,
isotropic at 1.625 um**, so a whole frame is one network input — no patching, no tiling,
no seams. Cell spacing is ~8 um (~5 voxels) and nuclei are 4-8 um (~2.5-5 voxels), so
three downsampling levels reach a receptive field well past a cell without crushing them.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3x3 convs with GroupNorm.

    GroupNorm rather than BatchNorm: batches are small (whole 64^3 volumes) and the
    foreground fraction swings by 20x between a sparse crop and a dense one, so batch
    statistics would be unstable in exactly the way that matters.
    """

    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, cout), cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, cout), cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """Voxelwise cell-centre logits. Input (B, 1, Z, Y, X) -> output (B, 1, Z, Y, X).

    Deliberately small. The contest is whether a learned appearance model beats a fixed
    band-pass at matched detection count, and that question is answerable at this size;
    capacity can come later once the loss is settled.
    """

    def __init__(self, base: int = 16, depth: int = 3, in_ch: int = 1):
        super().__init__()
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth + 1)]

        self.down = nn.ModuleList()
        c_prev = in_ch
        for c in chs[:-1]:
            self.down.append(ConvBlock(c_prev, c))
            c_prev = c
        self.bottom = ConvBlock(c_prev, chs[-1])

        self.up = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconv.append(nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2))
            self.up.append(ConvBlock(chs[i] * 2, chs[i]))
        self.head = nn.Conv3d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottom(x)
        for i, (upc, blk) in enumerate(zip(self.upconv, self.up)):
            x = upc(x)
            skip = skips[-(i + 1)]
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear",
                                  align_corners=False)
            x = blk(torch.cat([x, skip], 1))
        return self.head(x)


def naive_loss(logits, target, mask=None, prior=None, pos_weight: float = 50.0):
    """Variant A. Every unlabelled voxel is a negative.

    The control, and expected to be the weakest: `notes/04` §5b measured annotations as a
    uniform random ~1/28 sample, so this pushes 27 of every 28 real cells towards zero.
    `pos_weight` is still needed because positives are ~0.1% of voxels even before that.
    """
    return F.binary_cross_entropy_with_logits(
        logits, target.float(),
        pos_weight=torch.as_tensor(pos_weight, device=logits.device))


def masked_loss(logits, target, mask, prior=None, pos_weight: float = 50.0):
    """Variant B. Unlabelled-but-plausible voxels are dropped from the loss.

    Positives keep their gradient, clearly-empty voxels supply negatives, and the
    ambiguous middle contributes nothing. The model learns cell appearance from the
    annotated sample and applies it everywhere, which is exactly what a uniform random
    annotation permits.
    """
    per_voxel = F.binary_cross_entropy_with_logits(
        logits, target.float(), reduction="none",
        pos_weight=torch.as_tensor(pos_weight, device=logits.device))
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    return (per_voxel * m).sum() / denom


def pu_loss(logits, target, mask=None, prior=None, pos_weight: float = 50.0):
    """Variant C. Non-negative positive-unlabelled risk, with the prior we actually know.

    ``prior`` is P(annotated | cell-bearing voxel) for this sample, taken from the GEFF's
    own ``n_annotated / estimated_number_of_nodes`` -- so unlike most PU setups this is a
    measured quantity, not a hyperparameter.

    Risk decomposition: the unlabelled set is a mixture of positives and negatives in a
    known ratio, so the negative risk is estimated as
    ``R_u^- - pi * R_p^-`` and clamped at zero (the nnPU correction of Kiryo et al.);
    without the clamp the estimate goes negative and the network overfits into it.
    """
    pi = float(np.clip(prior if prior is not None else 0.05, 1e-4, 0.95))
    pos = target > 0
    unl = ~pos

    logp = F.logsigmoid(logits)          # log P(y=1)
    logn = F.logsigmoid(-logits)         # log P(y=0)

    n_p = pos.sum().clamp_min(1)
    n_u = unl.sum().clamp_min(1)
    r_p_pos = -(logp[pos]).sum() / n_p           # positives called positive
    r_p_neg = -(logn[pos]).sum() / n_p           # positives called negative
    r_u_neg = -(logn[unl]).sum() / n_u           # unlabelled called negative

    neg_risk = r_u_neg - pi * r_p_neg
    if neg_risk < 0:                              # nnPU: do not chase a negative estimate
        return -neg_risk
    return pi * r_p_pos + neg_risk


LOSSES = {"naive": naive_loss, "masked": masked_loss, "pu": pu_loss}


@torch.no_grad()
def predict_volume(model, vol: np.ndarray, device, amp: bool = True) -> np.ndarray:
    """Probability map for one input, as float32 numpy.

    Accepts either a single ``(Z, Y, X)`` volume or a stacked ``(C, Z, Y, X)`` temporal
    window, so the same function serves a single-frame model and a ``temporal_radius``
    one. The output is always ``(Z, Y, X)``: the network predicts the CENTRE frame's
    cell centres whatever it was given as context.
    """
    model.eval()
    x = torch.as_tensor(vol, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x[None, None]           # (Z,Y,X)   -> (1, 1, Z, Y, X)
    elif x.ndim == 4:
        x = x[None]                 # (C,Z,Y,X) -> (1, C, Z, Y, X)
    else:
        raise ValueError(f"expected a 3D volume or a 4D temporal window, got {x.shape}")
    in_ch = next(model.parameters()).shape[1]
    if x.shape[1] != in_ch:
        raise ValueError(
            f"model expects {in_ch} input channel(s) but was given {x.shape[1]}. A "
            "temporal model fed single frames (or the reverse) fails silently on shape-"
            "compatible input only when in_ch is 1, so this is checked rather than left "
            "to broadcast."
        )
    if amp and device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
    else:
        logits = model(x)
    return torch.sigmoid(logits.float())[0, 0].cpu().numpy()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["UNet3D", "ConvBlock", "LOSSES", "naive_loss", "masked_loss", "pu_loss",
           "predict_volume", "count_params"]
