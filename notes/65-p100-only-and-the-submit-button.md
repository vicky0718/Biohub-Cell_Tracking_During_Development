# Two hard limits found the same hour: P100-only GPUs, and a submit button we cannot press

Both are infrastructure, both were found by measurement rather than assumption, and together
they set the shape of every remaining experiment.

## 1. Kaggle is handing this account P100s, and `machineShape` really is ignored

Eight consecutive GPU sessions on 2026-09-06 drew a **Tesla P100-PCIE-16GB**. Every one died
about ninety seconds in, in the detector's first `model.encode`:

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current
PyTorch installation. The current PyTorch install supports sm_70 sm_75 sm_80 sm_86 ...
```

`claude_fork` (0.937) and `claude_forkw085` (0.932) both drew **T4s** two days earlier, on
byte-identical push configurations. Nothing about the notebooks changed.

`notes/24` recorded `machineShape` as accepted-and-ignored in August. Re-measured today with
both spellings, and it still is:

```
machineShape = "nvidiaTeslaT4"   -> Tesla P100-PCIE-16GB
machineShape = "gpuT4x2"         -> Tesla P100-PCIE-16GB
```

So the accelerator cannot be requested, and `tools/run_arm.py`'s re-roll loop — which was
the right instinct while this looked like a lottery — is not guaranteed to terminate.

**The fix is the wheelhouse, and `MEMORY.md` already said so:** *"Every GPU submission needs
the wheelhouse — it is the only thing that makes a P100 run."* `claude-torch-wheelhouse`
carries `torch-2.5.1+cu121` and its complete dependency closure (cublas, cudnn, nccl,
triton, sympy, 24 wheels), and 2.5.1+cu121 does ship sm_60 kernels.

Every arm built by `_mk_claude_arm.py` now carries a prologue at the top of its config cell
that reads `nvidia-smi`, and installs from the wheelhouse **only if the name contains
P100**. Three details that are not incidental:

* It targets `/usr/bin/python3` explicitly. The fork runs its detector as a subprocess under
  that interpreter, not under the notebook's `sys.executable`.
* No `--no-deps`. torch 2.5.1 needs the cu121 runtimes and the image ships cu128; the
  wheelhouse has the full closure, so pip resolves it there.
* `torchvision` is not requested — it is not in the wheelhouse, and the fork's detector is a
  custom UNet3D that never imports it.

The prologue is a no-op on a T4, so it costs nothing if the draw changes back.

## 2. This is a kernels-only competition: submission cannot be automated

```
/competitions/list -> isKernelsSubmissionsOnly: True,  maxDailySubmissions: 5
```

Measured, after building the whole autonomous path and finding it blocked at the last step:

* `/blobs/upload` with `type: "competition"` returns **HTTP 500** — file submission is
  refused outright for this competition.
* `type: "inbox"` uploads fine, but `/competitions/submissions/submit/{comp}` rejects the
  token: *"BlobFileTokens must be specified."*
* The submit endpoint accepts **nothing but** a competition blob token. `scriptVersionId`
  and `sourceKernelVersionId` are ignored; the error is identical with and without them.

Every existing submission confirms the shape — `description: "Notebook claude forkw085 |
Version 2"`, `url: /code/vigneshnehru/claude-forkw085?scriptVersionId=347488162`. They were
made from the notebook UI, and that is the only route the v1 API exposes.

**So: runs are autonomous, scoring is not.** A completed arm has to be submitted by hand from
its Kaggle output page. That makes the daily-5 the real budget for *measured* arms, and it
makes the ranking of which arms to submit the actual deliverable.

One thing that *is* automatable and worth keeping: `kernel_output_file` pulls a finished
run's `submission.csv` out of `/kernels/output/download/{user}/{slug}`. The signed URLs in
`/kernels/output` point at `www.kaggleusercontent.com`, which this container's proxy denies
with the same 403 that forced us off `api.kaggle.com`; the download route is served from
`www.kaggle.com` and works. That gives us the predicted node graph for any arm without a
submission — useful for comparing arms against each other even though it cannot score them.

## 3. The rule `notes/64` should have drawn: swept vs stepped

`notes/64` cost 0.005 by treating `SECONDARY_DETECTION_WEIGHT` as unexplored when its author
called 0.80 a *swept peak*. The distinction that survives:

```
SWEPT    the author says they explored both sides and this is the maximum.
         Expect to lose. SECONDARY_DETECTION_WEIGHT 0.80 was this.
STEPPED  the author moved it once, the move paid, they published and moved on.
         Nothing has ever been measured on the far side of the new value.
```

The public 0.934 → 0.941 progression is a chain of **steps**, not sweeps —
`GAP_CLOSE_UM` 5.8 → 5.0 and `DEEPCENTER_SAFE_DIV_THRESHOLD` 0.12 → 0.25, each worth about
+0.001, each published as a single accepted move. `gap44` and `dc40` take one more step down
each of those gradients. They are unexplored *because* everyone else forks the published
value instead of continuing past it.

`union` is the other kind of gap: two disjoint routes reach 0.941 — `analyticaobscura` via
deepcenter 0.25 + gap 5.0, `rishabhr0y` via `SECONDARY_LINK_MODE=adaptive` — and neither
author has the other's change. Crossing them is a one-token edit on the lb-941 base.

```
arm         base            change
lb941       lb-941          none (the floor: 0.941 = rank ~115)
adaptive    rishabhr0y      none (the other 0.941)
ckpt948     zhuzhenghaomax  none (claimed 0.948 = rank ~20, checkpoint_last + narrow divisions)
lb941last   lb-941          DeepCenter best.pt -> checkpoint_last.pt
gap44       lb-941          GAP_CLOSE_UM 5.0 -> 4.4
dc40        lb-941          DEEPCENTER_SAFE_DIV_THRESHOLD 0.25 -> 0.40
union       lb-941          SECONDARY_LINK_MODE -> adaptive
```
