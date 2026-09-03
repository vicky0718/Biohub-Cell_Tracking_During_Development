# Prebuilt detectors: what exists, what it scores, and why the 12-hour limit decides

Prompted by a direct question. The honest answer has three parts: one model is genuinely
strong and newly available, it **cannot be the submission-time detector**, and the use that
survives is not the obvious one.

---

## 1. What has already been tried, and closed

```
spotiflow (r35's domain fine-tune, 35M params)   recall saturates 0.547 at ANY threshold   notes/47
our own UNet from scratch, 5 runs                CV ceiling 0.649                          notes/23
temporal-unet3d-seed314159 (2nd pack model)      +0.0026, t=0.63, unresolved at n=12       notes/42
```

`notes/47` also measured the pack's own detector as the most selective thing we had —
**0.996 node recall at 24,605 nodes** — beating a 35M-parameter domain fine-tune at every
matched node count. That finding is what made "swap the detector" look closed.

**Training from scratch is not closed for the reason `MEMORY.md` gave.** `notes/33` §5
already corrected the 725 h figure: the forum's measured rate is ~50 min/epoch on a T4, so
400 epochs is **~333 h against ~140 h of remaining Kaggle quota**. Too big, but not by the
order of magnitude that was recorded. Fine-tuning, which is what was actually asked about,
is hours rather than hundreds of hours.

## 2. What is newly available — and none of it appears in 52 notes

`FOCUS-3D`, `HOCT`, `cellpose`, `stardist` appear **nowhere** in this project's notes.
Only `Trackastra` does, once, in `notes/03`, before anything was built. Meanwhile
competitors have packaged all of them as ordinary Kaggle datasets — mountable exactly like
the pack, with no scraping:

```
qiweiyin/focus3d-nuclei-runtime                FOCUS-3D code + model_final_nuclei.pth (4.5 GB)
hiroyasuokuno/hoct-general-v0-weights          HOCT (royerlab -- the hosts' own lab)
serudeeven/biohub-hoct-codebase                HOCT codebase, competition-specific
rudispresence/biohub-stabledet-hoct-runtime    someone's HOCT integration
subinium/biohub-trackastra-public-weights-mirror
rudispresence/biohub-embryo-stardist-bayesian-assets
anatolykabcnd/cells-stardist3d-bright-holdout-model
```

The `biohub-` prefixes are the tell: these are purpose-built for this competition.

`FOCUS-3D` is the one with a public, measured result. hengck23 (GRANDMASTER, thread 738217,
17 votes) posted it as *"one of the best 3d cell segmentation"*, and Qiwei (MASTER)
published a working notebook — `qiweiyin/focus3d-nuclei-physical-pp-submit`, whose only
dataset sources are the pack and the FOCUS-3D runtime. Its own reported validation on train
video `44b6_12dfb391`:

```
66,351 raw detections -> 60,625 nodes / 58,478 edges / 227 divisions after PP
estimated truth size 58,672 nodes  ->  detection rate ~103%
sparse-truth check (788 annotated nodes): node recall 98.5%, annotated-edge recall 95.2%
```

**Read that against ours carefully — it is one video against our 24-dataset means, which is
the exact mismatch `notes/47` and `notes/50` were both about.** Not a like-for-like
comparison. But the *operating point* is qualitatively different in a way that does not
depend on the aggregate:

```
              detection rate vs N_est      divisions
FOCUS-3D            ~103%   (ratio +0.03)      227 on ONE video
ours                  87%   (ratio -0.129)   1,443 across 24
```

They sit *at* budget; `notes/52` measured us at 12.9% under it.

## 3. The 12-hour limit rules out the obvious use

```
submission limit (notes/07)              720 min
our whole pipeline, ~200 test datasets    ~52 min
FOCUS-3D, Qiwei's own timing          ~12-15 min per video
FOCUS-3D x ~200 datasets                ~2,400-3,000 min
```

**Over the limit by 3-4x**, before linking. FOCUS-3D runs at 3.6 s/frame on a V100 and an
estimated 7-9 s/frame on the T4 we would actually get. It is roughly **50x slower per
dataset than our entire current chain**. Swapping it in as the submission detector does not
fit, and no amount of tuning closes a 3x gap.

The same arithmetic applies to any heavy instance-segmentation backbone, which is the real
reason the public field converged on the pack's small UNet.

## 4. The use that is not ruled out

hengck23's actual suggestion in that thread is not "use it as the detector":

> *"Once you have dense segmentation label, you can use many opensource tracker like hoct,
> itec, trackastra to make dense tracks for better training. Then you can do longer range
> tracking over window of 5 or 8 (instead of 2)."*

> *"FOCUS3d --> label --> augmentation (affine, elastic deform) to create window of T=2
> pairs. then you can train link transformer etc."*

**Run FOCUS-3D offline to make dense pseudo-labels, then train a fast model on them.** The
heavy model never runs at submission time, so the runtime constraint disappears entirely.

This targets the failure the forum keeps naming. The competition's own labels are sparse —
0.13% to 20.21% of nodes — and that sparsity is what breaks training:

> nusrati: *"peaks around epoch 5-10, then collapses… adj_ej 0.40 → 0.07 by epoch 120"*
> hengck23: *"this is the effect of sparse annotations. the unlabelled nodes change from
> noisy targets to negative targets after 10 epochs."*
> abcbcbc: *"the training data is too sparsely labeled… very difficult to train a
> detection model."*

Our own five from-scratch runs hit a CV 0.649 ceiling (`notes/23`) against exactly this.
Dense labels are the one input that changes the problem rather than the hyperparameters,
and `claude_zhpilot` was built for the ZebraHub version of this idea and never run.

## 5. The caution, from our own anatomy

Before spending on any detector, `notes/51`'s 583 `fn_detect` is not all detector capacity.
`claude_divsweep`'s arms separate it:

```
ctl/raw   fn_detect 238        pack default ILP weights
inc/raw   fn_detect 608        ratio0.4_2.0                 +370  <- our own tax
```

**Roughly 370 of the 583 is imposed by our ILP weights, not missed by the detector.** The
detector's own floor is ~238 edges, 1.72% of GT. So replacing the detector outright is worth
at most ~1.7%, while our own configuration accounts for more — and `claude_loosen` is
running now to test whether that 370 comes back by loosening the threshold.

**Order of operations: read `claude_loosen` first.** If the 370 is recoverable with a knob,
that is hours of work for most of the available gain, and a detector project is a large
investment for the smaller remainder.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.935 bronze    0.947 gold
detector swap at submission time: ruled out by the 12 h limit
dense pseudo-labels for training: open, untried, and aimed at the documented failure
```
