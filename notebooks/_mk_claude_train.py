"""Build a Kaggle notebook that FINE-TUNES the public checkpoint instead of forking it.

    python notebooks/_mk_claude_train.py            # -> claude_train_elastic.ipynb

`notes/75` is why. We are rank 589 at 0.945 and 252 teams sit on exactly 0.947, which is one
public notebook forked 252 times. Our measured mechanism gains are +0.001 each and the
plateau moves faster than they arrive, so the remaining arms are all forks of a lineage that
is moving away from us. Tang (rank 5, 0.961): *"the current ckpt has kind of hit a wall...
most of the gains come from model improvements."*

Three facts make the model reachable, all collected from our own kernel output:

* `tracking_repo/scripts/train_unet_transformer.py` ships **in the support pack**, with
  `--unet-weights` ("loaded with strict=False") and `--max-iters`. Fine-tuning is a flag.
* Kaggle mounts all 199 training movies and their `.geff` ground truth in any kernel listing
  the competition as a source. Nothing to download.
* `load_dataset_windows` opens zarrs with `load_image=False` and keeps "no image tensor", so
  the loader is metadata-only and 169 movies fit in RAM. Images stream during training.

This notebook is assembled rather than written: everything up to and including the pack's own
primary-weight checksum is taken **verbatim** from `claude_arm_tta946_source.json`, because
that prefix already resolves artifacts, installs the offline wheels and materialises the repo,
and it verifies itself. Our code starts after that, which also means the repo's integrity
check runs *before* we edit `augmentations.py` rather than after.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K          # noqa: E402
from _mk_claude_arm import WHEELHOUSE, TTA946_SOURCES   # noqa: E402

HERE = Path(__file__).resolve().parent

# The cut. Everything before this string is setup that verifies itself; everything after it
# in the source notebook is inference machinery we do not want.
CUT = "_deepcenter_candidate_strings"

ELASTIC = (HERE / "elastic_augment.py").read_text()

# The augmentation is APPENDED to the pack's `scripts/augmentations.py`, so anything in it
# that must come first in a file breaks the concatenation. A `from __future__` import there
# killed a run ninety seconds in, an hour after the identical rule broke the notebook
# assembly below. Checked here so it cannot happen a third time.
for _line in ELASTIC.splitlines():
    if _line.startswith("from __future__"):
        raise SystemExit("REFUSING TO BUILD — elastic_augment.py opens with a __future__ "
                         "import; it is appended to another file and cannot.")

TRAIN = '''

# ==========================================================================
# CLAUDE FINE-TUNE -- everything above this line is the support pack's own
# setup, taken verbatim from the 0.946 notebook and left unmodified.
# ==========================================================================
import random as _rnd

TRAIN_DIR = COMP_DIR / 'train'
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f'No training data mounted at {TRAIN_DIR}')
_stems = sorted(p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith('.zarr'))
print(f'Training movies mounted: {len(_stems)}', flush=True)
if len(_stems) < 50:
    raise RuntimeError(f'expected ~199 training movies, found {len(_stems)}')

# A holdout the checkpoint we are fine-tuning has ALREADY SEEN. notes/72 section 3: the
# public checkpoint trained on all 199 movies, so this comparison is biased in ITS favour --
# which is what makes a win on it real rather than a training-set artifact. It is the first
# honest validation signal this project has had.
_HOLDOUT = int(os.environ.get('BIOHUB_TRAIN_HOLDOUT', '30'))
_shuffled = list(_stems)
_rnd.Random(20260912).shuffle(_shuffled)
_val, _tr = sorted(_shuffled[:_HOLDOUT]), sorted(_shuffled[_HOLDOUT:])
_splits = REPO_DIR / 'claude_finetune_splits.json'
_splits.write_text(json.dumps([{'split': 0, 'train': _tr, 'test': _val}], indent=2))
print(f'Fold 0: {len(_tr)} train / {len(_val)} holdout', flush=True)
print('holdout:', _val[:6], '...', flush=True)

# ---- the augmentation the pack does not have ----------------------------
_AUG = REPO_DIR / 'scripts' / 'augmentations.py'
_aug_src = _AUG.read_text()
if 'def elastic_augment' in _aug_src:
    raise RuntimeError('augmentations.py already defines elastic_augment')
_AUG.write_text(_aug_src + "\\n\\n" + ELASTIC_SOURCE)

_T = REPO_DIR / 'scripts' / 'train_unet_transformer.py'
_t = _T.read_text()
_edits = [
    ('from augmentations import brightness_augment, flip_augment\\n',
     'from augmentations import brightness_augment, flip_augment, elastic_augment\\n'),
    ('DEFAULT_AUGMENTATIONS = [brightness_augment, flip_augment]',
     'DEFAULT_AUGMENTATIONS = [brightness_augment, flip_augment, elastic_augment]'),
    # `WEIGHTS_PATH` comes from `dataspec` and lands inside the materialised repo, whose
    # `weights/` is a link into the read-only `/kaggle/input` mount -- the pack's own
    # checkpoints live there, so it never needed to be writable for inference. Training is
    # the first thing that writes to it: "OSError: [Errno 30] Read-only file system", after
    # the run had already loaded 169 movies. Redirect it somewhere we own.
    ('from dataspec import WEIGHTS_PATH\\n',
     'from dataspec import WEIGHTS_PATH  # noqa: F401\\n'
     'WEIGHTS_PATH = Path("/kaggle/working/claude_weights")\\n'
     'WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)\\n'),
    # `--unet-weights` restores the BACKBONE ONLY, and from this checkpoint it restores
    # nothing at all. The trainer does
    #
    #     unet = TemporalUNet3D(...)
    #     unet.load_state_dict(torch.load(unet_weights), strict=False)
    #
    # while `edge_predictor_best.pth` was saved from the whole `UNetNodeTransformer`, so its
    # keys are `unet.enc...`, `node_transformer...`, `edge_head...`. Loaded into a bare
    # TemporalUNet3D every one of those is "unexpected" and every backbone parameter is
    # "missing": strict=False turns a total mismatch into a silent no-op, and the run becomes
    # a from-scratch training that looks exactly like a fine-tune. The flag is not wrong --
    # it is for a UNet-only pretrain -- it is the wrong flag for this file.
    #
    # Restore the FULL model instead, association head included, and refuse to continue if
    # the restore did not actually take.
    ("""    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)
""",
     """    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)

    if unet_weights is not None:
        _full = torch.load(unet_weights, map_location="cpu", weights_only=True)
        _missing, _unexpected = model.load_state_dict(_full, strict=False)
        _restored = len(_full) - len(_unexpected)
        print(f"  FULL restore from {unet_weights}: {_restored}/{len(_full)} tensors "
              f"loaded, {len(_missing)} left at init, {len(_unexpected)} unused",
              flush=True)

        if _restored < len(_full) // 2 and os.environ.get(
                "BIOHUB_TRAIN_ALLOW_SCRATCH", "0") == "0":
            raise RuntimeError(
                f"only {_restored} of {len(_full)} checkpoint tensors matched the model -- "
                "this would train from scratch while looking like a fine-tune"
            )
"""),
    # The most valuable line in the run: score the checkpoint we are about to fine-tune on
    # the holdout BEFORE touching it, so every epoch after is measured against it.
    ('    for epoch in pbar:\\n        t0 = time.monotonic()\\n',
     '    _b_loss, _b_acc, _b_recall = evaluate(model, test_loader, device,'
     ' pool_kernel_um=pool_kernel_um)\\n'
     '    print(f"  BASELINE epoch -1 (public checkpoint, no training) | "\\n'
     '          f"acc={_b_acc:.4f} | recall={_b_recall:.4f} | score={_b_acc * _b_recall:.4f}",\\n'
     '          flush=True)\\n'
     '    best_score = _b_acc * _b_recall\\n'
     '    for epoch in pbar:\\n        t0 = time.monotonic()\\n'),
]
for _old, _new in _edits:
    if _t.count(_old) != 1:
        raise RuntimeError(f'trainer patch matched {_t.count(_old)}x, expected 1: {_old[:60]!r}')
    _t = _t.replace(_old, _new, 1)
compile(_t, str(_T), 'exec')
_T.write_text(_t)

# ---- make the temporal attention launchable on an sm_60 card -------------
# `_TemporalAttention.forward` flattens to (B * S, T, C) with S the whole downsampled
# volume, so attention runs over millions of length-T sequences in ONE call. cuBLAS batched
# GEMM takes its batch count as a CUDA grid dimension capped at 65535, and past that a P100
# answers `invalid configuration argument` -- which is what killed the run after it had
# already loaded all 169 movies. Slicing that batch is mathematically identical: every
# sequence attends only across its own T timesteps, so there is nothing between slices to
# lose. T is 2, so the loop costs seconds an epoch and lowers peak memory as well.
_TU = REPO_DIR / 'src' / 'biohub_tracking' / 'models' / 'temporal_unet.py'
_tu = _TU.read_text()
_attn_old = ("        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)\\n"
             "        h = self.norm(h)\\n"
             "        h, _ = self.attn(h, h, h, need_weights=False)\\n")
_attn_new = ("        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)\\n"
             "        h = self.norm(h)\\n"
             "        _chunk = int(os.environ.get('BIOHUB_ATTN_CHUNK', '32768'))\\n"
             "\\n"
             "        if _chunk <= 0 or h.shape[0] <= _chunk:\\n"
             "            h, _ = self.attn(h, h, h, need_weights=False)\\n"
             "        else:\\n"
             "            _parts = []\\n"
             "\\n"
             "            for _i in range(0, h.shape[0], _chunk):\\n"
             "                _p = h[_i:_i + _chunk]\\n"
             "                _parts.append(self.attn(_p, _p, _p, need_weights=False)[0])\\n"
             "            h = torch.cat(_parts, dim=0)\\n"
             "            del _parts\\n")
if _tu.count(_attn_old) != 1:
    raise RuntimeError(f'attention patch matched {_tu.count(_attn_old)}x, expected 1')
_tu = _tu.replace(_attn_old, _attn_new, 1).replace('import math\\n', 'import math\\nimport os\\n', 1)
compile(_tu, str(_TU), 'exec')
_TU.write_text(_tu)
print('temporal attention chunked at', os.environ.get('BIOHUB_ATTN_CHUNK', '32768'),
      'sequences per launch', flush=True)
print('elastic_augment installed; baseline eval added; best_score seeded from the baseline '
      'so nothing worse than the public checkpoint can be saved', flush=True)

# ---- train ---------------------------------------------------------------
_out_method = os.environ.get('BIOHUB_TRAIN_METHOD', 'unet_transformer_claude_elastic')
_cmd = [
    '/usr/bin/python3', 'scripts/train_unet_transformer.py',
    '--data-dir', str(TRAIN_DIR),
    '--splits', 'claude_finetune_splits.json', '--split', '0',
    '--method', _out_method,
    '--unet-weights', str(REPO_DIR / WEIGHTS_RELATIVE),
    '--epochs', os.environ.get('BIOHUB_TRAIN_EPOCHS', '30'),
    '--lr', os.environ.get('BIOHUB_TRAIN_LR', '3e-5'),
    # 2, not the pack's 8. Their 8 died on a P100 in the first forward pass with
    # `CUDA error: invalid configuration argument` -- a kernel launch whose grid exceeds
    # what sm_60 accepts, not an out-of-memory. Inference on the same card runs at
    # --unet-batch-size 4 and training holds activations for the backward pass on top, so
    # 2 is the conservative read of the one data point we have.
    '--batch-size', os.environ.get('BIOHUB_TRAIN_BATCH', '2'),
    '--max-iters', os.environ.get('BIOHUB_TRAIN_MAX_ITERS', '300'),
    '--num-workers', os.environ.get('BIOHUB_TRAIN_WORKERS', '2'),
    # Capacity. The defaults are the pretrained shapes, and they are the defaults for a
    # reason: change either one and the checkpoint's tensors stop matching, the full restore
    # refuses (BIOHUB_TRAIN_ALLOW_SCRATCH overrides), and the run becomes a from-scratch
    # training that has to beat a 400-epoch model inside what is left of the quota.
    #
    # BIOHUB_TRAIN_DOWNSAMPLE is the one capacity knob that costs nothing: convolutions do
    # not care about spatial extent, so '1,2,2' keeps every pretrained weight and gives the
    # detector 4x the resolution in Y and X. notes/04 measured detection as essentially the
    # whole contest. It costs ~4x the compute and memory, not parameters.
    '--unet-out-channels', os.environ.get('BIOHUB_TRAIN_OUT_CH', '32'),
    '--unet-layers', os.environ.get('BIOHUB_TRAIN_LAYERS', '32,64,128'),
    '--downsample', os.environ.get('BIOHUB_TRAIN_DOWNSAMPLE', '1,4,4'),
    '--window-size', '2', '--pool-kernel-um', '5.0',
    '--single-gpu',
]
print(' '.join(_cmd), flush=True)
_rc = subprocess.run(_cmd, cwd=REPO_DIR, env={**os.environ, 'PYTHONPATH': 'src'})
print('training rc =', _rc.returncode, flush=True)
if _rc.returncode != 0:
    raise RuntimeError(f'training failed rc={_rc.returncode}')

# ---- collect --------------------------------------------------------------
_weights_out = Path('/kaggle/working/claude_finetuned')
_weights_out.mkdir(parents=True, exist_ok=True)
_src_dir = Path('/kaggle/working/claude_weights') / _out_method / 'split_0'
for _f in sorted(_src_dir.iterdir()):
    shutil.copy2(_f, _weights_out / _f.name)
    print(f'  saved {_f.name} ({_f.stat().st_size:,} bytes)', flush=True)
(_weights_out / 'claude_finetune_manifest.json').write_text(json.dumps({
    'base_weights_sha256': _primary_actual_sha256,
    'holdout': _val, 'n_train': len(_tr),
    'augmentations': ['brightness_augment', 'flip_augment', 'elastic_augment'],
    'command': _cmd[1:],
}, indent=2))
print('DONE -- fine-tuned weights are in /kaggle/working/claude_finetuned', flush=True)
'''


def build() -> int:
    prov = HERE / "claude_arm_tta946_source.json"
    rec = json.loads(prov.read_text())
    nb = json.loads(rec["source"])
    cells = nb["cells"]

    idx = next(i for i, c in enumerate(cells)
               if c.get("cell_type") == "code" and "os.environ" in "".join(c["source"]))
    body = "".join(cells[idx]["source"])
    cut = body.find(CUT)
    if cut < 0:
        print(f"REFUSING TO WRITE — cut marker {CUT!r} not found")
        return 1
    prefix = body[:cut]
    if "Support repo manifest checksum mismatch" not in prefix:
        print("REFUSING TO WRITE — the prefix does not include the repo integrity check")
        return 1

    # The source notebook opens with `from __future__ import annotations`, and prepending the
    # wheelhouse above it makes the file uncompilable even though IPython tolerates it at
    # runtime (every arm we have shipped does exactly this and runs). Hoist it instead of
    # dropping the compile check -- the check is the only thing standing between a builder
    # typo and a wasted twelve-hour GPU run.
    future = "from __future__ import annotations\n"
    head = future if future in prefix else ""
    prefix = prefix.replace(future, "", 1)

    # Refuse a P100 in the first seconds rather than three minutes in.
    #
    # The traceback is `activation.py` line 1308 -- `MultiheadAttention.forward` -- not the
    # convolutions. `temporal_unet.py` reshapes to `(B * S, T, C)` where S is the whole
    # downsampled volume, so the attention batch is tens of millions of sequences, and on
    # sm_60 that launch exceeds a CUDA grid limit: `invalid configuration argument`. It is
    # not out-of-memory and a smaller batch may not be enough; the pack's authors trained
    # this on sm_80-class cards where the attention path differs.
    #
    # `machineShape` is accepted and ignored on push (`notes/65`), so the accelerator can
    # only be re-rolled. That is cheap if the run dies immediately and expensive if it dies
    # after materialising the repo and loading 169 movies, which is what just happened.
    # `run_arm.py` retries on this message.
    guard = (
        'import subprocess as _gsp\n'
        'try:\n'
        '    _gpu = _gsp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],\n'
        '                    capture_output=True, text=True, timeout=60).stdout.strip()\n'
        'except Exception:\n'
        '    _gpu = ""\n'
        'print("accelerator:", _gpu, flush=True)\n'
        # No longer fatal. Six consecutive P100 draws said the lottery is not winnable on
        # this account, so the model is made trainable on the card we actually get instead.
        'if "P100" in _gpu:\n'
        '    print("P100 -- temporal attention will be chunked to stay inside the sm_60 "\n'
        '          "batched-GEMM limit", flush=True)\n'
    )

    source = (head + WHEELHOUSE + guard
              + prefix
              + "\n# The elastic augmentation, verbatim from notebooks/elastic_augment.py\n"
              + "ELASTIC_SOURCE = " + repr(ELASTIC) + "\n"
              + TRAIN)
    compile(source, "claude_train_elastic", "exec")

    nb["cells"] = [cells[0], {"cell_type": "code", "execution_count": None,
                              "metadata": {}, "outputs": [],
                              "source": source.splitlines(keepends=True)}]
    out = HERE / "claude_train_elastic.ipynb"
    out.write_text(json.dumps(nb, indent=1))

    (HERE / "claude_train_elastic_push.json").write_text(json.dumps({
        "slug": "claude-train-elastic", "title": "Claude train elastic",
        "notebook": str(out), "dataset_sources": TTA946_SOURCES,
        "competition_sources": rec["competitionDataSources"],
        "kernel_sources": [f"{K.username()}/claude-torch-wheelhouse"],
        "enable_gpu": True, "enable_internet": False,
    }, indent=1))
    print(f"wrote {out.name}: {len(source):,} chars "
          f"({len(prefix):,} from the pack, {len(TRAIN):,} ours)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
