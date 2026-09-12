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
    '--batch-size', os.environ.get('BIOHUB_TRAIN_BATCH', '8'),
    '--max-iters', os.environ.get('BIOHUB_TRAIN_MAX_ITERS', '300'),
    '--num-workers', os.environ.get('BIOHUB_TRAIN_WORKERS', '2'),
    '--unet-out-channels', '32', '--unet-layers', '32,64,128',
    '--downsample', '1,4,4', '--window-size', '2', '--pool-kernel-um', '5.0',
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
_src_dir = REPO_DIR / 'weights' / _out_method / 'split_0'
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

    source = (head + WHEELHOUSE
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
