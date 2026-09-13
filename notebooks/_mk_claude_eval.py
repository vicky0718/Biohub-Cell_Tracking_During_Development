"""Score an arm OFFLINE, on held-out training movies, with the official metric.

    python notebooks/_mk_claude_eval.py ttasec ttaz16

Why this exists. `harness/purescore.py` on the four visible clips went **0-for-4** as a
ranking instrument -- `notes/60`, `notes/64`, `ttaz16dom`, and finally `ttaz16`, which it
called **+0.0082** against an actual **-0.003**. It was not wrong because offline scoring is
impossible. It was wrong because those four clips are placeholders (`notes/66`) carrying
52-1,229 sparse annotations and **three** ground-truth divisions between them, and
`per_sample()` refuses predictions containing forks at all.

The fine-tune run established the two facts that fix this: Kaggle mounts **all 199 training
movies with their `.geff` ground truth** in any kernel listing the competition as a source,
and the support pack materialises the **official scorer** into the kernel at
`src/biohub_tracking/{metrics,division_metrics}.py`, with the offline wheels it needs.

So an arm can be scored on real movies, with dense GT, by the same code the leaderboard runs:

    evaluate(pred, gt, scale) -> EvaluationResult      edge + division TP/FP/FN
    node_recall(pred, gt)     -> float
    per_sample_metrics(er, n_total, node_recall) -> dict
    summarise(rows) -> {edge_jaccard, division_jaccard, adj_edge_jaccard, score}

**Validity, stated precisely.** The checkpoint trained on all 199 movies (`notes/72` §3), so
the absolute level is optimistic. Arms differ only in post-processing, so that optimism is
*identical across arms* and the **ordering** is what this instrument is for. Nothing gets
ranked on it until it reproduces a known ordering -- `ttasec` (LB 0.945) above `ttaz16`
(0.942) -- which is the acceptance test the old proxy would have failed four times.

**And it finally measures the other tenth of the metric.** `score = adj_edge_jaccard +
0.1 x division_jaccard`, and `division_jaccard` has never been measured on this pipeline.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import claude_kaggle_api as K          # noqa: E402
from _mk_claude_arm import ARMS, TTA946_SOURCES     # noqa: E402

HERE = Path(__file__).resolve().parent

# ---- edit 1: point the pipeline at held-out TRAINING movies -------------------------
# The whole notebook keys off TEST_DIR -- `list_test_stems()`, the splits file it writes, and
# the end-of-run guard that checks the submission covers exactly the datasets in it. Building
# a directory of symlinks and pointing TEST_DIR there means every one of those keeps working
# unmodified, which is the opposite of patching four places and hoping.
REDIRECT_OLD = "TEST_DIR = COMP_DIR / 'test'"
REDIRECT_NEW = """TEST_DIR = COMP_DIR / 'test'
# Baked in at build time: `kernels/push` carries no environment, so an env var read here
# would always be the default and eval mode would silently never engage -- notes/68's trap.
EVAL_MOVIES = __EVAL_MOVIES__
_eval_n = EVAL_MOVIES

if _eval_n:
    import random as _erand
    _esrc = COMP_DIR / 'train'

    if not _esrc.exists():
        raise FileNotFoundError(f'eval mode needs the training movies at {_esrc}')
    _estems = sorted(p.name[:-5] for p in _esrc.iterdir() if p.name.endswith('.zarr'))
    _eshuf = list(_estems)
    _erand.Random(20260912).shuffle(_eshuf)
    _epick = sorted(_eshuf[:_eval_n])
    _edst = Path('/kaggle/working/eval_subset')
    _edst.mkdir(parents = True, exist_ok = True)

    for _s in _epick:
        _link = _edst / f'{_s}.zarr'

        if not _link.exists():
            _link.symlink_to(_esrc / f'{_s}.zarr')
    TEST_DIR = _edst
    print(f'EVAL MODE: {len(_epick)} held-out training movies', flush = True)
    print('  ', _epick, flush = True)"""

# ---- edit 2: score what was written, with the official metric ------------------------
# Appended after the notebook's own verification block, so a scoring bug cannot cost the
# inference that produced the submission -- that is the expensive half, and this is seconds.
# Everything here is wrapped: the run is already a success once submission.csv exists.
SCORE = '''

# ==========================================================================
# OFFICIAL OFFLINE SCORING -- notebooks/_mk_claude_eval.py
# ==========================================================================
if EVAL_MOVIES:
    import traceback as _tb

    try:
        import inspect as _insp
        import pandas as _pd
        import tracksdata as _td2
        sys.path.insert(0, str(REPO_DIR / 'src'))
        from biohub_tracking import metrics as _M

        def _node_budget(geff_path):
            """The scorer's node budget, as harness/scorer.py reads it."""
            try:
                from geff import GeffMetadata
                meta = GeffMetadata.read(str(geff_path))
                val = (meta.extra or {}).get('estimated_number_of_nodes')
                return float(val) if val is not None else float('nan')
            except Exception as _e:
                print('   node budget unavailable:', _e, flush = True)
                return float('nan')

        _sub = _pd.read_csv(SUBMISSION_PATH)
        _rows = []
        _probe = True

        for _ds in sorted(_sub['dataset'].astype(str).unique()):
            _d = _sub[_sub['dataset'].astype(str) == _ds]
            _nrows = _d[_d['row_type'] == 'node']
            _erows = _d[_d['row_type'] == 'edge']
            _g = _td2.graph.IndexedRXGraph()

            if _probe:
                print('   add_node signature:', _insp.signature(_g.add_node), flush = True)
                _probe = False
            _map = {}

            for _r in _nrows.itertuples(index = False):
                _attrs = {'t': int(_r.t), 'z': float(_r.z),
                          'y': float(_r.y), 'x': float(_r.x)}
                try:
                    _map[int(_r.node_id)] = _g.add_node(_attrs)
                except TypeError:
                    _map[int(_r.node_id)] = _g.add_node(attrs = _attrs)

            for _r in _erows.itertuples(index = False):
                _g.add_edge(source_id = _map[int(_r.source_id)],
                            target_id = _map[int(_r.target_id)], attrs = {})
            _gt_path = COMP_DIR / 'train' / f'{_ds}.geff'
            _gt = graph_from_geff(_gt_path)
            _er = _M.evaluate(_g, _gt, VOXEL_SCALE_UM)
            _row = _M.per_sample_metrics(_er, _node_budget(_gt_path),
                                         _M.node_recall(_g, _gt))
            _row['dataset'] = _ds
            _rows.append(_row)
            print(f"   {_ds:<16} edge_J={_row['edge_jaccard']:.4f} "
                  f"adj={_row['adj_edge_jaccard']:.4f} "
                  f"div_tp/fp/fn={_row['division_tp']}/{_row['division_fp']}/"
                  f"{_row['division_fn']} ratio={_row['total_node_ratio']:+.4f}",
                  flush = True)
        _summary = _M.summarise(_rows)
        print('\\nEVAL SUMMARY ' + '=' * 60, flush = True)

        for _k in ('n', 'edge_jaccard', 'adj_edge_jaccard', 'division_jaccard',
                   'division_tp', 'division_fp', 'division_fn', 'node_recall', 'score'):
            if _k in _summary:
                print(f'   {_k:<20} {_summary[_k]}', flush = True)
        print('EVAL_SCORE', _summary.get('score'), flush = True)
        Path('/kaggle/working/eval_summary.json').write_text(
            json.dumps({'summary': _summary, 'per_dataset': _rows}, indent = 1, default = str))
    except Exception:
        # The inference already succeeded and submission.csv is on disk. Print everything a
        # second attempt would need and do NOT fail the run.
        print('EVAL SCORING FAILED -- inference output is still valid', flush = True)
        _tb.print_exc()
'''


def build(name: str, movies: int = 12) -> int:
    if name not in ARMS:
        print(f"REFUSING TO BUILD — {name!r} is not in ARMS")
        return 1
    src_nb = HERE / f"claude_arm_{name}.ipynb"
    if not src_nb.exists():
        print(f"REFUSING TO BUILD — {src_nb.name} not built; run _mk_claude_arm.py {name}")
        return 1

    nb = json.loads(src_nb.read_text())
    cells = nb["cells"]
    idx = next(i for i, c in enumerate(cells)
               if c.get("cell_type") == "code" and "os.environ" in "".join(c["source"]))
    body = "".join(cells[idx]["source"])

    if body.count(REDIRECT_OLD) != 1:
        print(f"REFUSING TO BUILD — TEST_DIR anchor matched {body.count(REDIRECT_OLD)}x")
        return 1
    body = body.replace(REDIRECT_OLD, REDIRECT_NEW.replace("__EVAL_MOVIES__", str(movies)), 1) + SCORE

    # Compile a copy with `from __future__ import annotations` hoisted. Every arm carries the
    # wheelhouse prologue above that import, which IPython tolerates and `compile` does not,
    # so checking the shipped text verbatim would reject notebooks that demonstrably run.
    # `_mk_claude_train.py` hit the same thing; the check is worth keeping either way, since
    # it is what stands between a builder typo and an hour of wasted GPU.
    future = "from __future__ import annotations\n"
    compile(future + body.replace(future, "", 1) if future in body else body,
            f"claude_eval_{name}", "exec")

    cells[idx]["source"] = body.splitlines(keepends=True)
    out = HERE / f"claude_eval_{name}.ipynb"
    out.write_text(json.dumps(nb, indent=1))

    (HERE / f"claude_eval_{name}_push.json").write_text(json.dumps({
        "slug": f"claude-eval-{name}", "title": f"Claude eval {name}",
        "notebook": str(out), "dataset_sources": TTA946_SOURCES,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [f"{K.username()}/claude-torch-wheelhouse"],
        "enable_gpu": True, "enable_internet": False,
    }, indent=1))
    print(f"wrote {out.name}: {len(body):,} chars (+{len(SCORE):,} scoring)")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    movies = 12
    if args and args[0].isdigit():
        movies, args = int(args[0]), args[1:]
    names = args or ["ttasec"]
    raise SystemExit(max(build(n, movies) for n in names))
