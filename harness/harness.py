"""Honest validation harness for the Biohub cell-tracking contest.

Wraps the *official* scorer (`tracking_cellmot.metrics`) so that a number measured
here is the same number the leaderboard computes. Everything else in this file is
bookkeeping in service of one rule, carried over from the ROGII contest:

    a change is promoted only when it improves the pooled score AND does not
    regress any fold.

Aggregate-only gains are how you buy leaderboard noise and call it progress.

Usage
-----
    from harness import Harness

    h = Harness(data_dir=TRAIN)                       # official splits auto-loaded
    base = h.evaluate(my_predictor, arm="baseline")
    arm  = h.evaluate(my_predictor_v2, arm="lower_threshold")
    print(gate(base, arm))                            # PROMOTE / REJECT, with reasons

`predict_fn` has the signature ``fn(name: str, data_dir: Path) -> td.graph.BaseGraph``
and must return a graph in *voxel* coordinates, the same space as the ground truth.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import tracksdata as td

from .scorer import (
    DEFAULT_SCALE,
    evaluate,
    node_recall,
    per_sample_metrics,
    read_estimated_nodes,
    read_scale,
    summarise,
)

PredictFn = Callable[[str, Path], "td.graph.BaseGraph"]


def load_geff(path: Path | str) -> td.graph.BaseGraph:
    result = td.graph.IndexedRXGraph.from_geff(str(path))
    return result[0] if isinstance(result, tuple) else result


@dataclass
class Result:
    """Per-dataset metric rows plus the aggregates the leaderboard reports."""

    arm: str
    rows: dict[str, dict] = field(default_factory=dict)      # name -> per_sample_metrics
    fold_of: dict[str, int] = field(default_factory=dict)    # name -> fold index

    @property
    def summary(self) -> dict:
        return summarise(list(self.rows.values()))

    @property
    def score(self) -> float:
        return self.summary["score"]

    def fold_summaries(self) -> dict[int, dict]:
        """Aggregate separately within each fold — this is where regressions hide."""
        out: dict[int, dict] = {}
        for fold in sorted(set(self.fold_of.values())):
            members = [r for n, r in self.rows.items() if self.fold_of.get(n) == fold]
            if members:
                out[fold] = summarise(members)
        return out

    def to_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arm": self.arm,
            "rows": self.rows,
            "fold_of": self.fold_of,
            "summary": self.summary,
            "fold_summaries": {str(k): v for k, v in self.fold_summaries().items()},
        }
        path.write_text(json.dumps(payload, indent=2, default=_json_safe))
        return path

    @classmethod
    def from_json(cls, path: Path | str) -> "Result":
        d = json.loads(Path(path).read_text())
        return cls(arm=d["arm"], rows=d["rows"],
                   fold_of={k: int(v) for k, v in d["fold_of"].items()})

    def report(self) -> str:
        s = self.summary
        lines = [
            f"=== {self.arm} ===",
            f"  n={s['n']}  SCORE={_f(s['score'])}",
            f"  edge_jaccard={_f(s['edge_jaccard'])}  adj_edge_jaccard={_f(s['adj_edge_jaccard'])}"
            f" (n_adj={s['n_adj']})",
            f"  division_jaccard={_f(s['division_jaccard'])}"
            f"  (TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})",
            f"  node_recall={_f(s['node_recall'])}",
        ]
        folds = self.fold_summaries()
        if len(folds) > 1:
            lines.append("  per fold:")
            for f, fs in folds.items():
                lines.append(f"    fold {f}: score={_f(fs['score'])} "
                             f"edge_J={_f(fs['edge_jaccard'])} n={fs['n']}")
        return "\n".join(lines)


class Harness:
    """Scores predictors against ground truth on fixed folds, with caching."""

    def __init__(
        self,
        data_dir: Path | str,
        splits: list[dict] | None = None,
        max_distance: float = 7.0,
        cache_dir: Path | str | None = None,
        n_total_override: dict[str, float] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_distance = max_distance
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.n_total_override = n_total_override or {}
        self.splits = splits if splits is not None else self._load_splits()
        self._fold_of = self._build_fold_map()
        self._warned_missing_budget = False

    # -- dataset discovery ---------------------------------------------------

    def dataset_names(self) -> list[str]:
        """Datasets with both an image and ground truth (scoring needs the .geff)."""
        return sorted(p.stem for p in self.data_dir.glob("*.geff"))

    def _load_splits(self) -> list[dict] | None:
        for cand in (self.data_dir / "dataset_splits.json",
                     self.data_dir.parent / "dataset_splits.json"):
            if cand.exists():
                return json.loads(cand.read_text())
        return None

    def _build_fold_map(self) -> dict[str, int]:
        """name -> fold index. Falls back to deterministic folds if none ship."""
        if self.splits:
            m: dict[str, int] = {}
            for i, fold in enumerate(self.splits):
                for name in fold.get("test", []):
                    m[name] = i
            if m:
                return m
        # Deterministic fallback: stable hash of the name, so folds never drift
        # between runs or machines the way `hash()` would.
        import hashlib
        names = self.dataset_names()
        k = min(5, max(1, len(names)))
        return {
            n: int(hashlib.sha1(n.encode()).hexdigest(), 16) % k
            for n in names
        }

    def fold_of(self, name: str) -> int:
        return self._fold_of.get(name, 0)

    # -- scoring -------------------------------------------------------------

    def _n_total(self, name: str) -> float:
        if name in self.n_total_override:
            return float(self.n_total_override[name])
        n = read_estimated_nodes(self.data_dir / f"{name}.geff")
        if math.isnan(n) and not self._warned_missing_budget:
            warnings.warn(
                f"{name}.geff has no `estimated_number_of_nodes`; the node-budget term "
                "cannot be reproduced locally, so adj_edge_jaccard will be NaN. Pass "
                "n_total_override= to score the adjusted metric.",
                stacklevel=2,
            )
            self._warned_missing_budget = True
        return n

    def score_graph(self, name: str, graph: td.graph.BaseGraph) -> dict:
        """Score one predicted graph against this dataset's ground truth."""
        gt = load_geff(self.data_dir / f"{name}.geff")
        scale = read_scale(self.data_dir / f"{name}.zarr")
        er = evaluate(graph, gt, scale=scale, max_distance=self.max_distance)
        rec = (node_recall(graph, gt)
               if graph.num_nodes() > 0 and graph.num_edges() > 0 else 0.0)
        return per_sample_metrics(er, self._n_total(name), rec)

    def evaluate(
        self,
        predict_fn: PredictFn,
        arm: str,
        names: Iterable[str] | None = None,
        use_cache: bool = True,
        verbose: bool = True,
    ) -> Result:
        """Run `predict_fn` over datasets and score each one.

        Results are cached per (arm, dataset) so a rerun never repeats work —
        delete the cache file to force recomputation.
        """
        names = list(names) if names is not None else self.dataset_names()
        res = Result(arm=arm)
        cache_path = self.cache_dir / f"{arm}.json" if self.cache_dir else None
        cached: dict[str, dict] = {}
        if use_cache and cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text()).get("rows", {})

        for name in names:
            if name in cached:
                res.rows[name] = cached[name]
                src = "cache"
            else:
                graph = predict_fn(name, self.data_dir)
                res.rows[name] = self.score_graph(name, graph)
                src = "computed"
            res.fold_of[name] = self.fold_of(name)
            if verbose:
                r = res.rows[name]
                print(f"  [{arm}] {name:<26} J={_f(r['edge_jaccard'])} "
                      f"adj={_f(r['adj_edge_jaccard'])} "
                      f"TP/FP/FN={r['edge_tp']}/{r['edge_fp']}/{r['edge_fn']} "
                      f"n_pred={r['num_pred_nodes']} ({src})", flush=True)

        if cache_path:
            res.to_json(cache_path)
        return res


# -- the gate ----------------------------------------------------------------

@dataclass
class Verdict:
    promote: bool
    pooled_delta: float
    fold_deltas: dict[int, float]
    reasons: list[str]

    def __str__(self) -> str:
        head = "PROMOTE" if self.promote else "REJECT"
        lines = [f"{head}  pooled delta = {self.pooled_delta:+.4f}"]
        for f, d in sorted(self.fold_deltas.items()):
            flag = "" if d >= 0 else "   <-- REGRESSION"
            lines.append(f"  fold {f}: {d:+.4f}{flag}")
        lines += [f"  {r}" for r in self.reasons]
        return "\n".join(lines)


def gate(baseline: Result, arm: Result, min_pooled_gain: float = 0.0) -> Verdict:
    """Preregistered promotion rule: pooled gain AND no fold regression.

    Compares only datasets scored in both arms, so the comparison stays paired.
    Higher score is better (unlike ROGII's RMSE, where lower won).
    """
    shared = sorted(set(baseline.rows) & set(arm.rows))
    reasons: list[str] = []
    if not shared:
        return Verdict(False, float("nan"), {}, ["no datasets in common — nothing to compare"])
    if len(shared) < len(baseline.rows) or len(shared) < len(arm.rows):
        reasons.append(f"compared on {len(shared)} shared datasets "
                       f"(baseline {len(baseline.rows)}, arm {len(arm.rows)})")

    def _sub(res: Result) -> Result:
        return Result(arm=res.arm,
                      rows={n: res.rows[n] for n in shared},
                      fold_of={n: res.fold_of.get(n, 0) for n in shared})

    b, a = _sub(baseline), _sub(arm)
    pooled = a.score - b.score

    bf, af = b.fold_summaries(), a.fold_summaries()
    fold_deltas = {f: af[f]["score"] - bf[f]["score"] for f in sorted(set(bf) & set(af))}

    regressions = [f for f, d in fold_deltas.items() if d < 0]
    promote = pooled > min_pooled_gain and not regressions

    if pooled <= min_pooled_gain:
        reasons.append(f"pooled gain {pooled:+.4f} does not clear {min_pooled_gain:+.4f}")
    if regressions:
        reasons.append(f"regresses folds {regressions} — a pooled gain built on a fold "
                       "regression is the shape of overfitting, not improvement")
    if promote:
        reasons.append("gain is positive in every fold")
    if len(fold_deltas) < 2:
        reasons.append("WARNING: fewer than 2 folds — the no-regression half of the gate "
                       "is not actually testing anything")
    return Verdict(promote, pooled, fold_deltas, reasons)


# -- helpers -----------------------------------------------------------------

def _f(v) -> str:
    try:
        return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
    except (TypeError, ValueError):
        return str(v)


def _json_safe(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


__all__ = ["Harness", "Result", "Verdict", "gate", "load_geff", "DEFAULT_SCALE"]
