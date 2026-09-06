"""Check a finished arm did what it claims, before it costs a submission slot.

    python tools/verify_arm.py dc40 [baseline]      # baseline defaults to lb941

Three things go wrong between "the builder wrote the edit" and "the run measured it", and
all three have already happened in this project:

1. **The edit never reached the code.** `notes/68`: the public 0.948 kernel sets
   `DEEPCENTER_CHECKPOINT` to a path that does not exist under the real mount layout, the
   notebook falls back to its default, and the run is a no-op wearing an experiment's name.
2. **The wrong dump gets read.** These notebooks print their config *twice* — a resolved
   block early, and a defaults block after `Wrote run_stats.csv`. The second one shows
   `gap_close_um 5.8` for a run that used 5.0. Reading the last match inverts the answer.
3. **A cap binds before the parameter does.** `notes/60`: the value moves, a volume cap
   holds the output where it was, and the arm reproduces its base exactly.

So this reports, for the arm and its baseline side by side: the resolved value of every
`BIOHUB_*` the registry says it edits, the node and edge counts, and the `run_stats.csv`
counters for whichever stage the edit touches. It does **not** score anything — `notes/66`
established there is no local score here worth having.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness import claude_kaggle_api as K   # noqa: E402

CACHE = Path("/tmp/claude-0/arm_stats")

# Which run_stats.csv counters matter for each parameter an arm might move. An arm whose
# counters are identical to the baseline's did not change anything, whatever the config says.
COUNTERS = {
    "SAFE_DIV_DIVERGE_UM": ["safe_division_geometric_candidates",
                            "safe_division_divergence_rejected", "safe_divisions_added",
                            "safe_division_skipped_cap"],
    "DEEPCENTER_SAFE_DIV_THRESHOLD": ["deepcenter_safe_div_checked",
                                      "deepcenter_safe_div_accepted",
                                      "deepcenter_safe_div_rejected", "safe_divisions_added"],
    "DEEPCENTER_GAP_THRESHOLD": ["deepcenter_gap_checked", "deepcenter_gap_accepted",
                                 "deepcenter_gap_rejected",
                                 "deepcenter_gap_bypassed_strong_motion"],
    "GAP_CLOSE_UM": ["gap_candidates", "gap_pairs_selected", "gap_added_nodes",
                     "gap_skipped_node_cap", "gap2_candidates", "gap2_pairs_selected"],
    "DET_THRESHOLD": ["raw_nodes", "nodes", "raw_edges", "edges"],
    "DUAL_SEED_EDGE_THRESHOLD": ["raw_edges", "edges", "motion_relink_edges"],
    "SECONDARY_EDGE_WEIGHT": ["raw_edges", "edges", "motion_relink_edges",
                              "dropped_multi_parent_edges", "dropped_multi_child_edges"],
    "SECONDARY_LINK_MODE": ["raw_edges", "edges", "motion_relink_edges"],
}
ALWAYS = ["nodes", "edges", "division_like_sources", "short_track_nodes_removed"]


def fetch_log(slug: str) -> str:
    log = K.kernel_output(slug).get("log") or ""
    if isinstance(log, str) and log.startswith("["):
        try:
            log = "\n".join(str(e.get("data", "")) for e in json.loads(log))
        except Exception:
            pass
    return log


def resolved(log: str, key: str):
    """The key's value in the RESOLVED dump, or None if that dump does not carry it.

    These notebooks print their config twice and the second block is defaults, not what
    ran (`notes/68`): it reports `gap_close_um 5.8` for a run that used 5.0, and
    `secondary_link_mode low_margin_consensus` for a run that used `adaptive`. Searching
    the whole log finds the wrong one for any key the resolved block omits, which is how
    v1 of this tool declared three landed edits dead. Everything after the
    `Wrote .../run_stats.csv` line is therefore cut off before searching, and a key the
    resolved block never mentions comes back None -- inconclusive, read the counters.
    """
    cut = log.find("Wrote /kaggle/working/run_stats.csv")
    head = log[:cut] if cut > 0 else log
    m = re.search(rf'"{key.lower()}"\s*:\s*("?[^,\n"]+"?)', head)
    return m.group(1).strip('"') if m else None


def stats(slug: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{slug}.csv"
    if not path.exists():
        if K.kernel_output_file(slug, path, "run_stats.csv") is None:
            return {}
    import csv
    rows = list(csv.DictReader(path.open()))
    out = {}
    for k in rows[0] if rows else []:
        try:
            out[k] = sum(float(r[k]) for r in rows)
        except (ValueError, TypeError):
            pass
    return out


def edits_of(name: str) -> list[tuple[str, str, str]]:
    """(key, baseline value, arm value) for every edit the registry declares."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    import _mk_claude_arm as M
    pat = re.compile(r'os\.environ\["BIOHUB_([A-Z0-9_]+)"\]\s*=\s*"([^"]*)"')
    out = []
    for old, new in M.ARMS[name]["edits"]:
        a, b = pat.search(old), pat.search(new)
        if a and b:
            out.append((a.group(1), a.group(2), b.group(2)))
    return out


def main(name: str, baseline: str = "lb941") -> int:
    arm, base = f"claude-arm-{name}", f"claude-arm-{baseline}"
    edits = edits_of(name)
    log = fetch_log(arm)

    print(f"=== {arm}  vs  {base} ===")
    gpu = ", ".join(sorted(set(re.findall(r"Tesla [A-Z0-9\-]+", log)))) or "?"
    wh = "yes" if "wheelhouse install rc=0" in log else "no"
    print(f"gpu {gpu}   wheelhouse install {wh}")

    if not edits:
        print("registry declares no edits — this arm is an unmodified fork")
    # Two things v1 got wrong here, both of which turned a landed edit into a scare.
    # "0.40" != "0.4" as strings, and the config dump does NOT list every BIOHUB_ variable
    # -- `safe_div_diverge_um` appears nowhere in it, and `secondary_edge_weight` appears
    # only in the late defaults block that `notes/68` established lies. An absent key is
    # therefore INCONCLUSIVE, not a failure; the counters below are the real evidence.
    def same(a, b):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return str(a) == str(b)

    verdicts = []
    for key, want_base, want_arm in edits:
        got = resolved(log, key)
        v = "OK" if got is not None and same(got, want_arm) else (
            "NOT IN DUMP — read the counters" if got is None else "DID NOT LAND")
        verdicts.append(v)
        print(f"  {key:<32} want {want_arm:<10} resolved {str(got):<10} {v}")
    ok = all(v != "DID NOT LAND" for v in verdicts)
    if edits and not ok:
        print("\nThe dump shows the OLD value where it does report one. Do not spend a "
              "slot on this (notes/68 §2 is what this check exists for).")

    sa, sb = stats(arm), stats(base)
    if not sa or not sb:
        print("\nrun_stats.csv unavailable for one side; counters skipped")
        return 0 if ok else 1

    keys = list(dict.fromkeys(
        ALWAYS + [c for key, _, _ in edits for c in COUNTERS.get(key, [])]))
    print(f"\n{'counter':<40}{baseline:>14}{name:>14}{'delta':>12}")
    moved = False
    for k in keys:
        if k not in sa or k not in sb:
            continue
        d = sa[k] - sb[k]
        moved |= abs(d) > 0
        print(f"  {k:<38}{sb[k]:>14,.0f}{sa[k]:>14,.0f}{d:>+12,.0f}")
    if not moved:
        print("\nEvery counter is identical to the baseline. The parameter moved and the "
              "output did not — check whether a cap binds first (notes/60).")
    return 0 if ok else 1


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:3]))
