"""Split an arm's offline gain into its Jaccard half and its node-count half.

    python tools/split_score.py lb30 tight60 lb50 --base pub947bera

`MEMORY.md` §5 and `notes/89` §3: `adj = edge_J × (1 − 0.1 × (T_pred − T_est)/T_est)` is
unclamped above and the training labels are 0.2–15% dense, so **deleting nodes collects the
multiplier and the labels cannot charge you for it**. Three accounts measured the inversion
that follows — `zhincez` (0.952) spent four submissions on it, `rogerrogerroger3r` (0.953)
got the exact reverse of the leaderboard order on three, and our own `norelink` was offline
+0.0337 against a board −0.001.

The diagnostic is `zhincez`'s and it is the reason this file exists rather than another
threshold on PROXY_SCORE: **look at where the change came from, not how big it is.** A gain
that is mostly multiplier is unproven no matter how large; a gain that is all Jaccard at
constant node count is the one case the offline validator can rank.

Reads the per-movie table the arm's own local validator prints:

    44b6_12dfb391   edge_jaccard=0.9256 adj=0.9472 T_pred=44940 T_true=58672.0 div(...)

and reports, per arm, the mean `adj` change against the base, decomposed. It calls nothing;
logs come from the cache `tools/run_arm.py` and the batch scripts already write to
`/tmp/claude-0/claude-arm-<name>.log`, and are fetched only if absent.

**Read the Δ columns, not the level.** `mean adj` here is the unweighted mean of the
per-movie `adj`; the validator's own `PROXY_SCORE` line micro-averages instead, and on the
same eight movies that is 0.9285 where this prints 0.9136. The two disagree on the *level*
because a mean of ratios is not the ratio of sums. They agree on *differences* between arms
to within 0.0001, which is the only thing this file is for — and a per-movie paired mean is
the better-powered statistic for that anyway.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

CACHE = Path("/tmp/claude-0")
ROW = re.compile(r"^\s*(\w{4}_\w+)\s+edge_jaccard=([\d.]+) adj=([\d.]+) "
                 r"T_pred=(\d+) T_true=([\d.]+)", re.M)
SUB = re.compile(r"^\s*(\w{4}_\w+): nodes=(\d+) edges=(\d+) divisions=(\d+)", re.M)


def log(name: str) -> str:
    slug = f"claude-arm-{name}"
    p = CACHE / f"{slug}.log"
    if p.exists():
        return p.read_text()
    t = K.kernel_output(slug).get("log") or ""
    if isinstance(t, str) and t.startswith("["):
        try:
            t = "\n".join(str(e.get("data", "")) for e in json.loads(t))
        except Exception:
            pass
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(t)
    return t


def movies(text: str) -> dict:
    """{movie: (edge_J, adj, T_pred, T_true)} from the LAST validator pass in the log.

    The last pass, because an arm whose sweep tried candidates prints one table per
    candidate and only the final one describes what shipped. `notes/89` §1 is what taking
    the wrong pass costs: three published node counts that were sweep repetitions.
    """
    hits = ROW.findall(text)
    out, seen = {}, set()
    for m, j, adj, tp, tt in reversed(hits):        # walk back, keep the last of each
        if m in seen:
            continue
        seen.add(m)
        out[m] = (float(j), float(adj), int(tp), float(tt))
    return out


def report(names: list[str], base: str = "pub947bera") -> int:
    B = movies(log(base))
    bsub = SUB.findall(log(base))
    bnodes = sum(int(x[1]) for x in bsub)
    if not B:
        print(f"no validator table in {base}'s log — was VALIDATOR_ENABLE off?")
        return 1
    print(f"base {base}: {len(B)} movies, mean adj "
          f"{sum(v[1] for v in B.values())/len(B):.4f}, submission nodes {bnodes:,}\n")
    print(f"{'arm':<12}{'mean adj':>10}{'Δadj':>9}{'from J':>9}{'from mult':>11}"
          f"{'mult share':>12}{'sub nodes':>11}{'Δnodes':>9}  verdict")
    for n in names:
        A = movies(log(n))
        common = sorted(set(A) & set(B))
        if not common:
            print(f"{n:<12}  no shared movies with the base — cannot compare")
            continue
        dJ = sum(A[m][0] - B[m][0] for m in common) / len(common)
        # multiplier half, holding the base Jaccard fixed, so the two halves add up
        dM = sum(B[m][0] * ((1 - .1 * (A[m][2] - A[m][3]) / A[m][3])
                            - (1 - .1 * (B[m][2] - B[m][3]) / B[m][3]))
                 for m in common) / len(common)
        dA = sum(A[m][1] - B[m][1] for m in common) / len(common)
        share = abs(dM) / (abs(dJ) + abs(dM) + 1e-12)
        asub = SUB.findall(log(n))
        anodes = sum(int(x[1]) for x in asub)
        # The rule, stated before any of these ran. Mostly-multiplier is unproven whatever
        # its size; that is exactly the pattern that cost zhincez four submissions.
        if abs(dA) < 2e-4:
            v = "no change"          # both halves ~0; a share of them is meaningless
        elif dA <= 0:
            v = "reject — no gain"
        elif share > 0.7:
            v = f"UNPROVEN — {share:.0%} multiplier"
        else:
            v = "admissible"
        print(f"{n:<12}{sum(A[m][1] for m in common)/len(common):>10.4f}{dA:>+9.4f}"
              f"{dJ:>+9.4f}{dM:>+11.4f}{share:>11.0%}{anodes:>12,}"
              f"{anodes - bnodes:>+9,}  {v}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    base = "pub947bera"
    if "--base" in args:
        i = args.index("--base")
        base, args = args[i + 1], args[:i] + args[i + 2:]
    if not args:
        raise SystemExit(__doc__)
    raise SystemExit(report(args, base))
