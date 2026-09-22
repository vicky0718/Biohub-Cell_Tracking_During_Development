"""Join every public notebook to the leaderboard by AUTHOR, and diff the good ones at us.

    python tools/scout_notebooks.py                 # the ranked table
    python tools/scout_notebooks.py --min 0.950     # only authors at 0.950+
    python tools/scout_notebooks.py --diff user/slug [user/slug ...]

**Why this exists.** `notes/87` scouted the field by reading notebook *titles* and concluded
the public frontier was a 704-team plateau at 0.947 with nothing runnable above it. That was
wrong, and the error was the instrument: a title is what an author calls their work, and the
board is what it scored. Joining the two by author found six accounts between **0.948 and
0.957** publishing forks of the same base we run, on the same three mounts.

What that one join bought, in an afternoon and zero submission slots (`notes/89`):

* `zhincez` (0.952) had already spent **four** submissions on the node-deletion axis we were
  two GPU-hours into re-running — offline +0.013, board −0.004. Axis closed for free.
* `rogerrogerroger3r` (0.953) had paid for a division ablation: **the whole division channel
  is worth 0.020**, so `div_J` on this lineage is ~0.20 with ~0.08 still in the term.
* Four independent sources — `zhincez`, `thtennant` (0.953), `amanatar` (0.948) and the
  author of our own base in a later version — all run `MOTION_RELINK_TIGHT_UM` at **6.0**
  where our fork runs 5.5.
* Two of them add whole subsystems (flow relink, image-space gapfill, low-detection readmit)
  that no notebook on the plateau has.

The board snapshot is `discussions/raw/leaderboard.json`, written by
`discussions/scrape_discussions.py`. It is matched on team name, on every team member's
`userName`, and on the team leader's `userName`, because `/kernels/list` reports the author as
a username and the board reports a team name and they are only sometimes the same string.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
COMP = "biohub-cell-tracking-during-development"
BOARD = HERE / "discussions" / "raw" / "leaderboard.json"
PULLED = Path("/tmp/claude-0/pulled")


def board() -> tuple[dict, dict]:
    """{name-or-username: score}, {name-or-username: rank} from the saved snapshot."""
    if not BOARD.exists():
        raise SystemExit(f"no board snapshot at {BOARD}; "
                         f"run discussions/scrape_discussions.py first")
    d = json.loads(BOARD.read_text())
    score, rank = {}, {}
    for row, team in zip(d["publicLeaderboard"], d["teams"]):
        s, r = float(row["displayScore"]), row.get("rank", 0)
        keys = [team.get("teamName")]
        keys += [m.get("userName") for m in (team.get("teamMembers") or [])]
        keys.append(((team.get("teamUpInfo") or {}).get("teamLeader") or {}).get("userName"))
        for k in keys:
            if k:
                score[k], rank[k] = s, r
    return score, rank


def kernels(pages: int = 8) -> list[dict]:
    out = []
    for page in range(1, pages + 1):
        try:
            ks = K.get_json("/kernels/list", competition=COMP, pageSize=100, page=page,
                            sortBy="dateRun")
        except Exception as e:
            print(f"  ! page {page}: {e}", file=sys.stderr)
            break
        if not ks:
            break
        out += ks
    return out


def table(min_score: float = 0.948, since: str = "") -> int:
    score, rank = board()
    rows = []
    for k in kernels():
        a = k.get("author") or (k.get("ref") or "/").split("/")[0]
        rows.append((score.get(a, 0.0), rank.get(a, 99999), a, k.get("ref", ""),
                     k.get("totalVotes", 0), (k.get("lastRunTime") or "")[:10],
                     k.get("title", "")))
    rows.sort(key=lambda r: (-r[0], r[5]))
    shown = [r for r in rows if r[0] >= min_score and r[5] >= since]
    print(f"{len(rows)} notebooks, {len({r[2] for r in rows})} authors; "
          f"{len(shown)} at >= {min_score}" + (f" since {since}" if since else ""))
    print(f"{'score':>6} {'rank':>6} {'votes':>5} {'run':<11} ref")
    for s, rk, a, ref, v, d, t in shown:
        print(f"{s:6.3f} {rk:6} {v:5} {d:<11} {ref}")
    return 0


def _envs(text: str) -> dict:
    return dict(re.findall(
        r'os\.environ\[[\'"]BIOHUB_([A-Z0-9_]+)[\'"]\]\s*=\s*[\'"]([^\'"]*)[\'"]', text))


def _code(ref_or_path) -> str:
    p = Path(ref_or_path)
    if p.exists():
        d = json.loads(p.read_text())
        nb = json.loads(d["source"]) if isinstance(d.get("source"), str) else d
    else:
        user, slug = str(ref_or_path).split("/")
        nb = json.loads(K.get_json("/kernels/pull", userName=user,
                                   kernelSlug=slug)["blob"]["source"])
    return "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c.get("cell_type") == "code")


def diff(refs: list[str],
         base: str = "notebooks/claude_arm_pub947bera_source.json") -> int:
    """Env-block diff of each ref against our base. Keys only — the code diff is too big
    to read, but a notebook that ADDS a group of keys is announcing a whole subsystem."""
    B = _envs(_code(HERE / base))
    print(f"base {base}: {len(B)} BIOHUB_* keys")
    PULLED.mkdir(parents=True, exist_ok=True)
    for ref in refs:
        try:
            text = _code(ref)
        except Exception as e:
            print(f"\n=== {ref}: {e}")
            continue
        (PULLED / (ref.replace("/", "__") + ".txt")).write_text(text)
        E = _envs(text)
        chg = {k: (B[k], E[k]) for k in E if k in B and B[k] != E[k]}
        print(f"\n=== {ref}  ({len(E)} keys, {len(text):,} chars)")
        if chg:
            print("   CHANGED:", json.dumps(chg))
        add = {k: v for k, v in E.items() if k not in B}
        if add:
            print("   ADDED  :", json.dumps(add))
        gone = {k: B[k] for k in B if k not in E}
        if gone:
            print("   REMOVED:", json.dumps(gone))
        if not (chg or add or gone):
            print("   env block identical to our base")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--diff" in args:
        raise SystemExit(diff(args[args.index("--diff") + 1:]))
    lo = float(args[args.index("--min") + 1]) if "--min" in args else 0.948
    since = args[args.index("--since") + 1] if "--since" in args else ""
    raise SystemExit(table(lo, since))
