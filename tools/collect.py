"""Poll a set of kernels slowly, cache each log the moment it lands, and say what happened.

    python tools/collect.py tight60 lb50 geofus            # arms
    python tools/collect.py --slug claude-probe            # any kernel slug

**Why this is separate from `run_arm.py`.** That file conflates launching with collecting,
and the two have opposite failure modes. Launching wants to react fast; collecting wants to
be patient and cheap, because a run is an hour long and the API is rate-limited. When they
share a process, one 429 in the collecting half destroys the launching half's record of the
run: `claude-arm-tight60` was pushed, started, and executing on Kaggle when `kernel_wait`
raised `HTTP 429` through `run_queue`, which wrote it down as **FAILED** and moved on. The
arm was fine. The transport was not.

So: this process only reads, at a deliberately slow cadence, retries every rate limit
forever rather than concluding anything from one, and writes each finished log to
`/tmp/claude-0/<slug>.log` where `tools/split_score.py` and `tools/verify_arm.py` expect it.
A kernel it cannot reach is reported as unreachable, never as failed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

CACHE = Path("/tmp/claude-0")
TERMINAL = ("complete", "error", "cancelled", "cancelacknowledged")


def fetch_log(slug: str) -> str:
    log = K.kernel_output(slug).get("log") or ""
    if isinstance(log, str) and log.startswith("["):
        try:
            log = "\n".join(str(e.get("data", "")) for e in json.loads(log))
        except Exception:
            pass
    return log


def collect(slugs: list[str], poll: int = 300, hours: float = 13.0) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    pending = list(dict.fromkeys(slugs))
    deadline = time.time() + hours * 3600
    while pending and time.time() < deadline:
        for slug in list(pending):
            try:
                status = (K.kernel_status(slug).get("status") or "").lower()
            except K.KaggleError as e:
                # 429 and friends say nothing about the run. Neither does a 404 here --
                # it means no run has been recorded yet, which is a state, not a verdict.
                print(f"[{time.strftime('%H:%M:%S')}] {slug}: unreachable "
                      f"({e.status}); will retry", flush=True)
                continue
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] {slug}: unreachable ({e}); "
                      f"will retry", flush=True)
                continue
            if status not in TERMINAL:
                print(f"[{time.strftime('%H:%M:%S')}] {slug}: {status or '(no status)'}",
                      flush=True)
                continue
            try:
                log = fetch_log(slug)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] {slug}: {status}, log not ready "
                      f"({e})", flush=True)
                continue
            (CACHE / f"{slug}.log").write_text(log)
            pending.remove(slug)
            lines = [l.strip() for l in log.splitlines()
                     if "PROXY_SCORE" in l or "Final submission" in l
                     or "SELECTED:" in l or "EVAL_SCORE" in l]
            print(f"[{time.strftime('%H:%M:%S')}] {slug}: {status.upper()} "
                  f"({len(log):,} chars) -> {CACHE / (slug + '.log')}", flush=True)
            for l in lines[-4:]:
                print("    " + l[:150], flush=True)
        if pending:
            time.sleep(poll)
    if pending:
        print(f"still pending after {hours:g} h: {', '.join(pending)}", flush=True)
        return 1
    print("all collected", flush=True)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    slugs = []
    i = 0
    while i < len(args):
        if args[i] == "--slug":
            slugs.append(args[i + 1])
            i += 2
        else:
            slugs.append(f"claude-arm-{args[i]}")
            i += 1
    if not slugs:
        raise SystemExit(__doc__)
    raise SystemExit(collect(slugs))
