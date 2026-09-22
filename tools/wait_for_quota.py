"""Poll until Kaggle's weekly GPU quota has room again, then start the queue.

    python tools/wait_for_quota.py lb80 lb150 lb50t60 lb30t60 flow2 x138

At 22:55 on 2026-09-22 every push came back with

    push discarded (v0) — Maximum weekly GPU quota of 30.00 hours reached.

`run_arm.run` reads that field and gives up immediately rather than burning five retries on
something that cannot clear in minutes, which is right — but then nothing restarts when it
*does* clear, and the quota is the binding resource with seven days to the deadline. So this
sits on it.

**The probe is a real push, because there is no quota endpoint.** That is safe: a push while
the quota is exhausted is discarded with `versionNumber: 0` and consumes nothing. The moment
one is accepted, that arm is running and the rest go to `run_queue.py`.

Cadence is deliberately slow. Kaggle rate-limits its endpoints on separate budgets
(`notes/90` §5) and `/kernels/push` is the one that matters here; a probe every twenty
minutes is enough to catch a reset that happens once a week.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def probe(arm: str) -> tuple[bool, str]:
    cfg = json.loads((HERE / f"notebooks/claude_arm_{arm}_push.json").read_text())
    r = K.kernel_push(cfg["slug"], cfg["notebook"], title=cfg["title"], is_private=True,
                      enable_gpu=cfg.get("enable_gpu", True),
                      enable_internet=cfg.get("enable_internet", False),
                      dataset_sources=cfg["dataset_sources"],
                      competition_sources=cfg.get("competition_sources") or [],
                      kernel_sources=cfg.get("kernel_sources") or [],
                      machine_shape="gpuT4x2")
    why = (r.get("error") or r.get("errorNullable") or "").strip()
    return bool(r.get("versionNumber")), why


def main(arms: list[str], poll: int = 1200, hours: float = 96.0) -> int:
    deadline = time.time() + hours * 3600
    first, rest = arms[0], arms[1:]
    while time.time() < deadline:
        try:
            ok, why = probe(first)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] probe unreachable: {e}", flush=True)
            time.sleep(poll)
            continue
        if ok:
            print(f"[{time.strftime('%H:%M:%S')}] QUOTA IS BACK — {first} accepted, "
                  f"starting queue on {' '.join(rest) or '(nothing else)'}", flush=True)
            if rest:
                subprocess.Popen([sys.executable, str(HERE / "tools/run_queue.py"), *rest],
                                 stdout=open("/tmp/claude-0/queue.log", "w"),
                                 stderr=subprocess.STDOUT, start_new_session=True)
            return 0
        print(f"[{time.strftime('%H:%M:%S')}] still blocked — "
              f"{why or 'no reason given'}", flush=True)
        time.sleep(poll)
    print("gave up waiting for quota", flush=True)
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1:]))
