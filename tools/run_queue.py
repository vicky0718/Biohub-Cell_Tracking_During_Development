"""Keep both Kaggle GPU sessions busy with a prioritised queue of arms.

    python tools/run_queue.py dc40 dcgap40 adaptive union gap44 dse44 dse52 det960

`notes/65` §2: runs are autonomous but scoring is not — a human presses Submit, five a day.
So the point of a queue is not throughput for its own sake; it is that when a slot frees at
04:00 there should already be a finished notebook waiting rather than an idle GPU and a
turn-around delay. Two workers, because Kaggle allows two concurrent GPU sessions.

Each worker runs one arm to completion through `tools/run_arm.py`'s logic, then takes the
next name. Order is priority order: the queue is consumed left to right, so put the arm you
most want a result from first.

A P100 draw is retried (`notes/65` §1 — the accelerator cannot be requested and every draw
on 2026-09-06 was a P100, which is what the wheelhouse prologue exists for). Any other
failure takes that arm out of the queue and the worker moves on, because a notebook with a
real bug in it will fail identically on retry and the GPU quota is the scarce resource.

State goes to `/tmp/claude-0/queue_state.json` after every transition, so the queue's
progress survives this process being killed and can be read without parsing the log.
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_arm import run  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
STATE = Path("/tmp/claude-0/queue_state.json")

_lock = threading.Lock()
_state = {"queued": [], "running": {}, "done": {}, "failed": {}}


def _save():
    with _lock:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(_state, indent=1))


def worker(wid: int):
    while True:
        with _lock:
            if not _state["queued"]:
                return
            name = _state["queued"].pop(0)
            _state["running"][str(wid)] = {"arm": name, "since": time.strftime("%H:%M:%S")}
        _save()

        cfg = HERE / "notebooks" / f"claude_arm_{name}_push.json"
        if not cfg.exists():
            print(f"[w{wid}] {name}: no push config at {cfg}", flush=True)
            with _lock:
                _state["failed"][name] = "missing push config"
                _state["running"].pop(str(wid), None)
            _save()
            continue

        print(f"[w{wid}] starting {name}", flush=True)
        try:
            rc = run(str(cfg))
        except Exception as e:                       # a crashed worker must not silently
            print(f"[w{wid}] {name} raised {type(e).__name__}: {e}", flush=True)
            rc = 1                                   # take the rest of the queue with it
        with _lock:
            (_state["done"] if rc == 0 else _state["failed"])[name] = \
                time.strftime("%Y-%m-%d %H:%M:%S")
            _state["running"].pop(str(wid), None)
        _save()
        print(f"[w{wid}] {name} -> {'done' if rc == 0 else 'FAILED'}", flush=True)


def main(names: list[str], workers: int = 2) -> int:
    if not names:
        raise SystemExit(__doc__)
    _state["queued"] = list(names)
    _save()
    print(f"queue: {' '.join(names)}   ({workers} workers)", flush=True)
    threads = [threading.Thread(target=worker, args=(i,), daemon=False)
               for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\ndone: {sorted(_state['done'])}", flush=True)
    print(f"failed: {sorted(_state['failed'])}", flush=True)
    return 0 if not _state["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
