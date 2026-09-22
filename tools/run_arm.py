"""Push a fork notebook and keep re-pushing until Kaggle hands us a GPU it can run on.

    python tools/run_arm.py notebooks/claude_arm_lb941last_push.json

**The P100 lottery.** `MEMORY.md` records it and it has now cost four runs. A free Kaggle
GPU session draws either a Tesla T4 (sm_75) or a Tesla P100 (sm_60); the image's torch ships
kernels for sm_70 and up, so on a P100 `torch.cuda.is_available()` returns True and the first
real forward pass dies with *"no kernel image is available for execution on the device"*.
`machineShape` is accepted by `kernels/push` and silently ignored, so the accelerator cannot
be requested from the API — it can only be re-rolled.

Re-rolling is cheap, which is what makes this worth automating rather than avoiding: a P100
run dies about four minutes in, while a good run is hours. `claude_fork` and `claude_forkw085`
both drew T4s and completed; `claude_fork941_lb941` and `claude_fork941_adaptive` both drew
P100s and died in the detector's first `model.encode`. Nothing about the notebooks differed.

The loop only retries a **diagnosed** P100 death. Any other failure stops immediately and
prints the error region, because silently re-running a notebook with a real bug in it burns
the GPU quota that is our actual scarce resource (~140 h against ~115 submission slots).
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

P100 = re.compile(r"Tesla P100.*?is not compatible|no kernel image is available", re.S)
ERRLINE = re.compile(r"Traceback|Error:|error:|assert|RuntimeError|KeyError|FileNotFound")


def fetch_log(slug: str) -> str:
    try:
        log = K.kernel_output(slug).get("log") or ""
    except Exception as e:                      # output is not always ready at the instant
        return f"(log unavailable: {e})"        # the status flips; treat as empty, not fatal
    if isinstance(log, str) and log.startswith("["):
        try:
            log = "\n".join(str(e.get("data", "")) for e in json.loads(log))
        except Exception:
            pass
    return log


_BUSY = {"at": 0.0, "n": 0}     # memoised busy-slot count, shared across queue workers


def busy_slots(prefix: str = "claude-arm-", fresh_h: float = 48.0,
               ttl: float = 300.0, cap: int = 2) -> int:
    """How many of our GPU kernels Kaggle currently has running.

    Kaggle allows two concurrent GPU sessions and **refuses a third silently**: the push
    succeeds, returns ``versionNumber: 0``, creates the kernel, and starts no run. Nine
    arms launched that way in five seconds and every one reported success. Counting first
    is the only way to tell a queued arm from a discarded one.

    v1 globbed ``claude_arm_*_push.json`` only, which was right when arms were the only
    thing we ran. `claude_eval_*` kernels are GPU kernels too, and so is the **graded rerun
    Kaggle starts when a human submits an arm** — that one occupies a slot for ~11 h under
    the arm's own slug. With the eval kernels invisible to this count, `claude-eval-nrmtl3`
    pushed six times against two busy slots, was discarded six times, and `run()` reported
    *"6 consecutive P100 draws"* — a diagnosis it had no evidence for.

    Now counts every ``claude_*_push.json`` that declares ``enable_gpu``. CPU kernels
    (``claude-score``) do not consume a GPU session and are skipped.

    **And it is bounded, in two ways, because v2 was not.** v2 statused *every* GPU push
    config ever written -- 30 of them and growing -- so one call cost 30 requests, two queue
    workers calling it every 180 s cost ~60 requests per 3 minutes, and Kaggle answered
    `429 TooManyRequests` to everything, including the reads that were trying to collect
    results. So: only configs touched within ``fresh_h`` are considered (a kernel we have
    not pushed in two days is not occupying a slot today), and the whole count is memoised
    for ``ttl`` seconds and shared between the queue's workers. Both bounds are there to
    keep a *count* from starving the *reads*.
    """
    now = time.time()
    if _BUSY["at"] + ttl > now:
        return _BUSY["n"]
    n = 0
    for p in sorted(Path(__file__).resolve().parent.parent
                    .glob("notebooks/claude_*_push.json")):
        if now - p.stat().st_mtime > fresh_h * 3600:
            continue
        cfg = json.loads(p.read_text())
        if not cfg.get("enable_gpu", True):
            continue
        try:
            if (K.kernel_status(cfg["slug"]).get("status") or "").lower() in ("running",
                                                                             "queued"):
                n += 1
        except Exception:
            pass                                  # 404 = never run; not occupying a slot
        if n >= cap:
            break        # the caller only ever asks ">= 2"; counting past it buys nothing
    _BUSY.update(at=time.time(), n=n)
    return n


def wait_for_run(slug: str, grace: int = 900, poll: int = 30) -> dict | None:
    """Block until the kernel reports a status, or None if no run ever appeared.

    ``/kernels/status`` answers 404 *"No runs found for this kernel"* between a push and
    the run actually starting, and `kernel_wait` raised straight through it. Treated as
    "not started yet" for ``grace`` seconds, then as "the push did not start a run".
    """
    t0 = time.time()
    while time.time() - t0 < grace:
        try:
            return K.kernel_wait(slug, poll=poll)
        except K.KaggleError as e:
            if e.status != 404:
                raise
            time.sleep(poll)
    return None


def run(push_config: str, attempts: int = 6, poll: int = 90,
        machine_shape: str | None = "gpuT4x2", busy_wait_h: float = 8.0) -> int:
    """Run the arm described by a `claude_arm_<name>_push.json` written by _mk_claude_arm.

    ``machine_shape`` is sent even though `notes/24` measured it as silently ignored --
    that measurement is from August, `nvidiaTeslaT4` is still ignored today, and
    `gpuT4x2` is the value the current UI uses. Sending it costs nothing and the log's
    GPU line says whether it took.

    ``attempts`` is a budget for things that can *go wrong* -- a P100 draw, a push that
    starts no run. **A busy GPU session is not one of them**, and v3 spent the budget on it:
    `tight60` and `lb50` were queued behind two hour-long runs, and both would have exhausted
    six attempts in eighteen minutes against a wait that was always going to be an hour. The
    fix is not more attempts; it is to stop counting a wait as a failure. Waiting is bounded
    separately by ``busy_wait_h``, long enough to outlast any single run of this pipeline.
    """
    cfg = json.loads(Path(push_config).read_text())
    slug, notebook, title = cfg["slug"], cfg["notebook"], cfg["title"]
    discards = 0
    attempt = 0
    busy_deadline = time.time() + busy_wait_h * 3600
    while attempt < attempts:
        attempt += 1
        # Wait for a free session before pushing. Kaggle silently discards the run
        # otherwise (see busy_slots), and a discarded push is indistinguishable from a
        # successful one in the response.
        while busy_slots() >= 2:
            time.sleep(180)

        r = K.kernel_push(slug, notebook, title=title, is_private=True,
                          enable_gpu=cfg.get("enable_gpu", True),
                          enable_internet=cfg.get("enable_internet", False),
                          dataset_sources=cfg["dataset_sources"],
                          competition_sources=cfg.get("competition_sources") or [],
                          kernel_sources=cfg.get("kernel_sources") or [],
                          machine_shape=machine_shape)
        version = r.get("versionNumber")
        print(f"[{time.strftime('%H:%M:%S')}] {slug} attempt {attempt}: "
              f"pushed v{version}", flush=True)

        # `versionNumber: 0` IS the discard, and it is the only reliable signal of one.
        # `wait_for_run` catches a discard by way of /kernels/status 404ing -- which only
        # happens on a kernel that has never run. Re-push a kernel that HAS run, get a
        # discarded v0, and status still answers `complete` from the previous run: the
        # discard reports success in one second, reads the old log, and prints DONE. That
        # is exactly how a clean re-push of `claude-arm-tta946` was recorded as having
        # happened when Kaggle had thrown it away. Check the version, not the aftermath.
        if not version:
            discards += 1
            # Kaggle SAYS why, in the push response, and v1 never looked. Six pushes of
            # `claude-arm-ftune` were reported as "no free GPU session" while every response
            # carried `"error": "Maximum weekly GPU quota of 30.00 hours reached."`. A
            # discard has at least two causes and they need opposite responses: a busy slot
            # clears in minutes, an exhausted quota does not clear for days, and retrying it
            # five more times is pure noise. Read the field.
            why = (r.get("error") or r.get("errorNullable") or "").strip()
            print(f"   push discarded (v{version}) — "
                  f"{why or 'no reason given by Kaggle'}", flush=True)
            if "quota" in why.lower():
                print(f"GAVE UP {slug}: weekly GPU quota is exhausted. Retrying cannot help "
                      f"until it resets; nothing ran.", flush=True)
                return 1
            # "Maximum batch GPU session count of 2 reached" means the two slots are full --
            # a queue position, not a fault. `busy_slots()` is supposed to catch this before
            # the push, but it reads OUR push configs and races a run that is starting, so
            # the authoritative answer is this field. Wait it out WITHOUT spending an attempt.
            if "session count" in why.lower() or "concurrent" in why.lower():
                discards -= 1                       # not a fault; do not report it as one
                attempt -= 1
                if time.time() > busy_deadline:
                    print(f"GAVE UP {slug}: GPU sessions stayed full for {busy_wait_h:g} h. "
                          f"Nothing ran; this is NOT a P100 or quota problem.", flush=True)
                    return 1
                left = (busy_deadline - time.time()) / 3600
                print(f"   both GPU sessions busy — waiting (up to {left:.1f} h more)",
                      flush=True)
                time.sleep(180)
                continue
            print(f"   retrying ({attempt}/{attempts})", flush=True)
            time.sleep(180)
            continue

        # Then wait for the NEW run to actually start before believing any status.
        # `/kernels/status` reports the kernel, not the version, so for the first minute
        # after a push it still answers with the previous run's terminal state. The version
        # check above catches the discard case; this catches its mirror, which cost a real
        # reading the same day: `claude-arm-ttadom` v2 pushed fine and was recorded as
        # `failed` eight seconds later, because v1 had ended in `error` and that is what
        # status still said. Both arms were running normally at the time.
        # Every call here is wrapped: a 429 from polling too eagerly is a rate limit, not a
        # verdict on the arm, and letting it propagate took `ttadom` out of a queue whose
        # run was proceeding normally on Kaggle at that moment.
        for _ in range(20):
            try:
                cur = (K.kernel_status(slug).get("status") or "").lower()
            except Exception:
                cur = ""
            if cur in ("running", "queued"):
                break
            time.sleep(30)

        st = wait_for_run(slug, poll=poll)
        if st is None:
            print(f"   push started no run (v{r.get('versionNumber')}) — retrying"
                  f" ({attempt}/{attempts})", flush=True)
            continue
        status = (st.get("status") or "").lower()
        log = fetch_log(slug)
        gpu = ", ".join(sorted(set(re.findall(r"Tesla [A-Z0-9\-]+", log)))) or "?"
        print(f"[{time.strftime('%H:%M:%S')}] {slug} -> {status} on {gpu} "
              f"({len(log):,} log chars)", flush=True)

        if status == "complete":
            for line in log.splitlines():
                if "PROXY_SCORE" in line or "score" in line.lower()[:40]:
                    print("   " + line.strip()[:160], flush=True)
            print(f"DONE {slug}", flush=True)
            return 0

        if P100.search(log):
            print(f"   P100 draw — re-rolling ({attempt}/{attempts})", flush=True)
            continue

        print(f"FAILED {slug}: not a P100 death, stopping.", flush=True)
        lines = log.splitlines()
        hit = next((i for i, l in enumerate(lines) if ERRLINE.search(l)), None)
        if hit is not None:
            print("\n".join(l.rstrip() for l in lines[hit:hit + 40]), flush=True)
        return 1

    # Say which failure actually happened. v1 printed "consecutive P100 draws"
    # unconditionally, so `claude-eval-nrmtl3` -- discarded six times because two GPU
    # slots were busy -- was reported as a GPU-lottery problem it never had. A wrong
    # diagnosis printed confidently is worse than none; it sent the next reader looking
    # at accelerators instead of at concurrency.
    if discards == attempts:
        print(f"GAVE UP {slug}: {attempts} pushes all discarded (v0) — Kaggle had no\n   free GPU session. Nothing ran; this is NOT a P100 problem.", flush=True)
    elif discards:
        print(f"GAVE UP {slug}: {attempts} attempts — {discards} discarded for want of a\n   free GPU session, {attempts - discards} P100 draws.", flush=True)
    else:
        print(f"GAVE UP {slug}: {attempts} consecutive P100 draws", flush=True)
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raise SystemExit(run(sys.argv[1]))
