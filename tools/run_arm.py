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


def busy_slots(prefix: str = "claude-arm-") -> int:
    """How many of our arm kernels Kaggle currently has running.

    Kaggle allows two concurrent GPU sessions and **refuses a third silently**: the push
    succeeds, returns ``versionNumber: 0``, creates the kernel, and starts no run. Nine
    arms launched that way in five seconds and every one reported success. Counting first
    is the only way to tell a queued arm from a discarded one.
    """
    n = 0
    for p in sorted(Path(__file__).resolve().parent.parent
                    .glob("notebooks/claude_arm_*_push.json")):
        slug = json.loads(p.read_text())["slug"]
        try:
            if (K.kernel_status(slug).get("status") or "").lower() in ("running", "queued"):
                n += 1
        except Exception:
            pass                                  # 404 = never run; not occupying a slot
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
        machine_shape: str | None = "gpuT4x2") -> int:
    """Run the arm described by a `claude_arm_<name>_push.json` written by _mk_claude_arm.

    ``machine_shape`` is sent even though `notes/24` measured it as silently ignored --
    that measurement is from August, `nvidiaTeslaT4` is still ignored today, and
    `gpuT4x2` is the value the current UI uses. Sending it costs nothing and the log's
    GPU line says whether it took.
    """
    cfg = json.loads(Path(push_config).read_text())
    slug, notebook, title = cfg["slug"], cfg["notebook"], cfg["title"]
    for attempt in range(1, attempts + 1):
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
            print(f"   push discarded (v{version}) — no run started, retrying"
                  f" ({attempt}/{attempts})", flush=True)
            time.sleep(180)
            continue

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

    print(f"GAVE UP {slug}: {attempts} consecutive P100 draws", flush=True)
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raise SystemExit(run(sys.argv[1]))
