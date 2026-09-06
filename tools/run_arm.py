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
        r = K.kernel_push(slug, notebook, title=title, is_private=True,
                          enable_gpu=cfg.get("enable_gpu", True),
                          enable_internet=cfg.get("enable_internet", False),
                          dataset_sources=cfg["dataset_sources"],
                          kernel_sources=cfg.get("kernel_sources") or [],
                          machine_shape=machine_shape)
        print(f"[{time.strftime('%H:%M:%S')}] {slug} attempt {attempt}: "
              f"pushed v{r.get('versionNumber')}", flush=True)

        st = K.kernel_wait(slug, poll=poll)
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
