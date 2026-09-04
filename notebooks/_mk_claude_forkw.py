"""Build fork variants sweeping BIOHUB_SECONDARY_DETECTION_WEIGHT past where its author stopped.

    python notebooks/_mk_claude_forkw.py 0.85 0.90 0.95

`claude_fork` (nusrati/0-938 unmodified) scored **0.937**, PROXY_SCORE 0.9266 on its own
10-dataset validator. That calibration point is the reason this sweep is affordable: every
run prints a PROXY_SCORE, so arms can be screened WITHOUT spending a submission slot.

The knob. `nusrati/0-936`'s header states its single change over the 0.931 parent was
`BIOHUB_SECONDARY_DETECTION_WEIGHT` **0.475 -> 0.80**, "our independently-swept dual-seed
detection-fusion peak (+0.003 on our lineage)". The fork's own validator enforces
``must be in the half-open interval [0, 1)``, so **0.80-0.99 is unexplored**.

They call 0.80 a peak. This project has recorded three "located interior optima" that were
noise (`notes/41`, `notes/42`, `notes/44` withdrew all three), so a claimed peak from an
unknown-n sweep is worth one cheap check -- but expectations are tempered accordingly, and
this is a screen against PROXY rather than a submission.

`notes/49` applies to every PROXY reading: the validator runs on TRAINING embryos, `notes/24`
records that the pack's split_0 membership is unknowable so those datasets may be
contaminated, and the test set is a third pair. PROXY is a screen, not a decision.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROV = HERE / "claude_fork_source.json"
OLD = 'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.80"'

HEADER = """\
# ==========================================================================
# FORK of nusrati/0-938 (public) -- NOT OUR WORK, except the one line below.
#   lineage  stephennedumpally/pls-upvote-share-higher-scoring-ideas  LB 0.931
#              -> nusrati/0-936  (SECONDARY_DETECTION_WEIGHT 0.475 -> 0.80)
#              -> nusrati/0-938  (division geometry)
# All credit to those authors. The unmodified reproduction scored 0.937 (PROXY 0.9266).
#
# OUR ONE CHANGE: BIOHUB_SECONDARY_DETECTION_WEIGHT 0.80 -> {w}
#
# 0.80 is where their sweep stopped and they call it a peak. The code caps this
# at [0, 1), so 0.80-0.99 is unexplored. notes/41/42/44 record three "located
# interior optima" in this project that were later withdrawn as noise, so a
# claimed peak from an unknown-n sweep is worth one screening run.
#
# SCREEN ONLY -- judged on this notebook's own PROXY_SCORE (fork = 0.9266),
# not submitted unless it beats that. notes/49: PROXY runs on training embryos
# and the test set is a third pair, so this screens direction, not magnitude.
# ==========================================================================
"""


def build(weight: str) -> int:
    prov = json.loads(PROV.read_text())
    nb = json.loads(prov["source"])
    cells = nb["cells"]
    new = f'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "{weight}"'

    hits = [i for i, c in enumerate(cells)
            if c.get("cell_type") == "code" and OLD in "".join(c["source"])]
    total = sum("".join(cells[i]["source"]).count(OLD) for i in hits)
    if total != 1:
        print(f"REFUSING TO WRITE — matched {total}x, expected 1")
        return 1
    i = hits[0]
    cells[i]["source"] = "".join(cells[i]["source"]).replace(OLD, new, 1) \
        .splitlines(keepends=True)
    if cells[0].get("cell_type") != "code":
        print("REFUSING TO WRITE — first cell is not code")
        return 1
    cells[0]["source"] = (HEADER.format(w=weight)
                          + "".join(cells[0]["source"])).splitlines(keepends=True)

    out = HERE / f"claude_forkw{weight.replace('.', '')}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out.name}: SECONDARY_DETECTION_WEIGHT 0.80 -> {weight}")
    return 0


def main(argv) -> int:
    weights = argv or ["0.85", "0.90", "0.95"]
    for w in weights:
        if not (0.0 <= float(w) < 1.0):
            print(f"REFUSING — {w} outside the code's own [0, 1) bound")
            return 1
        if build(w):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
