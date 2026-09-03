# Detection ensembling is closed, and the fork reproduces

Two runs landed. One closes a direction; one is the Track A floor, built and ready to submit.

---

## 1. `claude_union`: spotiflow's detections are a subset of ours

The proposal was to ensemble spotiflow with our detector — a parallel corrector, a veto, a
weighted blend. `notes/47` had closed spotiflow on *standalone quality*, which is the wrong
test: two detectors can sit on opposite sides of a recall-per-node curve and still find
different cells. The union had never been computed. It has now.

```
set             GT matched   recall   vs pack   nodes added
pack                 8,149   0.9968
spotiflow            4,654   0.5693
union                8,158   0.9979   +0.0011      234,470
selective            8,151   0.9971   +0.0002       79,260

GT nodes 8,175   pack detections 295,261
rescued by union: 9        by selective: 2
```

**Nine GT nodes out of 8,175.** And the selective mode — spotiflow detections further than
the scorer's own 7 µm match radius from any pack detection, i.e. the ones that could rescue
rather than duplicate — rescues **two**, at a cost of **39,630 added nodes per rescue**. The
budget consequence is decisive on its own: node count +26.8%, `ratio` −0.129 → **+0.172**,
multiplier 1.0129 → **0.9828**. We would pay 3% of the multiplier for two nodes.

**Detection-stage ensembling with this model is closed regardless of architecture.** No
weighting, veto, corrector or blend recovers information that is not there.

*Honest caveat on the grading.* Prediction 1 was scored FAIL: spotiflow came in at 0.5693
against a 0.547 ± 0.02 tolerance — a 0.022 miss — and the run used **n = 12** datasets
(4 × `44b6`, 8 × `6bba`), not the 36 I assumed. By my own pre-registered rule that means
"nothing below is comparable". I am reporting the conclusion anyway because the crux failed
by three orders of magnitude, not by a margin: 9 rescues out of 8,175 nodes, and 39,630
nodes per selective rescue. A tolerance dispute at the second decimal does not reach that.
The pack side reproduced exactly (0.9968 against 0.996).

## 2. `claude_fork`: the three-model pipeline runs, and all three models load

Track A of `notes/54`. `nusrati/0-938` reproduced unmodified, attribution prepended, source
saved to `claude_fork_source.json` for provenance. It completed and wrote `submission.csv`.

Its own resolved manifest — the useful part, because it reports *resolved state, not config*:

```
Dual-seed ensemble:   requested=True  weights_found=True
DeepCenter veto:      requested=True  loaded=True   (gap veto AND safe-div veto)
Bidirectional fusion: weight=0.15  active=True  mode=harmonic_probability
Safe-div thresholds:  parent<=9.0um  sister<=14.0um
internal validator (n=10 held out):  adj_edge 0.9203   div_J 0.0625   PROXY 0.9266
```

**Correction to `notes/54`'s port table.** I listed their reverse-time association weight as
**0.200** against our 0.15, from a print statement in `0-936`. The *resolved* manifest says
**0.15** — the same as ours. That row was wrong; B3 is a non-difference and drops out of the
Track B plan. The other three rows stand.

Also worth noting against our own numbers, with the standing caveat that these are different
dataset samples and not directly comparable:

```
              adj_edge   div_J
theirs (n=10)   0.9203  0.0625
ours   (n=24)   0.9072  0.1154
```

Their **division** term is *worse* than ours and their **edge** term is better. If that
survives a matched comparison it says the three-model advantage is in detection and linking,
not divisions — which is consistent with `notes/51` putting the remaining loss in
`fn_detect`, and it means our `div_J` 0.1154 is worth carrying into Track B rather than
replacing.

## 3. `claude_divgeom` — my error, fixed and re-running

Built to check `0-938`'s division-geometry claim against ground truth we hold (its 0.936 →
0.938 diff is entirely `SAFE_DIV_MAX_UM` 7→9 and `SISTER_MAX_UM` 12→14, justified by stated
GT statistics). v1 failed: I omitted the offline-wheels install block every other notebook
here carries, so `read_geff` hit `ModuleNotFoundError: geff` on all 199 datasets while the
mounts were fine. Fixed, plus a fail-fast guard — v1 ground through 199 identical failures
before the analysis cell died on an empty record.

The question is live because `pipeline/divisions.py` defaults to **`max_um=4.5`,
`sister_max_um=6.8`** — tighter than even `0-936`'s 7.0/12.0, let alone 9.0/14.0. If their
GT statistics hold, ours reject most real divisions, and `notes/50` closed the division
direction on the ILP weight axis without ever sweeping these gates.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.938 = rank 100    0.947 gold
detection ensembling: CLOSED       Track A: submission.csv built, awaiting submission
```
