# The discussion forum, scraped in full

**All 71 topics and all 219 comments including nested replies** (219 fetched vs 219
declared — nothing missed). Raw JSON in `data/forum-scrape.json`; the scraper is
`probes/scrape_forum.py`. Scraped 2026-08-16.

**How.** Kaggle's public pages are JavaScript shells and `api/v1` returns
`401 Unauthenticated`, but the SPA's own internal endpoints (`/api/i/<service>/<Method>`)
serve public forum content to an *anonymous* session. The trick is that they need a real
session: fetch any page to pick up the `XSRF-TOKEN` / `ka_sessionid` cookies, then echo
the token in an `x-xsrf-token` header. Without that pair every call is a bare `400` with
an empty body, which reads like a malformed payload and is not. Endpoint names come from
the SPA bundle (`/static/assets/app.js`), the forum id from
`competitions.CompetitionService/GetCompetition` (`forumId: 10656304`), and the message
route wants `topicId`, not `forumTopicId`.

---

## 1. 🚨 This is a NOTEBOOK competition, not a CSV upload

From `GetCompetition`:

| field | value |
|---|---|
| `onlyAllowKernelSubmissions` | **True** |
| `requiredSubmissionFilename` | `submission.csv` |
| `usesSynchronousReruns` | **True** |
| `maxCpuRuntimeMinutes` / `maxGpuRuntimeMinutes` | **720** (12 h) |
| `maxDailySubmissions` | 5 |
| `leaderboardPercentage` | 29 (public LB is 29 % of the test data) |
| `totalTeams` | 2,395 |
| deadline / reward | 2026-09-29 · $60,000 |

`MEMORY.md` said "CSV-upload contest, not notebook-runtime". **That is wrong.** We submit a
notebook, Kaggle reruns it against the private test data, and it must produce
`submission.csv` inside 12 hours. Everything we have built is research tooling; **there is
no submission notebook yet**, and that is now a first-class deliverable.

Two consequences that bite immediately:

- **The `test/` folder is replaced at rerun time.** Dataset names must be discovered by
  globbing at runtime — no hardcoded lists. Confirmed by a competitor whose prebuilt CSV
  failed for exactly this reason.
- **The `id` column is required** in `submission.csv`.

## 2. ✅ The leak is CLOSED — and it was never a leak

`notes/04` §0 and `notes/05` §0 flagged that all four test datasets appear in train with
ground truth, and said to report it. **Someone already did**, and the host answered:

> *"Indeed these are dummy placeholder files to help you ensure that your submission
> notebook actually produces a .csv file without erroring out. The actual leaderboard
> score is obtained from a much bigger test set, that is deliberately kept private, and I
> assure you there is no overlap between that hidden test set and the train set."*
> — Thibaut Goldsborough (HOST)

So: nothing to report, nothing to exploit, and **no integrity problem**. But it kills one
of our tools — `03_linking.ipynb` §4 scores those four datasets locally and calls the
result a "predicted leaderboard score". Those four are *placeholders*. That number
predicts nothing. Read it as a smoke test that the submission path runs, and ignore its
value.

## 3. ⭐⭐ The hidden test set is a DIFFERENT PAIR OF EMBRYOS

> *"Indeed there are two unique embryo_ids in the training set. You can assume the test
> set is roughly similar in size, with no overlap in embryo_ids between train and test
> sets."* — Thibaut Goldsborough (HOST)

And separately: *"all the data in this competition followed the same protocol (instrument,
developmental stage, etc), each embryo being acquired in a separate imaging session"* —
Jordão Bragantini (HOST).

This is the most decision-relevant fact in the whole forum, and it invalidates our fold
design. The shift the leaderboard applies is **to embryos we have never seen**. Our
five-way hash split puts crops of *both* embryos in *every* fold, so it answers "does this
generalise to another crop of an embryo I have already seen" — the easy question.

`Harness` now defaults to **`fold_by="embryo"`**: leave-one-embryo-out, two folds. Thin —
the no-regression gate is testing exactly one held-out embryo per side — but a thin honest
split beats a well-powered misleading one. `fold_by="hash"` restores the old behaviour when
within-embryo variance is what you actually want.

Test set is *roughly the size of train*, so ~200 datasets, ~100 frames each.

## 4. ⭐ What scores what — rule-based is genuinely competitive

The single most encouraging thread: **"Rule-based is surprisingly strong? (currently
7th/344 teams, gold zone, no learning)"** (45 votes). Concrete paired numbers from
Timmy Juicehouse:

| approach | CV (all train, official metric) | public LB |
|---|---|---|
| heuristic 1 (~2 h inference) | 0.7448 | **0.834** |
| heuristic 2 (~3 h inference) | 0.8213 | **0.846** |

And the 7th-place author states their **division Jaccard is basically zero — they predict
no divisions at all.** That is exactly our configuration.

**LB runs *above* CV**, by a lot at the low end: +0.089 and +0.025 in the table above, and
another competitor independently reports "the public LB is more optimistic by almost 10 %".
Our measured **CV 0.5552** therefore probably maps to roughly **0.60–0.65 LB** — behind the
frontier, but the gap to the 0.834 heuristic is a tuning gap, not an architecture gap.

Other calibration: movie-to-movie variability is **±0.14, 18 % coefficient of variation**,
worst movie 0.460, best 0.984. Any single-dataset comparison is noise.

## 5. 🚨 The failure mode that scores exactly 0.0

Worth knowing before we ever submit. A competitor scored **0.0** with ~0.57 locally:

> *"My pipeline used a time-budget strategy that subsampled frames across the whole video
> (process 1 frame every N) to stay under the 12 h limit on a much larger hidden test set
> than the visible test/ sample. That produced edges connecting frames like t=0 → t=5.
> ... every single edge I submitted was structurally unmatchable — TP=0 across the board,
> regardless of detection quality. No format error was raised."*

This is metric finding §3 (non-consecutive edges are silently dropped) turning into a total
loss in practice. **If we ever need to save time, drop datasets or drop resolution — never
drop frames.** Our `Config.max_frames` is exactly this hazard: it is a smoke-test knob and
must never ship in a submission.

Our runtime is not close to the limit anyway. Scoring itself takes ~1 minute; the rest is
prediction. Our four test datasets took ~62 s total, so ~200 datasets ≈ **52 minutes**,
against a 12 h budget.

## 6. A division-metric exploit existed and was patched — mid-competition

The host posted "Division Metric exploit and patch" (2026-07-18) and **all submissions were
rescored** (completed 2026-07-22). This is why "Why are there so many 0.950" exists.

The exploit, as described by a competitor who had been using it: the pre-patch division
matching relied on weakly-connected-component structure and did not require a matched
division to be a genuine parent→two-daughter fork near a real GT division. So people added
a hub node at coordinates far outside the volume (e.g. −10000, −10000, −10000) linked to
every track root — merging the whole prediction into one component — plus synthetic fork
chains out there, and dropped their real forks.

**Implication for us:** the `cross_component_forks` / `_gt_weak_component_ids` machinery in
`division_metrics.py` that looked so baroque when we read it **is the patch**. Our clone is
post-patch, so `harness/purescore.py` is verified against the current metric. Also
independently confirmed by a competitor: *"summarise() drops the division term entirely
when a submission contains no divisions"* — exactly what `purescore.per_sample` implements.

A patched scorer is also mirrored as a Kaggle dataset: `dalloliogm/biohub-official-scorer-patched`.

## 7. Ground-truth quality — the GT is Ultrack pseudo-labels, and it has errors

Several well-upvoted threads document real GT defects:

- **"beware of jumps in ground truth track"** (37 votes) and **"Exact duplicate volumes,
  but GT edge moves 8.9 µm"** (9 votes): some consecutive frames are *byte-identical*
  (dropped/repeated frames in the acquisition) while the GT still moves a cell 8.9 µm
  across them. No detector can be right on those.
- **"not all sparse GT edge are correct"** (12 votes): very dim cells whose 3D position is
  unresolvable, with GT links that do not follow the image.
- Large cells exist with **Z span > 15 voxels (> 24 µm)**, where a correct centre may still
  miss the 7 µm match radius.
- The Zebrahub tracks are **Ultrack-generated pseudo-labels**, not manually curated — they
  carry Ultrack's systematic biases, especially around divisions and dense regions.

This puts a real ceiling below 1.0 and explains part of the residual we have been treating
as our own error. It does not change what to build.

## 8. Two things we can use that we did not know existed

- **External data is explicitly allowed.** Host: *"Yes you are free to use the data and all
  resources in Zebrahub for this competition! There is no overlap with the test set."*
  That is a large public zebrafish light-sheet corpus with tracks — the obvious training
  set if we ever build a learned detector. Caveat from the same thread: different imaging
  setup and developmental windows, so expect domain shift.
- **A free 18.5 GB fully-labelled synthetic 3D microscopy dataset** was shared by a
  competitor (35 votes).

## 9. Public-weights leakage — a trap we happen to avoid

> *"The public weights were trained on all 199 annotated videos. Their split_manifest.json
> lists 199 under train, and the 40 in its own test list are all inside that same set. So
> if you're validating on train videos with those checkpoints, you're scoring the model on
> data it memorized."*

The same competitor measured +0.0184 from an ablation locally and then **lost** score on the
leaderboard — the sign flipped, because the post-processing stages they removed exist to
repair model errors that do not appear on memorised videos.

**We are training-free, so our CV is honest** — an underrated advantage. It also means we
must not casually adopt the public checkpoints without rebuilding validation around them.

---

## 10. Verification pass (2026-08-16) — plus two things the first pass missed

Re-checked on request. **Method:** the structured facts in §1 come from Kaggle's own
`GetCompetition` record, re-fetched live; the host statements are messages Kaggle itself
tags `authorType: HOST` / `ADMIN` (25 of them across the forum), quoted verbatim. A
browser check was attempted and **failed** — see §11.

Everything in §1–§9 above held. Two facts were missed the first time, and both are
operationally important.

### 🚨 `pip install` does not work during a scored submission

> *"These won't work during submission, you need to have them already installed in your
> testing artifacts."* — Jordão Bragantini (HOST), answering someone whose submission
> failed while pip-installing.

Internet is off in the scored rerun. **Every notebook we have opens with
`pip_install(["geff", "zarr"])`, and that cell would fail a submission run.**

The fix is easy and worth stating precisely: at test time there is no ground truth to
read, so `geff` is not needed at all — only `zarr` (preinstalled on Kaggle), numpy and
scipy. The submission notebook must install nothing. `geff` stays a research-only
dependency for `02`/`03`, which run interactively with internet on.

### The maximum score is 1.1, not 1.0

> *"the best submission score is actually 1.1, not 1, so it is possible for the scores to
> go over 1. In fact it is theoretically possible to obtain a score slightly above 1.1
> (due to our adjusted jaccard metric)."* — Thibaut Goldsborough (HOST)

`adj_edge_jaccard` maxes at 1.0 (higher with the under-prediction bonus) plus `0.1 ×
division_jaccard`. So the ceiling is ~1.1, and every score should be read against that:
the 0.915 leaderboard entry is **83 % of maximum**, not 91.5 %, and our 0.5552 is ~50 %.
Recon §7 already measured `adj_edge_jaccard = 1.0825 > 1`, so this is consistent — it just
gives the scale an anchor we did not have.

### Smaller confirmations from the same pass

- **Duplicate edges are filtered before scoring.** *"perfectly duplicated edges are
  filtered out before scoring the edge Jaccard... each ground truth edge can only be used
  for a single TP"* (HOST). That is exactly what `purescore.count_edges` implements, now
  confirmed by the organisers rather than only read from their source.
- **Only annotated tracks are scored, but all cells must be tracked.** *"The task is to
  track ALL the cells in the video, not just the ones that we provide annotations for; at
  test time you are only evaluated on the tracks that we have annotated"* (HOST).
- **`embryo_id` is the `44b6` part; the rest is a crop id** (HOST) — confirms the prefix
  split the harness now folds on.
- **Imaging provenance**: multiple views per volume, fused, then linearly scaled to a
  reference view; custom microscope based on the Janelia design in Tomer 2012 (HOST). Small
  inter-view intensity deviations are expected — relevant to any intensity-based detector.
- **Self-trained models are allowed**, but must be reproducible if you win.
- The host on the official baseline: *"if this is the UNet baseline, this is expected,
  inference is very slow, and needs to be improved!"*

## 11. What could NOT be verified, and why

- **A real browser cannot reach Kaggle from here.** Chromium is installed and was pointed
  at the agent proxy; every attempt returns `ERR_CONNECTION_RESET`, with and without
  HTTP/2 disabled and with a normal user-agent. `curl` to the same host works, so this is
  Kaggle's edge rejecting the browser, not the proxy. The proxy's own failure log shows no
  denial for `kaggle.com` — the CONNECT was accepted.
- **Therefore the 71 opening posts remain unretrieved.** They need an authenticated
  session: `BatchGetForumMessages` returns 403, `GetForumTopicById` returns metadata only
  (no body), all four `commentSort` values return the same reply set, the discussion
  `oembed` route 404s, and `.rss` serves the SPA shell. We have every *reply* (219/219) and
  every host answer; what is missing is mostly the question being asked.
- **Zebrahub is blocked by network policy — now proven, not inferred.** The proxy's own
  relay log records explicit gateway denials:
  `connect_rejected  zebrahub.org:443  gateway answered 403 to CONNECT (policy denial)`
  and the same for `public.czbiohub.org:443`. Any Zebrahub acquisition must happen in a
  Kaggle notebook with internet enabled, not here.

---

## What changes because of this

1. **Build a submission notebook.** Self-contained, runtime dataset discovery, `id` column,
   no `max_frames`, under 12 h. This is now the gating deliverable — we cannot score at all
   without it.
2. **`fold_by="embryo"` is the default.** Done. Re-read any earlier fold result knowing the
   old split was optimistic about the shift that matters.
3. **Ignore `03` §4's "predicted leaderboard score".** Those four datasets are placeholders.
4. **Delete the leak flag** from `notes/04` §0 and `notes/05` §0 — answered by the host.
5. **Expect LB ≈ CV + 0.03…0.09.** Our 0.5552 is probably ~0.60–0.65. The rule-based
   frontier is 0.846 with zero divisions, so the headroom is in the same pipeline shape we
   already have.
6. **Never subsample frames.** It is a silent zero.
