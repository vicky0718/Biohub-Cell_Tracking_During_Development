# Scoring timeouts: evidence it's graph connectivity, not size

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724917
- **Topic id**: 724917
- **Author**: Alan Thanickal (CONTRIBUTOR)
- **Posted**: 2026-07-13T15:44:50.310528600Z
- **Votes**: 0
- **Comments**: 2

---

## Opening post

Several threads mention submissions timing out during scoring. I've hit it three times and have some data that might narrow down the cause, plus a question for the hosts.

Setup: classical pipeline (Otsu/watershed detection → nearest-neighbour linking → graph repair). Predictions on the 4 test videos, node counts roughly matching estimated_number_of_nodes (over_ratio ~0.85–0.95, i.e. slightly under-predicting, not spamming detections).
What scores and what doesn't:

| pipeline | nodes | edges | edges/node | scoring |
| --- | --- | --- | --- | --- |
| raw detect + NN link | ~62k | fragmented | — | 4–5 hrs  |
| + gap-close + short-track | 61,942 | 58,786 | 0.95 | timed out  |
| + relink + gap-close + filter | 61,670 | 58,460 | 0.95 | timed out  |

The node counts are essentially identical across all three. The thing that changes is edge structure: the raw linker leaves a fragmented graph (many short, broken tracks), while the repair stages knit those fragments into long continuous tracks.

Hypothesis: scoring cost is driven by connected-component structure, not node count. A graph of N nodes in many short components appears to be much cheaper to score than the same N nodes in few long components — presumably the division/lineage analysis traverses components.

Things I ruled out:

Malformed graph — all edges span exactly 1 frame, max out-degree = 1, zero orphan edges, zero self-loops, zero duplicate edges, node IDs per-dataset starting at 1 (matching sample_submission.csv).
Over-detection — over_ratio < 1.0 on every video.

Node count as the driver — raising the detection threshold to cut nodes 17% costs ~0.18 edge Jaccard (there are no junk detections to remove; the node count reflects real cell density). Not a viable lever.

Happy to share more numbers if useful. If anyone else has a submission that scores with gap-closing/track-linking on the dense video, I'd love to know your node/edge counts for comparison.

---

## Comments (2)


### Youri Matiounine (GRANDMASTER) — 2026-07-23T11:50:31.030Z

this is not caused by scoring; local scoring on all train data runs just over 1 minute. All the other time goes into constructing your prediction. Time how long your code takes over test set of 4 videos, scale it up to approximate full test set - that is how long your submission will run.

### FasterYouChase FasterIRun (CONTRIBUTOR) — 2026-07-15T04:15:33.537Z

Are you considering the time complexities of the function you are using?
For 60K nodes, using function having time complexity O(N^3) will also cause the time out.
