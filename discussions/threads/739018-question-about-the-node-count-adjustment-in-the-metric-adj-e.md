# Question about the node-count adjustment in the metric (adj_edge_jaccard can exceed 1)

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739018
- **Topic id**: 739018
- **Author**: Michael Hernandez (CONTRIBUTOR)
- **Posted**: 2026-09-02T14:03:48.861263500Z
- **Votes**: 3
- **Comments**: 2

---

## Opening post

Hi all, and thanks to the organizers for putting this competition together. The data has been a pleasure to work with.

I've been reading the released scoring code fairly closely and wanted to flag something in case it's unintended. In tracking_cellmot/metrics.py, the adjusted edge Jaccard is computed as:

```
adj_edge_jaccard = max(0.0, edge_jaccard * (1 - 0.1 * total_node_ratio))
```

where `total_node_ratio = (N_pred - N_est) / N_est`.

The `max(0.0, ...)` puts a floor on it, but there's no corresponding ceiling. So when a submission predicts fewer nodes than estimated_number_of_nodes, the ratio goes negative and the multiplier ends up greater than 1.

On one of the training clips (44b6_0b24845f) I get an edge Jaccard of 0.98 and a node ratio of about -0.29, which comes out as an adjusted value of 1.0102, a Jaccard above 1. The inline comment in the source describes the coefficient as a penalty, which is what made me think the symmetric bonus on the other side might not be what was intended.

I genuinely don't know whether this matters in practice, and I haven't been able to determine whether the scoring server behaves the same way as the released reference implementation. The difference between the two possibilities is smaller than the leaderboard's displayed precision, so I couldn't tell from the outside.

Entirely possible I'm misreading the intent here, in which case please ignore me. But it seemed better to ask than to sit on it. Happy to share the exact clip and numbers if that's useful.

Thanks again.

---

## Comments (2)


### hengck23 (GRANDMASTER) — 2026-09-02T14:14:33.800Z — 2 votes

yes. N_pred is a metric hack. From competition point of view: "it is better to detect just enough annotated nodes (up to N_est or less)" rather than all nodes". This change the way on how you treated unlablled data, apart from sparse labelled.

Obviously, the sparse annotation are not random tracks. they are difficult tracks (or tracks that are annotated by open source ultrack and hand corrected by host). you should expand sparse annotation to dense N_est annotation that is closest to sparse annotation. then train your model to detect N_est targets or less.

---

but of course, you can focus on division track and pay less attention on N_est. that is another separate strategy

### unknown — 2026-09-02T14:56:29.057Z — 1 votes

*(empty)*
