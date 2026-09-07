# Metric problem

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739686
- **Topic id**: 739686
- **Author**: Antonoof (GRANDMASTER)
- **Posted**: 2026-09-05T13:05:50.482469100Z
- **Votes**: 1
- **Comments**: 4

---

## Opening post

Two submissions, identical pipeline, identical hyper-parameters, differing only
in a post-processing step that deletes predicted tracks:

![image](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17954496%2Fe8b608858dd4a42fa7df29d929e6fe60%2Fcells.png?generation=1788613072033596&alt=media)

```
              nodes      edges     public LB
  full      121,782    117,187         0.954
  reduced    67,292     65,651         0.954
```

The second deletes 44.7% of the predicted nodes and 44.0% of the edges. The
public score does not move to three decimals. Two observations follow, and I
think the second matters more.

1) metrics.md and the scoring server appear to disagree.

  "To penalise false-positive node predictions, the edge Jaccard is scaled by
   a penalty on the total number of predicted nodes"
```
  adjusted_jaccard = max(0, jaccard * (1 - a * (T_pred - T_true) / T_true))
```

But as written the factor is two-sided. It is floored at zero and unbounded
above, so a submission with T_pred < T_true is multiplied *up*. In the reduced
submission above every video sits near (T_pred - T_true)/T_true = -0.5, i.e. a
factor of 1.05, and the weighted factor goes from 1.0009 to 1.0500.

Has anyone encountered this problem? We have achieved (in our opinion) good results over the past week, but we have not seen any changes in the PB metric, although the improvements seem better from our side.

---

## Comments (4)


### Sergio Alvarez (MASTER) — 2026-09-05T17:48:40.707Z

Hey @antonoof, you’re sure that your submission removed the edges? Every node/edges removal I've done reflected public lb/cv score. 
About the unbounded score, the hosts are aware. They've commented in other posts ready, its the expected behavior

#### ↳ Antonoof (GRANDMASTER) — 2026-09-05T18:48:28.507Z

> I deleted the ones that metric ignores by its own definition. And there is an experiment that shows this without a single controversial point: remove only edges, do not touch nodes — then the node counter and multiplier do not change at all, and any shift would be purely edge-based.

#### ↳ ↳ Ogurtsov (MASTER) — 2026-09-06T14:11:59.200Z

> > Did you delete non-ground-truth nodes/edges from predictions? It's easy to implement for training data, but how should it work on LB? Removing of random half of nodes/edges will reduce score a lot.

#### ↳ ↳ Antonoof (GRANDMASTER) — 2026-09-06T14:40:34.393Z

> > maybe I was mistaken. I tried deleting edges, deleted them more than 1/2, the metric did not change, deleted 2/3, the metric did not change. Okay, I think it's just me, I'll share the results after the competition.
