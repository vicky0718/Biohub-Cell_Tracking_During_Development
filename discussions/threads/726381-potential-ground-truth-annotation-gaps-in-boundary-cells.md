# Potential Ground Truth Annotation Gaps in Boundary Cells

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/726381
- **Topic id**: 726381
- **Author**: FasterYouChase FasterIRun (CONTRIBUTOR)
- **Posted**: 2026-07-15T08:22:12.441689200Z
- **Votes**: 5
- **Comments**: 0

---

## Opening post

## Potential Ground Truth Annotation Gaps in Boundary Cells

Hi everyone,

While analyzing the training data, I came across what appears to be inconsistent GT annotations for cells that travel along the **image boundary (y=0 edge)**.

### What I Found

In sample `44b6_12dfb391`, there is a cell tracked as **two separate GT lineages**:
- Track A: `t=21` to `t=24` (4 nodes), ending at `x=139, y=0, z=80`
- Track B: `t=29` to `t=37` (9 nodes), starting at `x=143, y=0, z=80`

The spatial distance between the end of Track A and the start of Track B is only **4.0 µm**, with identical `y=0` and `z=80` coordinates. The 4-frame gap (`t=25` to `t=28`) appears to be due to the cell **partially leaving the field of view** at the y=0 boundary.

When I visualized the raw image at `t=25–28`, the cell is still **partially visible at the edge** — my model correctly tracked it continuously (`t=19` to `t=33`), but is penalized because the GT treats it as two separate tracks.

### Why This Matters

The competition metric computes **Edge Jaccard** based on GT edges. If a cell is annotated as two tracks due to a boundary visibility gap, any model that correctly tracks it continuously will incur **false positive edges** for the gap frames and **false negative edges** for the missing GT connections — even though the biological interpretation is correct.

### Questions for Organizers

1. Is this a known annotation artifact for boundary-traveling cells?
2. Are there other samples with similar gaps?
3. Could the metric be adjusted to account for short re-entry events near image boundaries?

Happy to share the specific node IDs and visualization code if helpful.

Thanks!

---

## Comments (0)

*(none)*
