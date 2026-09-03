#  Quick question for anyone above the 0.94 line — is the detector still a 3D UNet heatmap for you, or did you move to something else ? 

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738276
- **Topic id**: 738276
- **Author**: Rishabh Roy (EXPERT)
- **Posted**: 2026-08-31T00:29:58.711370700Z
- **Votes**: 4
- **Comments**: 5

---

## Opening post

 did you move to something else (instance seg, a different backbone)? Trying to figure out if the jump is a detection-layer change or all in the linker/divisions. Thanks!

---

## Comments (5)


### hengck23 (GRANDMASTER) — 2026-09-02T23:41:57.577Z — 1 votes

i suggest you go through ultra api. there are many track post processing tools. try to 
1) use some opensource cell instance segmentation to get instance labels
2) use it as input to ultrack
3) then use ultrack to link
4) use post processing tools to correct /TrackEdit

make some manual annotations to appreciate the problems of ultrack and what can be solved by post processing.
Do it a few rounds and i think you can discover improvement points and how to get more data

### Дворкин Евгений Владимирович (EXPERT) — 2026-09-01T15:05:12.047Z

Hello. I don’t really understand what’s going on here and how it works; I’d like to know whether you focus on PROXY_SCORE when submitting work for review? I’m just playing around with the parameters, and sometimes LB improves when PROXY_SCORE increases, and sometimes the opposite happens — it’s unclear what to go by. There was a moment when PROXY_SCORE was 0.943+.

#### ↳ Rishabh Roy (EXPERT) — 2026-09-01T19:25:05.637Z — 1 votes

> The local proxy is scored against sparse ground truth (only a fraction of cells/frames are annotated), while the LB is scored against dense labels. The key consequence: the proxy barely penalizes over-detection. So any change that adds volume — more candidate detections, looser linking, recall-oriented tweaks — can push PROXY_SCORE up while quietly adding false positives that the dense LB does punish. But a proxy gain does not guarantee an LB gain, so don't chase proxy peaks.

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-02T23:44:50.487Z — 1 votes

> > Thanks for the reply. I checked the forum and competition webpage, but where is it mentioned that " LB is scored against dense labels. " I think all are using sparse labels like the downloaded train?

### unknown — 2026-08-31T15:02:43.280Z

*(empty)*
