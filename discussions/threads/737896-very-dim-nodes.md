# Very dim nodes?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737896
- **Topic id**: 737896
- **Author**: weke (EXPERT)
- **Posted**: 2026-08-28T01:13:05.783744900Z
- **Votes**: 3
- **Comments**: 3

---

## Opening post

Hello,
I was looking though my false negatives and realised that some of the nodes cannot be seen properly and i was wondering whether thsoe where annotation artifacts or cells can actually be this dim.
I am looking at the track for node `53000565`. This node is specifically at slice 31 of frame 52 of the video `6bba_3a1849c2` and is supposed to be at the red dot in the following picture:


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4090870%2Ffa1d164dfc3ad63e2260eb90d996b038%2FScreenshot%20from%202026-08-28%2003-06-45.png?generation=1787879525897274&alt=media)

Its whole track looks like this. Can this be an artifact, or can cells actually look like this?

Regards

---

## Comments (3)


### hengck23 (GRANDMASTER) — 2026-08-31T00:39:19.730Z

It can be false positive or results of interpolation. Eg annotation label frame t=1 and t=3 and interpolate for t=2

### Navneet (CONTRIBUTOR) — 2026-08-28T04:05:31.593Z

Thank you for sharing your false negatives. @cjpal18

### unknown — 2026-08-28T03:55:59.157Z

*(empty)*
