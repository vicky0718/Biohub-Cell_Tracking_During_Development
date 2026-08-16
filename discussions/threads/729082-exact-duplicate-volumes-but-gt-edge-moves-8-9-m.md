# Exact duplicate volumes, but GT edge moves 8.9 µm

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/729082
- **Topic id**: 729082
- **Author**: g john rao (MASTER)
- **Posted**: 2026-07-25T06:28:39.784938600Z
- **Votes**: 9
- **Comments**: 1

---

## Opening post

I found an example where two consecutive 3D image volumes are byte-for-byte identical, but the associated GT track moves substantially.

Dataset: `6bba_fc516dc6`  
Frames: `t=81` and `t=82`

`np.array_equal(volume[81], volume[82])` is `True`, and the subtraction contains zero nonzero voxels.

The GT edge is:

- `82001039`: `(t,z,y,x) = (81,23,175,169)`
- `83001053`: `(t,z,y,x) = (82,18,179,161)`

With voxel scale `(1.625, 0.40625, 0.40625)` µm, this is an 8.90 µm displacement. Both annotations appear to remain on the same elongated object, but the centroid changes despite the images being identical.

Since the matching radius is 7 µm, copying a detection across the duplicated frame may fail to match the second GT node.

---

## Comments (1)


### Bharat (CONTRIBUTOR) — 2026-07-28T02:47:18.860Z

I checked the other cells in 82nd frame, all have the same values as 81st except  this one.
This is actually a large cell, it's Z span itself is >15 voxels. 
If testset has such cells then even if our model predicts the right center it won't be able to cover both within 7µm.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2411334%2Fd53e3a1721c061c0eeda20b0392ed71c%2FScreenshot%20from%202026-07-28%2008-10-41.png?generation=1785206815250095&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2411334%2F36f3049e71523a9439788c2eb3c7c42c%2FScreenshot%20from%202026-07-28%2008-09-59.png?generation=1785206835736136&alt=media)
