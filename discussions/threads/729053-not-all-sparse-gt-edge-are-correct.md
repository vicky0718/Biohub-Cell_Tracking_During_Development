# not all sparse GT edge are correct

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/729053
- **Topic id**: 729053
- **Author**: hengck23 (GRANDMASTER)
- **Posted**: 2026-07-25T04:01:20.815014700Z
- **Votes**: 13
- **Comments**: 1

---

## Opening post

Not all edge annotations are correct. But the error rate is very low. Here is an error detected by checking neighbouring cell motion

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F5cc8a0f35bc698acbb4d30de10f0977f%2FSelection_4399.png?generation=1784952074278647&alt=media)

---

## Comments (1)


### Bharat (CONTRIBUTOR) — 2026-07-27T07:24:05.757Z

I think the cell for track_id 73 is very dim and can't be located in 3d. With 2d slices from frame 15 to 16 the tracking is clear, but 16 to 17 have many changes in cell positions 
https://www.kaggle.com/code/bharat0/check-sample-6bba-1d0d8384?scriptVersionId=338241708

![Frame 15-16](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2411334%2F69f2427d5b18da710848da7d5b40f13c%2FScreenshot%20from%202026-07-27%2012-51-53.png?generation=1785137016207578&alt=media)

![Frame 16 - 17](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2411334%2Ff4bc5d01e077d9ac9f561c8aa8237dc7%2FScreenshot%20from%202026-07-27%2012-52-02.png?generation=1785137035868401&alt=media)
