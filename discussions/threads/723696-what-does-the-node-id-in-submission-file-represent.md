# what does the node_id in submission file represent?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723696
- **Topic id**: 723696
- **Author**: NevilleAndrade (CONTRIBUTOR)
- **Posted**: 2026-07-07T20:39:50.977856900Z
- **Votes**: 1
- **Comments**: 2

---

## Opening post

In the submission file there is a field called node_id. Could you please explain what this id represents? 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F346284%2F237b5f8b8f5b9804a28814f748245d26%2FNode_id_question.png?generation=1783456560596473&alt=media)

---

## Comments (2)


### g john rao (MASTER) — 2026-07-09T02:05:25.447Z

the actual cell identifier is coordinates/time and not node_ids, node_ids are arbitrary numbers assigned for cell detections, and it is to link the rows as edges

#### ↳ NevilleAndrade (CONTRIBUTOR) — 2026-07-12T10:32:20.620Z

> Understood about the cell identifier being coordinate and time. When you say arbitraty node_id I suppose I can give that particular cell a unique identifier of say 500 instead of 1 and the next cell identifier say 501 correct?
