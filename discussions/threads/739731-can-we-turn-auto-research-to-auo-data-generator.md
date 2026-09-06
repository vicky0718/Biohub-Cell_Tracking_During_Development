# can we turn auto research to auo data generator?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739731
- **Topic id**: 739731
- **Author**: hengck23 (GRANDMASTER)
- **Posted**: 2026-09-05T19:21:43.157971200Z
- **Votes**: 4
- **Comments**: 5

---

## Opening post

i have a feeling that it can be done. Ask Codex to generate psf, then style it for fluorescence microscopy. We have a discriminator as a judge. it fits the loop generator --> test --> iterate.

this is just psf generator

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F9fe94cd69cb343584fb191a0cde22e8e%2FSelection_4781.png?generation=1788636058965350&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fe19f9ac9ec485441cb97a7349b86545e%2FSelection_4786.png?generation=1788636100528234&alt=media)

---

## Comments (4)


### Sergio Alvarez (MASTER) — 2026-09-05T20:48:27.127Z — 1 votes

I've tried the synthetic volume idea for ~2 weeks with no clear improvement in my scores, so I dropped it. Still, it would be cool to see this strategy work here, as it helped improve a little in the CZII competition. (Polnet was used there: https://github.com/anmartinezs/polnet)

I suggest to send DaXi microscope paper as reference (https://www.nature.com/articles/s41592-022-01417-2)

Here is one example of synthetic volume I tried to use:

 ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2221915%2F508c44ad1c064184a3a589b7de41f5dd%2Fsynthetic_z_stack_optimized.gif?generation=1788641223173538&alt=media)

#### ↳ hengck23 (GRANDMASTER) — 2026-09-05T23:02:44.757Z

> Your synthetic data quality is good! I will check the DaXi paper. Thanks!

### hengck23 (GRANDMASTER) — 2026-09-05T19:29:22.643Z

close to my idea  
https://arxiv.org/pdf/2107.10180  
3D fluorescence microscopy data synthesis for segmentation and benchmarking  
https://www.biorxiv.org/content/10.1101/2022.06.10.495713v1  
NISNet3D: Three-Dimensional Nuclear Synthesis and Instance Segmentation for Fluorescence Microscopy Images  


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fda81e458e6ac40760f6bd444de79eb82%2FSelection_4788.png?generation=1788636533698104&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-05T19:27:55.870Z

differentiable render
https://github.com/VirtualEmbryo/deltaMic?utm_source=chatgpt.com

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F4ca457bf73e3526165ae8d2e1a59bbd6%2FSelection_4789.png?generation=1788636465458910&alt=media)
