# How can i reach a 0.9 score at least?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739220
- **Topic id**: 739220
- **Author**: Abhirup Choudhury (CONTRIBUTOR)
- **Posted**: 2026-09-03T07:55:55.088140100Z
- **Votes**: 1
- **Comments**: 11

---

## Opening post

Is everyone using the 3DTemporalUnet + CrossAttention with SImpleNodeTransformer? what is the training method and lose used here? same as the internal repo?

I am stuck on fixing paramters in the post processing part, in ilp and NMS

---

## Comments (11)


### hengck23 (GRANDMASTER) — 2026-09-04T10:08:29.180Z — 1 votes

if we are talking about a general solution for cell tracking (not restricted to the kaggle competition), to improve tracking, just increase the frame rate of the captured volume. Then you just need to have a good detector and nearest neighbour + linear assignment would have solved 95% of the problem.

#### ↳ Abed Merii (EXPERT) — 2026-09-04T19:52:17.620Z

> This would work assuming most cells have similar velocity, unless I'm missing something.

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-04T23:34:02.753Z — 1 votes

> > Imagine if you can improve the camera and capture image at say 100 fps instead of 10 fps. Then the cell move very little between each frame and in-fact they may just overlap a lot. So you can track them by nearest cell association. The trick is to capture faster than the cell moves or changes(division). This is hardware solution, which of course may be just a fancy solution ( eg due to limitation or cost. High speed camera are incredibly expensive)

### nusrati (CONTRIBUTOR) — 2026-09-03T13:03:05.090Z — 3 votes

the more you hands on the more you understand and only then its high chance you hit .9

there are many models on hugging face + already shared here publicly too. The one approach you mentioned is what evey LLM suggests upon given prompt.

understand problem first 
gothrough public notebooks second
broaden your exposure third
apply test, apply test, apply test, finally

### Yassine Alouini (GRANDMASTER) — 2026-09-03T08:36:32Z — 1 votes

Try to take advantage of this competition to learn new things (detection, tracking, post processing, 3D viz, etc) and have fun most importantly. Don't chase public score for now as suggested @rustambazarbayev. Good luck and enjoy!

#### ↳ Abhirup Choudhury (CONTRIBUTOR) — 2026-09-03T09:07:31.017Z — 1 votes

> Yeah i get that but i am genuinely lost. All the notebooks publicly shared are just inference notebooks at this point. I want to look into the training methods too

#### ↳ ↳ Rustam Bazarbayev (CONTRIBUTOR) — 2026-09-03T10:00:51.667Z

> > Try model separate into several parts like object detection linkage...etc

#### ↳ ↳ Yassine Alouini (GRANDMASTER) — 2026-09-03T10:06:45.710Z — 1 votes

> > if you have access to an LLM, ask it to explain one training notebook. From there, keep digging until you understand what the notebook does. Then, ask the LLM again for ways to improve. Keep iterating, explore the data, and be creative.

### Rustam Bazarbayev (CONTRIBUTOR) — 2026-09-03T08:14:07.053Z

Don't chase public score

### unknown — 2026-09-04T22:51:57.370Z

*(empty)*

#### ↳ Abhirup Choudhury (CONTRIBUTOR) — 2026-09-05T17:16:43.437Z

> thank you, this really puts things into perspective
