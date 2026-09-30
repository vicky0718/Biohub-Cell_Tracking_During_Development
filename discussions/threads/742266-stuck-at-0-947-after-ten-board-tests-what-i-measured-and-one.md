# Stuck at 0.947 after ten board tests. What I measured, and one question about the relink stage 

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266
- **Topic id**: 742266
- **Author**: Justin CH123 (CONTRIBUTOR)
- **Posted**: 2026-09-21T03:39:55.992547600Z
- **Votes**: 5
- **Comments**: 14

---

## Opening post

Hi all. I have been stuck on the 0.947 plateau for a week. I wanted to share what I measured in case it saves someone time, then ask one question.

I tested ten single-knob changes on top of the public Harmonic Fusion pipeline and every one lost on the board. The clearest pattern was that the loss tracked the change in predicted node count on the test set, not my offline validation score. The change with my best offline edge score (+0.0011) scored worst on the board (-0.005), and it was also the one that dropped the most nodes on one test video. A change that moved the node count by only four nodes still lost 0.001. For this pipeline, offline edge J on the training videos was not a usable signal for me.

I also noticed that in the public post-processing the final edges come from the Hungarian motion relink in filter_output_graph, which replaces the ILP edge set. The network probability only enters that cost through the learned bonus, worth at most about 1 um against a median true step of about 1.8 um.

My question is whether anyone found a gain at that relink stage specifically, for example feeding dense edge probabilities into the relink cost, without changing which cells are kept. More broadly, for anyone above 0.95 willing to say, was most of your gap over the public stack on the detection side or the linking side?

Thanks, and good luck in the last week.

---

## Comments (14)


### OmerZalman (CONTRIBUTOR) — 2026-09-21T20:24:20.360Z — 1 votes

im 0.95 so im below the threshold, (0.958 soon maybe), but I think that you cant just be lazy and improve one thing you have to improve all parts and a guy who was like 0.969 told me this as well 👍

#### ↳ Justin CH123 (CONTRIBUTOR) — 2026-09-22T11:30:19.927Z — 1 votes

> Thanks, that matches what I am seeing. Changing one knob at a time on this stack only ever lost for me. Could you say which part moved you first from 0.947 to 0.95, the detector, the linking, or the post-processing? Even a rough answer would help me decide where to spend the last week.

### OpPrime (CONTRIBUTOR) — 2026-09-21T18:52:06.420Z — 1 votes

I’ve had a similar experience investigating the 0.947 pipelines. I haven’t found a reliable improvement to the division errors either.
In the cases I traced, the problem wasn’t confined to one stage: some correct mother–daughter links were missing before ILP, some were removed during selection, and subsequent relinking or validation could prevent recovery. My attempts to change the association logic have not produced a dependable gain.

The base U-Net seems fairly strong at locating cells in the annotated cases I examined. Identifying which cells belong together over time is much less straightforward. Results that looked promising with GT-centred candidates deteriorated substantially when using actual detector outputs. Crowded neighbourhoods seem particularly difficult, although I can’t attribute the entire gap to density.
I also trained an auxiliary density/localization model, but its encouraging controlled results did not translate into a reliable improvement in the working pipeline.

Have you spent much time watching the raw videos? I find this a fascinating challenge because, even as a human, I often struggle to follow individual cells through those crowded regions. It leaves me wondering how much of the remaining problem is association logic, how much is visual information we haven’t learned to extract, and how much is ambiguity in the recordings themselves.

I’d also be very interested to hear whether gains above this plateau came mainly from detection or linking.

#### ↳ Justin CH123 (CONTRIBUTOR) — 2026-09-22T11:31:54.900Z

> Thanks for the detailed trace, it is very close to what I found. On divisions, in the public post-processing the mother to daughter links are added back after the Hungarian relink by the safe-division step, so a miss can come from the candidates, the relink, or that repair. When you traced them, roughly what share was missing before the ILP versus removed later? I have not spent much time watching the raw videos, but your point about crowded regions makes me think I should.

### nusrati (CONTRIBUTOR) — 2026-09-26T20:15:54.697Z

hey how did you get that jump. would be grateful for the guide!

### You WeiLin (CONTRIBUTOR) — 2026-09-23T13:00:06.037Z

I think public notebooks are somewhat useful for reference, but their approaches are overengineered. This ends up lowering the ceiling for further improvement, and the implementations are not polished enough.

### Civitasmass (EXPERT) — 2026-09-22T20:17:42.160Z

I was stuck at 0.947 for a month.
 
I think it's important to look beyond what you're currently doing.
 
And if you dont have GPU, it's really hard to improve.

#### ↳ Civitasmass (EXPERT) — 2026-09-22T20:22:36.490Z

> and I also use public notes, but based my local cv I feel that PB overfitting risk is really high.

#### ↳ ↳ Gourav Roy (CONTRIBUTOR) — 2026-09-22T22:49:35.063Z

> > did you retrain the detector ?

#### ↳ ↳ Justin CH123 (CONTRIBUTOR) — 2026-09-23T00:12:36.163Z

> > No, I have not retrained the detector, only post-processing changes so far. Did you retrain, and did it actually move your board score? Also, do you know roughly what your division Jaccard is? Mine is about 0.23 on held-out train videos and I now suspect that is where my gap is, not the edge term.

#### ↳ ↳ Justin CH123 (CONTRIBUTOR) — 2026-09-23T00:13:29.957Z

> > Thanks, that is useful to hear from someone who got past it. Two questions if you are willing. When you moved off 0.947, was it the edge term or the division term that improved? And when your local CV suggested high private overfitting risk, was it the division part diverging from the board or the edge part? I ask because my division Jaccard is only about 0.23 on held-out train videos while my edge term looks fine, so I am trying to decide where to spend the last week.

#### ↳ ↳ Civitasmass (EXPERT) — 2026-09-23T04:02:52.477Z

> > >did you retrain the detector ?
> >  
> > yes i did it, but not all of the LB gain from the retrain.
> >  
> >  
> >  
> > @justinch123 
> > To answer your questions:
> >  
> > ・edge vs. division: because LB only gives a single total score, I honestly don't know which term pushed me past 0.947.
> >  
> > ・overfitting risk: some changes gave big local boosts but barely moved the board. I still don't know the exact reason yet.
> >  
> > ・division Jaccard: in my opinion,  I wouldn't assume 0.23 means division is your main bottleneck. one thing that really helped me was checking whether local gains are spread across all videos or just driven by 1–2 outlier cases.
> >  
> > Hope that helps!

#### ↳ ↳ Justin CH123 (CONTRIBUTOR) — 2026-09-23T05:13:16.137Z

> > Thanks, that per-video check is a great tip. Two follow-ups if you don't mind: of the gain that didn't come from the retrain, was it mostly on the detection side, like where cell centres land, or on the linking side? And when you check whether a gain is spread across videos, what bar do you use before trusting it, something like "most videos improve"?

### unknown — 2026-09-23T10:19:09.893Z

*(empty)*
