# what layer did ur gains actually come from

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543
- **Topic id**: 737543
- **Author**: kevin park (CONTRIBUTOR)
- **Posted**: 2026-08-26T03:10:45.084626300Z
- **Votes**: 3
- **Comments**: 9

---

## Opening post

im at 0.933 and pretty stuck. top of the board is 0.962 so thats a big gap and i genuinely dont know what its made of.

heres what i can see. the public notebooks are all basically the same stack. learned unet then node transformer then ILP. i checked the configs on a few of the higher ones and theyre nearly identical to mine knob for knob. so the public field and me are all the same lineage sitting between 0.92 and 0.933.

but the top 10 is 0.945 plus. thats way outside that cluster so it feels structural not js better tuning.

what i actually wanna know is which LAYER ur gains came from. not the solution js the layer.

detection. did u retrain the detector or swap it for smth else entirely.

linking. is ur linker still 1:1 hungarian or did u replace it.

divisions. inside the assignment or bolted on after.

post processing. how much of ur score is stuff that happens after the graph exists.

or none of those and its js a lot more training on the base model.

for context im not fishing for a config. ive tuned every knob i have and every division route i tried came back closed. so im trying to work out if im stuck on the wrong layer entirely rather than missing a setting.

even js which one mattered most for u would help. thanks

---

## Comments (9)


### Tang (MASTER) — 2026-08-27T00:36:00.903Z — 3 votes

improvements com from all of them.
it's hard to say which one matter most, but i think there's a "correct" order to work on them:
detection -> linking -> division.
detection should come first, once detection is solid, it's easier to improve others.

#### ↳ kevin park (CONTRIBUTOR) — 2026-08-28T06:38:46.290Z — 1 votes

> thanks for replying
> 
> sry i didnt put this in the main post my detection recall is already 0.9945 so i dont think theres much left there for me
> 
> my real problem is divisions my divJ is 0.1176 and my adjJ is 0.9212 so basically all my score is edges
> 
> did fixing detection actually move ur divisions or did u need something else for that

#### ↳ ↳ Lime1123 (EXPERT) — 2026-08-28T06:59:04Z — -1 votes

> > I think some division-specific handling is needed for divisions beyond just improving detection recall. Recall still matters, especially at shorter distances, but a high overall detection recall by itself doesn't seem to translate directly into a good divJ.
> > 
> > In my validation, divisions seemed much more sensitive to spatial accuracy, roughly within 3 um, even though the competition metric uses 7 um. For reference, in LB, I got about 0.28 divJ with an adjJ around 0.89...

#### ↳ ↳ kevin park (CONTRIBUTOR) — 2026-08-28T07:34:38.537Z — -1 votes

> > tysm this helps so much
> > 
> > i went and checked mine and my division nodes really do localize worse than everything else 39 percent of them are outside 3um even tho all of them pass the 7um metric
> > 
> > gonna give it a try

#### ↳ ↳ kevin park (CONTRIBUTOR) — 2026-08-28T10:28:48.563Z

> > also is that handling a model thing or something after the graph
> > 
> > no worries if u dont wanna say. this already helped a lot

#### ↳ ↳ Tang (MASTER) — 2026-08-29T11:18:26.907Z

> > glad to help.
> > in my case, most of the gains come from model improvements.

#### ↳ Rishabh Roy (EXPERT) — 2026-08-31T08:05:40.273Z

> > improvements com from all of them.
> > it's hard to say which one matter most, but i think there's a "correct" order to work on them:
> > detection -> linking -> division.
> > detection should come first, once detection is solid, it's easier to improve others.
> 
> @hirotetsu Can you help how you improved the detection beyond the normal unet 3d detector . any suggestion would be great .

### abcbcbc (CONTRIBUTOR) — 2026-08-27T07:10:52.763Z

Could it be because the training data isn't good enough? The training data is too sparsely labeled, and I feel like that makes it very difficult to train a detection model.

### unknown — 2026-08-28T06:35:48.187Z

*(empty)*
