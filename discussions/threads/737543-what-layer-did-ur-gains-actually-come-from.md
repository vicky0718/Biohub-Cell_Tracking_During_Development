# what layer did ur gains actually come from

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543
- **Topic id**: 737543
- **Author**: kevin park (CONTRIBUTOR)
- **Posted**: 2026-08-26T03:10:45.084626300Z
- **Votes**: 1
- **Comments**: 2

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

## Comments (2)


### Tang (MASTER) — 2026-08-27T00:36:00.903Z — 2 votes

improvements com from all of them.
it's hard to say which one matter most, but i think there's a "correct" order to work on them:
detection -> linking -> division.
detection should come first, once detection is solid, it's easier to improve others.

### abcbcbc (CONTRIBUTOR) — 2026-08-27T07:10:52.763Z

Could it be because the training data isn't good enough? The training data is too sparsely labeled, and I feel like that makes it very difficult to train a detection model.
