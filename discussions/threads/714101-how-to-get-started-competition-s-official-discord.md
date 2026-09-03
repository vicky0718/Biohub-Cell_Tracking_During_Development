# How to get started + Competition's Official Discord

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/714101
- **Topic id**: 714101
- **Author**: María Cruz (STAFF)
- **Posted**: 2026-06-25T16:44:43.592999100Z
- **Votes**: 27
- **Comments**: 8
- **Pinned**: yes

---

## Opening post

### Information for newbies
**New to machine learning and data science?** No question is too basic or too simple. Feel free to start your own thread, or use this thread as a place to post any first-timer clarifying questions for the Kaggle community to help you with!

**New to Kaggle?** Take a look at a few videos to learn a bit more about [site etiquette](https://www.youtube.com/watch?v=aIus8si_Et0), [Kaggle lingo](https://www.youtube.com/watch?v=sEJHyuWKd-s), and [how to enter a competition using Kaggle Notebooks](https://www.youtube.com/watch?&v=GJBOMWpLpTQ). Publish and share your [models on Kaggle Models](https://www.kaggle.com/docs/models#publishing-a-model)!

**Looking for a team?** Express your interest in joining a team through our [Team Up](https://www.kaggle.com/discussions/product-feedback/341195) feature.

**Remember**: Kaggle is for everyone. Whether you're teaming up or sharing tips in the competition forum, we expect everyone to follow our Kaggle community guidelines.

### Competition's Official Discord
In addition to this competition forum, you can continue the discussion in our official Kaggle Discord Server here:
# [discord.gg/kaggle](http://discord.gg/kaggle)
 
The Discord is a great place to ask getting started questions, chat about the nuances of this competition, and connect with potential team mates. Learn more about Discord at our [announcement here](https://www.kaggle.com/discussions/general/429933). Here are a few things to keep in mind though:

**1. Discord Competition Channels are 'Public' - Don't Share Private Information**

Discord channels for specific competitions are considered 'public' spaces where you are allowed to talk about competition details. Please remember that private sharing of competition code or data outside of your team is, as always, not permitted. Code sharing must always be done publicly through the Kaggle forums/notebooks.

**2. Discord Competition Channels are Not Monitored by Staff - Keep Important Information on the Kaggle Forums**

Kaggle Staff and Hosts running competitions will not monitor Discord or be available to answer questions in Discord. This is intended to be a more casual space to discuss competitions and help each other. Please keep important questions, insights, writeups, and other valuable conversation on the Kaggle forums. 

Happy modeling!

---

## Comments (8)


### Cho Royou (EXPERT) — 2026-07-17T15:45:54.417Z — 7 votes

Hi organizers,

This public notebook appears to exploit a weakness in how the official scoring script parses submitted submission.csv files, rather than reflecting genuine cell tracking accuracy:

https://www.kaggle.com/code/outwrest/metric-hack-minimal-baseline-tta-2gpu

The detection/tracking pipeline itself (TemporalUNet3D + node transformer + ILP solver, TTA, multi-GPU inference) is a legitimate baseline. The problem is a post-processing cell (augment_dataset) that runs after predictions are generated and modifies only the final submitted CSV — it does not affect the notebook's own local metric check, which is a strong signal it's specifically targeting the official scorer rather than improving predictions.

For each sample, it:

Splits the predicted lineage forest into connected components, sorts by size, and keeps up to MAX_COMPONENTS = 1400 of the largest trees.

Adds one synthetic "hub" node at sentinel coordinates (t=-1000, z=y=x=-10000 — clearly outside any real acquisition volume/time range) connected to the roots of those trees.

Appends a chain of 5 synthetic cell-division events (divider → child, continuation), also at sentinel coordinates (t≈-999..., z=y=x=-10000/-10001).

None of this corresponds to real detections — the coordinates are deliberately outside any physically valid range, so it can't be genuine matched signal under normal ground-truth matching. It looks designed to exploit something in how the scorer counts connected components / division events / normalizes its Jaccard-style metric.

This is a public, forkable notebook, so it may already be in use by other teams — I'd flag this as urgent. I forked/submitted a copy of it myself (submission ID 54786048, 2026-07-17) before fully realizing what the post-processing step did, and would like that submission disregarded/excluded from scoring.

Happy to provide more details if useful.

#### ↳ Jordão Bragantini (CONTRIBUTOR) — 2026-07-17T16:01:45.470Z — 3 votes

> Hi @ngyzly, thanks for bringing this to our attention. We are taking a look at it

#### ↳ ↳ outwrest (EXPERT) — 2026-07-17T17:56:41.693Z — 4 votes

> > Hi @jookuma, can you help patch this metric hack on the division_jaccard? Under the current four certeria (https://github.com/royerlab/kaggle-cell-tracking-competition/blob/7396b7e98e61844e799152ddda7e5493084cc8f3/metrics.md?plain=1#L62-L76), a division can count as a TP by connecting the predicted tracks to a fake fork outside the image, without acutally predicting the real division's location or time.
> > 
> > I found this trick and I am sure others did that were high on the scoreboard. The notebook shared is mine and I made it public so it can be verified & patched.

### Evdilos_Ikaria (MASTER) — 2026-07-03T09:26:08.437Z — 3 votes

Is there any available code for the applied combined tracking metric?
 Thanks

#### ↳ Tomás Travis (CONTRIBUTOR) — 2026-07-18T20:38:00.347Z — 1 votes

> Yes! It’s available in the [official GitHub repository](https://github.com/royerlab/kaggle-cell-tracking-competition). The metric itself is in [metrics.py](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/metrics.py), and the [README](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/README.md) shows how to run [scripts/evaluate.py](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/scripts/evaluate.py) locally against the training GEFF files. [metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md) also explains each part of the combined score.

### axorina (CONTRIBUTOR) — 2026-08-22T07:38:14.340Z

hi everyone im a 12th grader  and im looking for a beginner friendly team for biohub cell track if annyone is interested let me know!

### Asogwa Samson (CONTRIBUTOR) — 2026-07-31T16:53:46.037Z

Pardon me,
How long will it take for ones file to be processed?

### unknown — 2026-08-31T11:28:33.580Z

*(empty)*
