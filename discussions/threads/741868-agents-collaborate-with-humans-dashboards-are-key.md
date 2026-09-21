# Agents collaborate with Humans - Dashboards are key!

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/741868
- **Topic id**: 741868
- **Author**: Chris Deotte (GRANDMASTER)
- **Posted**: 2026-09-17T21:50:19.517781600Z
- **Votes**: 18
- **Comments**: 7

---

## Opening post

We are entering a new era of humans collaborating with LLM agents. I think one of the most valuable techniques will be asking LLM agents to make dashboards to show humans visually what they are doing. Then we humans can review the visualizations and make suggestions to the LLM agents:

Below is a 3D movie visualization from a dashboard my LLM agent made for me to review its model's predictions. The 3D uses quaternions and allows me to rotate, zoom, and pan to see its predictions.

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Sep-2026/dashboard2.png)

---

## Comments (7)


### hengck23 (GRANDMASTER) — 2026-09-20T08:18:28.930Z — 3 votes

3d visulisation is very difficult. LLM/Agent has been a great help to me. actually the NAPARI viewer has an MCP to talk to an AI agent it think

#### ↳ Chris Deotte (GRANDMASTER) — 2026-09-20T08:34:21.717Z — 1 votes

> Yes. Back in my graduate post doc work, I helped write some web GUI 3D viewers using quaternions and WebGL. It tooks weeks to get working. We had to get both the math and code working. And now agents can write it in minutes. It is absolutely incredible!

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-20T08:45:29.867Z — 3 votes

> > i actually think rna/dna folding by ai agent will be interesting for a Kaggle competition.  Most of the top CASP results are handcrafted and selected by humans. But I think this will change.

### Chris Deotte (GRANDMASTER) — 2026-09-19T08:39:20.743Z — 5 votes

I would also like to point out that @tom99763 does a great job collaborating human and AI. He gives great suggestions [here][1]. Specifically he asks the agent to make materials to help him (the human) understand what the AI is doing. He posts some of his resources [here][2] (text explain) and [here][3] (code explain)

(and he did similar things in Wellbore competition [here][4])

[1]: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/718110#3486814
[2]: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/717109
[3]: https://www.kaggle.com/code/tom99763/tracksdata-tutorial-evaluation-mock-tests
[4]: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424

#### ↳ Vaibhav Nakrani (CONTRIBUTOR) — 2026-09-19T12:00:27.247Z — 1 votes

> thanks for sharing this. appreciate it.

### Vaibhav Nakrani (CONTRIBUTOR) — 2026-09-18T06:31:37.763Z — 1 votes

definitely. the other day i was thinking a similar viz tool for models as well. so we ask these models to try some custom model families right, it would be great to see a forward pass visually -- how the features move through and then we humans can add to it.

#### ↳ Chris Deotte (GRANDMASTER) — 2026-09-18T12:14:44.647Z

> Absolutely. We could click a button and pop out a page showing visually the architecture and preprocess of each experiment pipeline
