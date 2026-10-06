# Biohub - Cell Tracking During Development: the top 25 on the private leaderboard

Scraped 2026-10-06 from <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard> and each team's solution writeup.
17 of the top 25 published a writeup. Regenerate from the Biohub repo with `python discussions/scrape_writeups.py --out <this folder>`.

| Private | Score | Public (rank) | Team | Writeup |
|---:|---|---|---|---|
| 1 | 0.97759 | 0.97635 (3) | Sergio Alvarez | [Tracking-by-detection with a per-cell visual tracker](01-1st-place-solution.md) |
| 2 | 0.97041 | 0.96842 (28) | Soheil Ayati | [From Microscopy to Lineage Graphs](02-2nd-place-solution.md) |
| 3 | 0.96714 | 0.97898 (1) | yu4u | [Cell Tracking with 2.5D/3D Ensembles and Lineage Graph Optimization](03-3rd-place-solution.md) |
| 4 | 0.96234 | 0.96987 (20) | Barry | [3D Net detection, learned linking, and a division prior inside the ILP](04-4th-place-solution.md) |
| 5 | 0.95462 | 0.97167 (8) | Tang | [5th Place: 3D U-Net + Transformer Linker + Multi-stage ILP Tracking](05-5th-place-3d-u-net-transformer-linker-multi-s.md) |
| 6 | 0.95398 | 0.96285 (94) | Cyrus | [One network that detects and links cells, a global ILP, and teacher ? student rounds on 3 % annotation](06-6th-place-solution.md) |
| 7 | 0.95273 | 0.97195 (7) | tatsutaka | [3D Cell Tracking with Joint Detection and Motion Prediction, Dedicated Division Models, and Lineage Optimization](07-7th-place-solution.md) |
| 8 | 0.95190 | 0.95918 (158) | Amin | no writeup |
| 9 | 0.94981 | 0.96799 (30) | yuto083 | [9th Place Solution: Own Detectors + Public Tracker + Restored Divisions](09-9th-place-solution-own-detectors-public-tracker.md) |
| 10 | 0.94884 | 0.95809 (191) | Kaggle Gentlemen | [10th place solution: Grandmaster Powered Agentic Approach](10-10th-place-solution-grandmaster-powered-agentic-a.md) |
| 11 | 0.94743 | 0.94905 (1087) | tanbo | [11th Place Solution](11-11th-place-solution.md) |
| 12 | 0.94664 | 0.97091 (14) | Corwin | [A tuned public 3D U-Net + ILP tracker, extended by measured, fail-safe repair stages](12-12th-place-solution.md) |
| 13 | 0.94556 | 0.97111 (11) | The Boys | no writeup |
| 14 | 0.94493 | 0.96675 (43) | Vibes & Edges Trade-Off | [14th place solution from Vibes&Edges Trade-Off](14-14th-place-solution-from-vibes-and-edges-trade-off.md) |
| 15 | 0.94435 | 0.97059 (16) | Yapay Hücre | no writeup |
| 16 | 0.94340 | 0.96355 (75) | r3takahashi | [Self-Trained 3D U-Net Detectors on the Public Tracking Stack](16-16th-place-solution.md) |
| 17 | 0.94235 | 0.96696 (40) | ibyyue | [17th Place Solution: Gradient-Boosted Tracking + Metric-Labelled Graph Edits](17-17th-place-solution.md) |
| 18 | 0.94188 | 0.97163 (9) | ymg_aq | [Few labels, many divisions: refining a public tracker](18-18th-place-solution-lineage-graph-refinement.md) |
| 19 | 0.94112 | 0.97113 (10) | slime | no writeup |
| 20 | 0.94085 | 0.95055 (1067) | Yurinchi | [ A Single Own Detector, Geometric Linking, and a Small Public-to-Private Drop](20-20th-place-solution.md) |
| 21 | 0.94000 | 0.96592 (51) | Lime1123 | no writeup |
| 22 | 0.93944 | 0.96342 (81) | 3D Research | no writeup |
| 23 | 0.93939 | 0.97475 (4) | taiseiu | [23rd Place Solution: Division Recovery on a Public Cell Tracker](23-23rd-place-solution-division-recovery-on-a-public.md) |
| 24 | 0.93925 | 0.96345 (79) | 0.948+ pls | no writeup |
| 25 | 0.93900 | 0.96282 (97) | zhuo wamg + AibePC | no writeup |

## Teams without a writeup

- **8th, Amin** (Amin (@aminvafaei)).
  - nothing public found
- **13th, The Boys** (Sergey Bryansky (@sggpls), Tony Li (@tonylica), Christoffer Thimsen (@lowasitgoes)).
  - nothing public found
- **15th, Yapay Hücre** (Deniz Özaydın (@deniz143wow), seymen altınel (@seymenaltnel)).
  - nothing public found
- **19th, slime** (Gaopeng Ren (@gaopengren), eijixxx (@eijixxx)).
  - nothing public found
- **21st, Lime1123** (Lime1123 (@lime1123), sghwr (@songhow), kevin park (@k3v1npark)).
  - forum post by a team member: [What's the probability we'll witness a massive shakeup?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744273)
  - forum post by a team member: [Any ideas or successful approaches for model blending?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742131)
  - forum post by a team member: [what layer did ur gains actually come from](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543)
  - forum post by a team member: [does anyone have a different design for divisions](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737438)
- **22nd, 3D Research** (Huizhen Yang (@infocode), zeta kang (@knnzeta), lemin mang (@mlemin)).
  - nothing public found
- **24th, 0.948+ pls** (Tierney Wang (@tierneywang), Xianqi Yu (@xianqiyu), iprofen (@iprofen), Janet Sun (@janetsun0810)).
  - nothing public found
- **25th, zhuo wamg + AibePC** (zhuo wamg (@zhuowamg), AibePC (@xsmaxpc)).
  - nothing public found
