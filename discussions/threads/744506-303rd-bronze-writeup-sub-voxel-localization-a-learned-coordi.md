# 303rd / Bronze writeup: sub-voxel localization, a learned coordinate head, and a lot of dead ends

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744506
- **Topic id**: 744506
- **Author**: Eesh saxena (CONTRIBUTOR)
- **Posted**: 2026-09-30T04:53:03.566405100Z
- **Votes**: 0
- **Comments**: 0

---

## Opening post

Bronze at 303/4017. Thanks to the organizers, and to the people whose public notebooks I started from, this is built on their work more than mine. I'll try to be honest about what actually moved the score, because most of what I tried didn't, and that's probably the more useful thing to share.

Where I started
Like a lot of people I began from the public temporal-UNet3D detector plus the ILP tracker, sitting around 0.947. From there I only really found two things worth keeping, and then spent days confirming that almost nothing else helped.

The thing that mattered: sub-voxel localization
The base pipeline puts every detection on a downsampled voxel grid, so all the coordinates are quantized. Instead of taking the argmax voxel, I did a soft-argmax over the detector logits in a small window around each peak, clipped to stay inside the voxel, done separately for the in-plane axes and for z. In plain terms: nudge each detection off the grid toward where the logits actually think the centre is.

This helps here specifically because of how the metric works. Edges are matched at a 7 um gate, and adjusted edge Jaccard rewards edges whose two endpoints both land inside that gate. A consistent sub-voxel correction drags the borderline edges back inside it. That was worth roughly +0.006 to +0.007, the biggest honest lever I found, and when I ablated it the z part carried most of the gain.

z is also where I think there's still room. A 3-point logit soft-argmax overshoots on z, and I never found a z estimator I was confident was better. If someone cracks that cleanly I'd expect a real bump.

Building on it
I ran two versions that share the detector and tracker but disagree on how they place coordinates:

One uses the public V1284 learned coordinate head for y/x, and my sub-voxel refinement for z. Public 0.955.
The other uses my sub-voxel refinement for both y/x and z. Public 0.954.
One detail that turned out to matter: I fed the refined z back into the edge model, not just into the output file. Refining coordinates and then handing the tracker the refined z (so it changes which edges it trusts) was worth about +2 ticks, versus refining only the numbers you write out at the end.

On the tracker side I tuned the motion relinker: a learned-edge bonus that I raised in steps (1.6, 2.0, 2.5, each one nudged my full-precision standing up), and a rule I called rawlink that lets high-confidence raw ILP edges override conflicting final edges, with divisions protected (adapted from a public V1057 post). Those two took the sub-voxel body from 0.953 to 0.954 and the learned-head body from 0.951 to 0.955.

The part I'd actually pay attention to: validation
Two things in the data description reset how I thought about all of this. The four visible test movies are placeholders copied from train, so local scoring on the public split tells you nothing. And train and test are embryo-disjoint, the hidden movies come from embryos nobody has seen, so leave-one-embryo-out is the only proxy that makes sense and any per-movie trick is useless.

So I built a replay harness that reruns the real notebook cells (real detector output, real ILP, the exact official scorer, which I ported call for call) on GT-rich train movies, roughly a minute a movie, paired bootstraps across the two embryos. It let me score a config change without spending a leaderboard submission.

The most useful thing it gave me wasn't a ranking of ideas, it was a calibration and a bias. Calibration: board delta was about replay delta minus 0.8 ticks on the sub-voxel body. Bias: the harness runs on trained movies, so it literally cannot see the y/x localization effect that the hidden embryos would feel, and it over-trusts anything that leans harder on the learned edges, because the edge model was trained on all 199 train movies. My other body read about -10 ticks on replay and was only -1 on the actual board. Once I knew that, I read every positive replay number as an upper bound rather than a result.

Blunt version: an offline harness built on an in-sample detector will call board-transferable localization changes wrong and over-rate anything that trusts the ILP more. The leaderboard, with its 6 to 8 hour latency and 5 reads a day, was the only instrument that was ever actually right. Budget your submissions around that.

What didn't work
Most of it. A few that looked especially convincing before they died:

Image-flow link repair was my best offline lever all week, big positive on replay and on LOEO, and it tied or lost on the board every single time. That's the one that taught me the harness's ILP-trust bias.
Blending the two coordinate estimates (half learned head, half sub-voxel) scored below both parents, because the two estimates are correlated around 0.8, so averaging buys almost nothing.
ILP division weight 0.4, which a public post reported as +0.001, was negative once I replayed the full path, the extra forks land on noise.
The public retrained localization head plus z-refiner made the >3 um displacement tail worse on both embryos, it optimizes a board-flat quantity.
Sub-voxel parameter sweeps (temperature, higher gains, per-axis gains), learned-head gain above 1, line-fit reweighting, short-track rescue tuning, node pruning, DET threshold moves, flow-inside-relink, and external models like Trackastra and DaXi, all neutral or negative.
Picking the final two
Kaggle keeps the better of your two on the private board, so I wanted them to fail independently. On an embryo-disjoint hidden set the scary failure is one coordinate scheme mislocalizing on an unseen embryo, so I picked one submission from each body, the 0.955 (learned head) and the 0.954 (sub-voxel), which differ far more from each other than any same-body pair would. I deliberately avoided picking two siblings that would sink together.

A last-day arm (bonus 3.0 plus a tighter relink gate) replayed +0.6 but came back 0.954 on the board, exactly the 0.8-tick gap I expected, so it didn't change anything.

Credit
The temporal-UNet3D detector and ILP support pack (pilkwang), the V1284 learned coordinate head, the rawlink idea from a public V1057 post, and the organizers' scorer. Happy to go deeper on the sub-voxel refinement or the replay-versus-board calibration if anyone's interested.

---

## Comments (0)

*(none)*
