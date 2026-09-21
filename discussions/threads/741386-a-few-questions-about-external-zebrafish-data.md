# A few questions about external zebrafish data

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/741386
- **Topic id**: 741386
- **Author**: Masha Mikhisor (CONTRIBUTOR)
- **Posted**: 2026-09-14T22:10:13.125144900Z
- **Votes**: 3
- **Comments**: 3

---

## Opening post

Hi, thanks for confirming that Zebrahub data is OK to use. I'm looking for other embryos to check my tracker on and have a few questions:

1. Is the [public Ultrack embryo](https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr/.zattrs) (2024_03_22_dorado/stabilized.zarr) one of the competition train or test embryos? If it's separate, can I use it for validation, including manually checking tracks?

2. Can I download the curated tracks and reviewed DaXi lineages from Figs. 4–5 of the [Ultrack paper](https://www.nature.com/articles/s41592-025-02778-0)? Are they in [tracks_benchmark](https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/tracks_benchmark/), and which images do they match? If they aren't public yet, could they be shared with everyone?

3. What's the time between competition frames? How were the annotations made, and how is estimated_number_of_nodes estimated?

Thanks!

---

## Comments (3)


### Sergio Alvarez (MASTER) — 2026-09-15T12:02:31.787Z — 1 votes

About question 1, they are not the same embryos used in training, but they are very similar and the spacing is the same. You can download it to check it out. If you want to use it, you just have to extract chunks from it because it's not the same 64×256×256 size as in the competition. Also, it wouldn’t make sense for it to be in the test set. It's standard in Kaggle competitions that the hidden test set is never shared anywhere, it's a new set images that the hosts have never shared, but prepared using the same methodology as the training data.

As far as I know the hosts never shared how tracks were made in discussions, I think it's based on their methods (ultrack?/hoct?) + manual review

### MOHAMMADJAFAR ZAMANI (CONTRIBUTOR) — 2026-09-17T18:40:37.507Z

Could the organizers clarify whether the public Janelia/linajea zebrafish recording **160328** is independent of every competition training and hidden-test embryo and is permitted for external validation?

For the Zebrahub/Ultrack data discussed above, which released recordings have independently reviewed sparse lineage annotations? A link binding each reviewed annotation export to its exact image recording/version, voxel spacing, frame interval and annotation coverage would help us evaluate trackers correctly.

Dataset reference: https://janelia.figshare.com/articles/dataset/Zebrafish_data_for_whole-embryo_lineage_reconstruction_with_linajea/24968724

### Navneet (CONTRIBUTOR) — 2026-09-15T13:36:06.680Z

Thank you for the external zebrafish data @mikhisor
