# GEFF node coordinates alignment with Zarr image volume

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/733389
- **Topic id**: 733389
- **Author**: Estee (CONTRIBUTOR)
- **Posted**: 2026-08-06T18:29:06.877668200Z
- **Votes**: 2
- **Comments**: 1

---

## Opening post

Hi!

I'm new to this. I am trying to overlay `.geff` ground-truth nodes on the `.zarr` microscopy volumes.

Example training sample:

* Volume shape: `(T,Z,Y,X) = (100,64,256,256)`
* Node: `node_id: 1000004` @  `t=0, x=21, y=28, z=15`

Plotting this coordinate on the corresponding z-slices (`z=12–18`) places the **marker in a dark background area** near the image corner, not on a visible fluorescent cell.

I tested `(x,y)`, `(y,x)`, and flipped variants, but no transformation consistently aligns the annotations.

Can anyone confirm whether GEFF `nodes/props/{t,z,y,x}` coordinates map directly to the Zarr `(t,z,y,x)` voxel coordinates, or is an additional axis/origin transformation required?
Thanks!

---

## Comments (1)


### Shree Dharshan (CONTRIBUTOR) — 2026-08-07T06:10:14.667Z

Hi! The GEFF coordinates map directly to the corresponding Zarr array as:

```python
image = zarr.open_group(zarr_path, mode="r")["0"]
voxel = image[t, z, y, x]

```
There is no additional axis swap, flip, or origin transformation. The metadata uses `(T, Z, Y, X)`, and the GEFF properties correspond directly to `t`, `z`, `y`, and `x`.

For an XY overlay:

```python
plt.imshow(image[t, z])
plt.scatter(x, y, c="red")

```
The voxel scaling is (Z, Y, X) = (1.625, 0.40625, 0.40625) µm/voxel, but that scaling should only be applied when calculating physical positions or distances not when indexing the array.

If the marker appears in a dark corner, I would first check that the ``.geff` `and` `.zarr`` files have exactly the same sample name, `t` is read from `nodes/props/t/values` rather than inferred from `node_id`, the image is opened from Zarr dataset `"0"`, you are plotting `(x, y)`, not `(y, x), physical scaling has not been applied before array indexing.

It can also help to inspect a small 3D crop around (z, y, x) using percentile based contrast, since the annotated centroid may not be the brightest individual voxel. We verified the direct mapping in the XY, XZ, and YZ views and found that the annotations aligned with the visible fluorescent structures.
