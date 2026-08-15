"""`Tracks` — a tracking graph as plain arrays, and a numpy-only reader for `.geff`.

The interchange format for everything in this project. `tracksdata` graphs are the
official currency, but `tracksdata` cannot be installed on Kaggle (numpy pin — see
`harness/purescore.py`), and Kaggle is where we actually run. So the pipeline produces
`Tracks`, the harness scores `Tracks`, and the submission writer serialises `Tracks`.
Conversion to and from a `tracksdata` graph is a one-line adapter used only where the
official code is available and wanted.

    t     : (N,) int      frame index per node
    zyx   : (N, 3) float  centroid in VOXEL units — the space the ground truth uses
    edges : (E, 2) int    (source_index, target_index); row order is the edge id
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Tracks:
    t: np.ndarray
    zyx: np.ndarray
    edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t).astype(np.int64).ravel()
        self.zyx = np.asarray(self.zyx, float).reshape(-1, 3)
        self.edges = np.asarray(self.edges, int).reshape(-1, 2)
        if len(self.t) != len(self.zyx):
            raise ValueError(f"{len(self.t)} times vs {len(self.zyx)} centroids")
        if len(self.edges) and (self.edges.max() >= len(self.t) or self.edges.min() < 0):
            raise ValueError("edge references a node index that does not exist")

    # -- basics --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.t)

    @property
    def n_nodes(self) -> int:
        return len(self.t)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def out_degree(self) -> np.ndarray:
        return np.bincount(self.edges[:, 0], minlength=self.n_nodes) if self.n_edges \
            else np.zeros(self.n_nodes, int)

    @property
    def in_degree(self) -> np.ndarray:
        return np.bincount(self.edges[:, 1], minlength=self.n_nodes) if self.n_edges \
            else np.zeros(self.n_nodes, int)

    @property
    def n_divisions(self) -> int:
        return int((self.out_degree > 1).sum())

    def has_forks(self) -> bool:
        """True if any node has more than one child.

        `purescore` is only exact for fork-free predictions, so this is the guard on
        which scoring path a prediction is eligible for.
        """
        return self.n_divisions > 0

    def coords_um(self, scale) -> np.ndarray:
        return self.zyx * np.asarray(scale, float)[None, :]

    # -- adapters ------------------------------------------------------------

    def to_tracksdata(self):
        """Build a `tracksdata` InMemoryGraph. Only for the official-scorer path."""
        import polars as pl
        import tracksdata as td

        g = td.graph.InMemoryGraph()
        for k in ("z", "y", "x"):
            g.add_node_attr_key(k, pl.Float64, 0.0)
        ids = [g.add_node({"t": int(self.t[i]), "z": float(self.zyx[i, 0]),
                           "y": float(self.zyx[i, 1]), "x": float(self.zyx[i, 2])})
               for i in range(self.n_nodes)]
        for s, d in self.edges:
            g.add_edge(ids[int(s)], ids[int(d)], {})
        return g

    @classmethod
    def from_tracksdata(cls, graph) -> "Tracks":
        import tracksdata as td

        na = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,
                                         td.DEFAULT_ATTR_KEYS.T, "z", "y", "x"])
        ids = na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_numpy()
        idx = {int(v): i for i, v in enumerate(ids)}
        zyx = np.stack([na["z"].to_numpy(), na["y"].to_numpy(), na["x"].to_numpy()], 1)
        ea = graph.edge_attrs(attr_keys=[])
        if len(ea):
            src = [idx[int(v)] for v in ea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy()]
            dst = [idx[int(v)] for v in ea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy()]
            edges = np.stack([src, dst], 1)
        else:
            edges = np.zeros((0, 2), int)
        return cls(na[td.DEFAULT_ATTR_KEYS.T].to_numpy(), zyx, edges)


def read_geff(path: Path | str) -> Tracks:
    """Read a `.geff` into `Tracks` using the light `geff` package only.

    Deliberately *not* `tracksdata`. `geff.read(..., backend='networkx')` pulls in
    networkx and zarr and nothing heavier, which is what makes ground truth readable
    inside a Kaggle kernel. `structure_validation=False` because we only need the
    node table and edge list, and validation is the slow part on 199 datasets.
    """
    import geff

    G, _ = geff.read(str(path), backend="networkx", structure_validation=False)
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    d = G.nodes
    t = np.array([d[n]["t"] for n in nodes], np.int64)
    zyx = np.array([[d[n].get("z", 0.0), d[n]["y"], d[n]["x"]] for n in nodes], float)
    e = list(G.edges())
    edges = np.array([[idx[u], idx[v]] for u, v in e], int) if e else np.zeros((0, 2), int)
    return Tracks(t, zyx, edges)


def read_estimated_nodes(geff_path: Path | str) -> float:
    """The scorer's ``N_total`` from GEFF metadata extras, or NaN.

    Recon §9: this varies 20.8x across datasets, so it must be read per dataset —
    a global constant is what turns the node-budget multiplier into a trap.
    """
    try:
        from geff import GeffMetadata
        meta = GeffMetadata.read(str(geff_path))
        v = _find_key(meta.extra or {}, "estimated_number_of_nodes")
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def read_scale(zarr_path: Path | str) -> tuple[float, float, float]:
    """Voxel size (z, y, x) in µm from the OME-Zarr multiscales metadata."""
    from .purescore import DEFAULT_SCALE
    try:
        import zarr
        attrs = dict(zarr.open_group(str(zarr_path), mode="r").attrs)
        tr = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if tr.get("type") == "scale":
            return tuple(float(v) for v in tr["scale"][-3:])
    except Exception:
        pass
    return DEFAULT_SCALE


def _find_key(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


__all__ = ["Tracks", "read_geff", "read_estimated_nodes", "read_scale"]
