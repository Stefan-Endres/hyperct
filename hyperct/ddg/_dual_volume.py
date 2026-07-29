"""Exact barycentric dual volumes from the explicit top-simplex container.

For the *barycentric* dual, the dual cell measure of vertex *i* obeys the
closed-form identity

    Vol_i = (1 / (dim + 1)) * sum_{T ∋ i} |T|

over the incident top-dimensional simplices *T*: the barycentric
subdivision of a d-simplex splits it into ``(d+1)!`` equal-measure flag
simplices, ``d!`` of which lie in each corner cell, so every vertex owns
exactly ``1/(d+1)`` of each incident simplex.  The identity is exact to
machine precision, tiles the domain (including boundary and corner
cells) by construction, and requires no dual-connectivity fan walks —
so there are no degenerate/exception paths.

Both routines require the explicit top-simplex cache ``HC._simplices``
(populate via :func:`hyperct.ddg.connect_and_cache_simplices` or an
active :class:`hyperct._simplicial.SimplicialComplex`).  They are NOT
valid for circumcentric duals — use the geometric dual-cell routines
(:func:`hyperct.ddg.dual_cell_area_2d`, ``v_star``) for those.
"""
from __future__ import annotations

import math

import numpy as np


def _require_simplices(HC, fname: str):
    """Return the top-simplex list or raise if the cache is absent."""
    simplices = getattr(HC, '_simplices', None)
    if simplices is None:
        raise ValueError(
            f"{fname} requires the explicit top-simplex cache "
            "HC._simplices (populate via connect_and_cache_simplices "
            "or a SimplicialComplex); it is None."
        )
    return simplices


def simplex_dual_volumes(HC, dim: int) -> dict:
    """Exact barycentric dual volumes for all vertices of ``HC``.

    Computes ``Vol_i = (1/(dim+1)) * sum_{T ∋ i} |T|`` over the cached
    top-dimensional simplices in a single vectorized pass.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with ``HC._simplices`` populated (a list of
        ``(dim+1)``-tuples of vertex objects).
    dim : int
        Spatial dimension (embedding dimension of the top simplices).

    Returns
    -------
    dict
        Mapping ``{vertex object: float}`` covering every vertex in
        ``HC.V`` (vertices not incident to any cached simplex map to
        ``0.0``).  The values sum to the total mesh volume exactly
        (partition of unity).

    Raises
    ------
    ValueError
        If ``HC._simplices`` is ``None``.
    """
    simplices = _require_simplices(HC, "simplex_dual_volumes")

    vols: dict = {v: 0.0 for v in HC.V}
    tops = [s for s in simplices if len(s) == dim + 1]
    if not tops:
        return vols

    coords = np.empty((len(tops), dim + 1, dim))
    for i, s in enumerate(tops):
        for j, vv in enumerate(s):
            coords[i, j, :] = vv.x_a[:dim]
    # |T| = |det([x_1 - x_0, ..., x_d - x_0])| / d!
    edges = coords[:, 1:, :] - coords[:, :1, :]        # (N, dim, dim)
    simplex_vols = np.abs(np.linalg.det(edges)) / math.factorial(dim)
    shares = simplex_vols / (dim + 1)

    for s, share in zip(tops, shares):
        w = float(share)
        for vv in s:
            vols[vv] = vols.get(vv, 0.0) + w
    return vols


def vertex_dual_volume(HC, v, dim: int) -> float:
    """Exact barycentric dual volume of a single vertex.

    ``Vol_i = (1/(dim+1)) * sum_{T ∋ i} |T|`` over the cached
    top-dimensional simplices incident to ``v``.  Scans
    ``HC._simplices`` once; for all-vertex computation prefer the
    batched :func:`simplex_dual_volumes`.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with ``HC._simplices`` populated.
    v : vertex object
        Primal vertex (membership tested by object identity).
    dim : int
        Spatial dimension.

    Returns
    -------
    float
        Exact barycentric dual cell measure (0.0 for a vertex not
        incident to any cached simplex).

    Raises
    ------
    ValueError
        If ``HC._simplices`` is ``None``.
    """
    simplices = _require_simplices(HC, "vertex_dual_volume")

    d_fact = math.factorial(dim)
    total = 0.0
    for s in simplices:
        if len(s) != dim + 1 or v not in s:
            continue
        pts = np.array([vv.x_a[:dim] for vv in s])
        total += abs(np.linalg.det(pts[1:] - pts[0])) / d_fact
    return total / (dim + 1)
