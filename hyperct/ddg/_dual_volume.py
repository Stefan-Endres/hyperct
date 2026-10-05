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

:func:`simplex_dual_face_areas` is the companion for the dual FACES: the
oriented area vector of the barycentric dual face of every edge, summed
over the incident top simplices in one vectorised pass.

All routines require the explicit top-simplex cache ``HC._simplices``
(populate via :func:`hyperct.ddg.connect_and_cache_simplices` or an
active :class:`hyperct._simplicial.SimplicialComplex`).  They are NOT
valid for circumcentric duals — use the geometric dual-cell routines
(:func:`hyperct.ddg.dual_cell_area_2d`, ``v_star``) for those.
"""
from __future__ import annotations

import math
from itertools import combinations

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


def _facet_parity(F: np.ndarray) -> np.ndarray:
    """Parity (+1 / -1) of the permutation that sorts each row of *F*
    (``(N, 2)`` or ``(N, 3)`` integer rows)."""
    if F.shape[1] == 2:
        inv = F[:, 0] > F[:, 1]
    else:
        inv = ((F[:, 0] > F[:, 1]).astype(int) + (F[:, 0] > F[:, 2])
               + (F[:, 1] > F[:, 2]))
    return np.where(inv % 2 == 0, 1.0, -1.0)


def _orientation_signs(idx: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Combinatorial orientation ``s`` (+1 / -1 per simplex) of the complex
    with simplex vertex rows *idx* and signed volumes *det* (of the stored
    vertex order).

    Two simplices that share a facet induce opposite orientations on it:
    ``s_1 o_1 = -s_2 o_2`` with ``o`` the orientation the stored order of
    the simplex induces on the sorted facet.  The relation is propagated
    over the facet adjacency (a two-layer graph whose connected components
    are the two orientations of every orientable component), and each
    component is signed so that its summed signed volume ``sum s det`` is
    positive.  On a valid embedding ``s = sign(det)`` for every simplex
    of positive measure; a FLAT simplex (``det == 0``, which qhull's
    triangulated output puts between its neighbours on cospherical point
    sets) gets the orientation its neighbours induce, which no geometric
    rule can give it.  A simplex whose two layers meet (non-orientable or
    non-manifold adjacency) falls back to ``sign(det)``, flat ones to +1.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n, k = idx.shape
    d = k - 1
    # facet opposite local vertex a: the other d vertices in stored order
    rows = []
    for a in range(k):
        F = np.delete(idx, a, axis=1)
        o = _facet_parity(F) * (-1.0) ** a
        rows.append((np.sort(F, axis=1), o))
    facets = np.concatenate([r[0] for r in rows])          # (n k, d)
    orient = np.concatenate([r[1] for r in rows])          # (n k,)
    owner = np.tile(np.arange(n), k)                       # (n k,)
    # pairs: the two owners of every facet shared by exactly two simplices
    order = np.lexsort(facets.T[::-1])
    sorted_f = facets[order]
    same_as_next = np.all(sorted_f[1:] == sorted_f[:-1], axis=1)
    pos = np.nonzero(same_as_next)[0]
    # keep only facets with exactly two owners (a run of two equal rows)
    two = np.ones(len(pos), dtype=bool)
    if len(pos) > 1:
        two[1:] &= pos[1:] != pos[:-1] + 1
        two[:-1] &= pos[1:] != pos[:-1] + 1
    pos = pos[two]
    t1, t2 = owner[order[pos]], owner[order[pos + 1]]
    r = -orient[order[pos]] * orient[order[pos + 1]]      # s_2 = r s_1
    # two-layer graph: node t is "s_t = +1", node t + n is "s_t = -1"
    same = r > 0
    src = np.concatenate([t1[same], t1[same] + n, t1[~same], t1[~same] + n])
    dst = np.concatenate([t2[same], t2[same] + n, t2[~same] + n, t2[~same]])
    g = coo_matrix((np.ones(len(src)), (src, dst)), shape=(2 * n, 2 * n))
    _, label = connected_components(g, directed=False)
    lp, lm = label[:n], label[n:]
    consistent = lp != lm
    # one orientation per orientable component: the layer that holds the
    # "+" node of its first simplex
    comp = np.minimum(lp, lm) * (2 * n) + np.maximum(lp, lm)
    _, comp_first, comp_idx = np.unique(comp, return_index=True,
                                        return_inverse=True)
    chosen = lp[comp_first][comp_idx]
    s = np.where(lp == chosen, 1.0, -1.0)
    # global sign per component: summed signed volume positive
    signed = np.bincount(comp_idx, weights=s * det, minlength=len(comp_first))
    flip = np.where(signed < 0, -1.0, 1.0)[comp_idx]
    s = s * flip
    geo = np.where(det < 0, -1.0, 1.0)
    return np.where(consistent, s, geo)


def simplex_dual_face_areas(HC, dim: int) -> dict:
    """Oriented barycentric dual face area vectors of every edge, from the
    top-simplex cache, in one vectorised pass.

    Inside a simplex ``T`` the dual face between the cells of ``i`` and
    ``j`` is the union of the barycentric-subdivision pieces at the edge
    (in 3D the two triangles (edge midpoint, face barycentre, cell
    barycentre) of the two faces at the edge; in 2D the segment from the
    edge midpoint to the triangle barycentre).  Its area vector, oriented
    from ``i`` to ``j``, is::

        A_ij^T = |T| (grad(phi_j) - grad(phi_i)) / (dim + 1)

    with ``phi`` the barycentric coordinates of ``T`` (so that
    ``A_ij^T . (x_j - x_i) = 2 |T| / (dim + 1) > 0``), and the dual face
    of the edge is the sum over the simplices that contain it.  ``|T|
    grad(phi)`` is built from the cofactors of the edge matrix (no
    division), so it is finite on a flat simplex; the sign of every
    simplex is its combinatorial orientation (:func:`_orientation_signs`),
    which equals ``sign(det)`` on a valid embedding and gives a flat
    simplex the orientation of its neighbours.

    Properties (tested):

    - ``A_ij = -A_ji`` exactly;
    - the cell of a vertex whose simplices surround it closes,
      ``sum_j A_ij = 0``, flat simplices included;
    - the half cell of a hull vertex closes with its hull facets:
      ``sum_j A_ij + sum_f |f| n_f / dim = 0`` over the hull facets ``f``
      at the vertex (outward ``n_f``);
    - linear precision: ``1/2 sum_j (x_j - x_i) (x) A_ij = Vol_i I`` at
      every vertex whose cell closes, with ``Vol_i`` of
      :func:`simplex_dual_volumes`;
    - equal to the per-edge polygon of tet barycentres interleaved with
      face barycentres (the DEC ``p_ij`` face) to round-off.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with ``HC._simplices`` populated.
    dim : int
        Spatial dimension, 2 or 3.

    Returns
    -------
    dict
        ``{id(v): {id(w): np.ndarray(dim,)}}`` for every directed edge of
        every cached top simplex (hull vertices included), the layout of
        ``batch_e_star(orient=True)``.

    Raises
    ------
    ValueError
        If ``HC._simplices`` is ``None``.
    NotImplementedError
        For ``dim`` not in (2, 3).
    """
    simplices = _require_simplices(HC, "simplex_dual_face_areas")
    if dim not in (2, 3):
        raise NotImplementedError(
            f"simplex_dual_face_areas supports dim 2 and 3, got {dim}")
    verts = list(HC.V)
    index = {id(v): k for k, v in enumerate(verts)}
    tops = [s for s in simplices if len(s) == dim + 1]
    if not tops:
        return {}
    idx = np.array([[index[id(w)] for w in s] for s in tops], dtype=np.intp)
    X = np.array([v.x_a[:dim] for v in verts], dtype=float)
    P = X[idx]                                     # (N, dim + 1, dim)
    E = P[:, 1:, :] - P[:, :1, :]                  # rows e_k = x_k - x_0
    # cofactors c_k (k = 1..dim): c_k . e_l = det delta_kl
    C = np.empty_like(E)
    if dim == 3:
        C[:, 0] = np.cross(E[:, 1], E[:, 2])
        C[:, 1] = np.cross(E[:, 2], E[:, 0])
        C[:, 2] = np.cross(E[:, 0], E[:, 1])
    else:
        C[:, 0, 0] = E[:, 1, 1]
        C[:, 0, 1] = -E[:, 1, 0]
        C[:, 1, 0] = -E[:, 0, 1]
        C[:, 1, 1] = E[:, 0, 0]
    det = np.einsum('tc,tc->t', E[:, 0], C[:, 0])
    s = _orientation_signs(idx, det)
    # B[t, k] = s |T| grad(phi_k) (k = 0..dim), with grad(phi_0) = -sum_k
    B = np.empty_like(P)
    B[:, 1:] = C * (s / math.factorial(dim))[:, None, None]
    B[:, 0] = -B[:, 1:].sum(axis=1)
    # pieces of every directed edge, accumulated over the simplices
    I, J, A = [], [], []
    for a, b in combinations(range(dim + 1), 2):
        piece = (B[:, b] - B[:, a]) / (dim + 1)
        I += [idx[:, a], idx[:, b]]
        J += [idx[:, b], idx[:, a]]
        A += [piece, -piece]
    I = np.concatenate(I)
    J = np.concatenate(J)
    A = np.concatenate(A)
    n_v = len(verts)
    key, inv = np.unique(I * n_v + J, return_inverse=True)
    acc = np.zeros((len(key), dim))
    np.add.at(acc, inv, A)
    ids = [id(v) for v in verts]
    out: dict = {}
    for e, k in enumerate(key):
        i, j = divmod(int(k), n_v)
        out.setdefault(ids[i], {})[ids[j]] = acc[e]
    return out
