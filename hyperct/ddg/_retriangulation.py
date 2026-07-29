"""Delaunay-based retriangulation helpers.

Centralises the pattern of running a scipy Delaunay triangulation on a
vertex cloud, connecting primal edges, and caching the explicit top-dim
simplex list on the Complex for use by simplex-aware dual / boundary
routines.

Why a helper
------------
The pattern

    tri = Delaunay(coords)
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                verts[simplex[i]].connect(verts[simplex[j]])
    HC._simplices = [
        tuple(verts[s[i]] for i in range(dim + 1))
        for s in tri.simplices
    ]

appears in several places (the ddgclib dynamic retopology loop,
multiphase domain builders, periodic-BC ghost resolution, benchmark
setups).  Forgetting the simplex-cache step silently drops the
simplex-aware dual + boundary code paths and reintroduces the
flag-complex (1-skeleton) ghost-clique bugs near domain boundaries.

This helper encapsulates the invariant: after calling it, ``HC`` has

- correct primal edges (``v.nn``), and
- a correct ``HC._simplices`` cache (a list of ``dim+1``-tuples of
  vertex *objects*, so all field data on each vertex stays attached).

Vertex-correspondence invariant
-------------------------------
The ``verts`` list passed in must be ``list(HC.V)`` taken AFTER any
disconnect/merge/ghost-resolution but BEFORE the Delaunay call.
Delaunay returns integer indices into ``coords``; we immediately
translate to the vertex *objects* ``verts[i]`` so even if ``HC.V``'s
iteration order changes later, the cache stays valid.

The cache MUST be invalidated whenever a vertex is added/removed/moved
without going through this helper — call :func:`invalidate_simplex_cache`
in those code paths.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def connect_and_cache_simplices(
    HC,
    verts: list,
    dim: int,
    simplices=None,
    coords=None,
    qhull_options: str | None = None,
) -> None:
    """Triangulate (if needed), connect primal edges, and cache simplices.

    Replaces the triplet (Delaunay → connect → cache HC._simplices)
    duplicated across dynamic retopology, multiphase domain builders,
    periodic BCs and benchmarks.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.  ``HC._simplices`` is set for both 2D and 3D
        (a list of ``dim+1``-tuples of vertex objects).
    verts : list of vertex
        Ordered list of primal vertices matching the
        ``coords``/``simplices`` integer indices.
    dim : int
        Spatial dimension (2 or 3).
    simplices : array-like of shape (N, dim+1), optional
        Pre-computed simplex index list.  When provided, ``coords`` is
        ignored and no Delaunay call is made — useful for callers that
        already ran Delaunay (e.g. periodic-ghost resolution).
    coords : array-like, optional
        Vertex coordinate array of shape ``(len(verts), dim)``.  Required
        when ``simplices`` is None.
    qhull_options : str, optional
        Passed through to ``scipy.spatial.Delaunay`` when ``simplices``
        is computed here.  ``None`` uses scipy defaults; falls back to
        ``"Qbb Qt Qz"`` on cospherical/cocircular failure.

    Raises
    ------
    ValueError
        If neither ``simplices`` nor ``coords`` is provided, or if
        ``dim`` is not 2 or 3.
    """
    if dim not in (2, 3):
        raise ValueError(
            f"connect_and_cache_simplices supports dim ∈ {{2, 3}}; got {dim}"
        )

    if simplices is None:
        if coords is None:
            raise ValueError("Must provide either simplices or coords")
        coords = np.asarray(coords)
        # NOTE(laneA-canonical-order, 2026-07-29): canonicalize the
        # qhull input order in 3D.  qhull tie-breaking on degenerate
        # (cospherical) point sets depends on input ORDER, so the same
        # point SET triangulated from different vertex orderings (mesh
        # builder vs list(HC.V) after re-keying) yields different
        # triangulations — the diagnosed cause of the ddgclib 3D
        # droplet retopology settle-step artifact (lane3 log).  Sorting
        # lexicographically by coordinates makes the triangulation a
        # function of the point set only; retopology of a static cloud
        # becomes idempotent (ddgclib 3D static-droplet floor plateau
        # 7.616854e-5 -> 7.274172e-5, settle step eliminated).  Gated
        # to dim == 3: the pinned 2D baselines must stay bit-identical.
        # Regression: tests/test_retriangulation_order.py.
        order = None
        if dim == 3:
            order = np.lexsort(coords.T[::-1])
            d_coords = coords[order]
        else:
            d_coords = coords
        try:
            if qhull_options is None:
                tri = Delaunay(d_coords)
            else:
                tri = Delaunay(d_coords, qhull_options=qhull_options)
        except Exception:
            # Cospherical / cocircular fallback
            tri = Delaunay(d_coords, qhull_options="Qbb Qt Qz")
        simplices = tri.simplices if order is None else order[tri.simplices]

    # Connect edges from the simplex list
    for simplex in simplices:
        n = len(simplex)
        for i in range(n):
            for j in range(i + 1, n):
                verts[simplex[i]].connect(verts[simplex[j]])

    # Cache explicit simplex list (top-dim only) for both 2D and 3D.
    # The ``len(s) == dim + 1`` filter lets ghost-dedup callers pass mixed
    # lists (e.g. periodic BCs) without spuriously caching boundary faces
    # as if they were top-dim simplices.
    HC._simplices = [
        tuple(verts[s[i]] for i in range(dim + 1))
        for s in simplices
        if len(s) == dim + 1
    ]


def invalidate_simplex_cache(HC) -> None:
    """Clear ``HC._simplices`` and any caches derived from it.

    Call after any topology change not routed through
    :func:`connect_and_cache_simplices` — for example after
    ``HC.V.merge_all``, after adaptive remesh batches
    (`hyperct.remesh.edge_split_2d` etc.), or after boundary-condition
    vertex injection/deletion.

    Derived caches cleared:
    - ``HC._simplices`` (top-dim simplex list).
    - ``HC._edge_to_apex`` (per-edge apex map; built lazily by
      :func:`get_edge_apex_map`).
    - When an active :class:`SimplicialComplex` (``HC.SC``) is present, it is
      marked dirty so it is regenerated on the next read.
    """
    HC._simplices = None
    if hasattr(HC, '_edge_to_apex'):
        HC._edge_to_apex = None
    sc = getattr(HC, '_SC', None)
    if sc is not None:
        sc.mark_dirty()


def rebuild_simplex_cache_2d(HC) -> int:
    """Rebuild ``HC._simplices`` from the current 1-skeleton (2D only).

    For use after local-operation batches (``hyperct.remesh`` edge
    split / collapse / flip) that maintain a conforming planar
    triangulation but do not track the explicit simplex list.  Unlike
    :func:`invalidate_simplex_cache`, this keeps the simplex-aware
    dual / boundary / exact-dual-volume code paths active — dropping
    the cache silently downgrades ``compute_vd``,
    ``boundary_from_simplices`` and the exact 2D dual volumes to the
    1-skeleton fallbacks, which measurably destabilises dynamic runs
    (adaptive-remesh KE tail, ddgclib lane4-remesh-upstream
    2026-07-02).

    Triangles are enumerated as K_3 cliques of the connectivity graph.
    Cliques that are not faces — a triangle of edges whose interior is
    subdivided by a vertex connected to all three corners — are
    filtered with a strict point-in-triangle test against the common
    neighbours (the classic flag-complex K_3 ambiguity).

    Derived caches (``HC._edge_to_apex``, an active ``HC.SC``) are
    reset exactly as in :func:`invalidate_simplex_cache`.

    Returns
    -------
    int
        The number of cached triangles.
    """
    tris = []
    for v in HC.V:
        vid = id(v)
        for v2 in v.nn:
            if id(v2) <= vid:
                continue
            for v3 in v.nn:
                if id(v3) <= id(v2):
                    continue
                if v3 not in v2.nn:
                    continue
                # Ghost-K_3 filter: skip cliques subdivided by a common
                # neighbour lying strictly inside the triangle.
                p0 = np.asarray(v.x_a[:2], dtype=float)
                p1 = np.asarray(v2.x_a[:2], dtype=float)
                p2 = np.asarray(v3.x_a[:2], dtype=float)
                d1 = p1 - p0
                d2 = p2 - p0
                det = d1[0] * d2[1] - d1[1] * d2[0]
                is_ghost = False
                if abs(det) > 0.0:
                    for w in (v.nn & v2.nn & v3.nn):
                        if w is v or w is v2 or w is v3:
                            continue
                        q = np.asarray(w.x_a[:2], dtype=float) - p0
                        # Barycentric coordinates of w w.r.t. (v, v2, v3)
                        s = (q[0] * d2[1] - q[1] * d2[0]) / det
                        t = (d1[0] * q[1] - d1[1] * q[0]) / det
                        if s > 0.0 and t > 0.0 and s + t < 1.0:
                            is_ghost = True
                            break
                if not is_ghost:
                    tris.append((v, v2, v3))

    HC._simplices = tris if tris else None
    if hasattr(HC, '_edge_to_apex'):
        HC._edge_to_apex = None
    sc = getattr(HC, '_SC', None)
    if sc is not None:
        sc.mark_dirty()
    return len(tris)


def get_edge_apex_map(HC) -> dict | None:
    """Return (lazily build) the per-edge apex map ``HC._edge_to_apex``.

    For each unordered edge ``(vi, vj)`` of every cached top-dim simplex
    in ``HC._simplices``, collect the *apex* vertex objects — i.e. the
    other vertices of every simplex containing that edge.

    Layout::

        HC._edge_to_apex : dict[frozenset[int, int], list[vertex]]

    where the key is ``frozenset((id(vi), id(vj)))`` and the value is a
    flat list of apex vertex objects (one per containing simplex in 2D
    triangle meshes; ``dim - 1`` per containing simplex in 3D tet meshes,
    so a list of vertex pairs flattened).

    Returns
    -------
    dict | None
        The cached apex map, or ``None`` if ``HC._simplices`` is ``None``
        (no simplex cache populated — caller must fall back to legacy
        ``vi.nn ∩ vj.nn``).

    Notes
    -----
    This is the simplex-aware replacement for the legacy
    ``vi.nn.intersection(vj.nn)`` apex-enumeration pattern in
    curvature / bubble / interface routines.  On Delaunay-derived
    meshes with skinny simplices the legacy pattern can return spurious
    K_{dim+1} clique vertices that are not real apices; this map walks
    the explicit top-dim simplex list and is exact.

    Cached on the Complex.  Invalidated by :func:`invalidate_simplex_cache`.
    """
    cached = getattr(HC, '_edge_to_apex', None)
    if cached is not None:
        return cached

    simplices = getattr(HC, '_simplices', None)
    if simplices is None:
        return None

    apex_map: dict = {}
    for s in simplices:
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                vi = s[i]
                vj = s[j]
                key = frozenset((id(vi), id(vj)))
                apex_list = apex_map.setdefault(key, [])
                for k in range(n):
                    if k == i or k == j:
                        continue
                    apex_list.append(s[k])

    HC._edge_to_apex = apex_map
    return apex_map


def apex_vertices(HC, vi, vj) -> list:
    """Return the apex vertices opposite the primal edge ``(vi, vj)``.

    The *apices* are the remaining vertices of every top-dim simplex that
    contains the edge — i.e. for a 2D triangle mesh, one apex per incident
    triangle (two for an interior edge, one for a boundary edge).

    When an explicit simplex cache is available (``HC._simplices`` /
    ``HC.SC``), this uses the exact :func:`get_edge_apex_map` and is correct
    even on Delaunay-derived meshes with skinny simplices.  Otherwise it
    falls back to the legacy ``vi.nn ∩ vj.nn`` intersection, which can
    return spurious ``K_{dim+1}`` clique vertices that are not real apices.

    Passing ``HC=None`` forces the legacy fallback (identical to the
    historical behaviour of the curvature routines).
    """
    if HC is not None:
        amap = get_edge_apex_map(HC)
        if amap is not None:
            return list(amap.get(frozenset((id(vi), id(vj))), []))
    return list(vi.nn.intersection(vj.nn))
