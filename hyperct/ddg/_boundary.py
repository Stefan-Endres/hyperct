"""Boundary detection from an explicit top-dim simplex list.

A face is on the boundary iff it appears in exactly one top-dim simplex.
This is exact for any valid simplicial complex and avoids the
flag-complex K_{dim+1} ambiguity that affects
:meth:`hyperct.Complex.boundary` on Delaunay-derived meshes with skinny
simplices.

Requires ``HC._simplices`` to be populated (e.g. via
:func:`hyperct.ddg.connect_and_cache_simplices`).  Raises ``ValueError``
if the cache is ``None``.
"""
from __future__ import annotations

from collections import defaultdict


def boundary_from_simplices(HC, dim: int) -> set:
    """Boundary vertex set computed from ``HC._simplices``.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with ``_simplices`` populated (a list of
        ``dim+1``-tuples of vertex objects).
    dim : int
        Spatial dimension (``>= 1``).  Each simplex must have ``dim+1``
        vertices.

    Returns
    -------
    set
        Set of vertex objects on the boundary.

    Raises
    ------
    ValueError
        If ``HC._simplices`` is ``None`` (no simplex cache populated) or
        if ``dim < 1``.

    Notes
    -----
    Algorithm: build ``face_count[face_id_tuple] -> int`` counting how
    many simplices contain each face (a face is a ``dim``-tuple of
    vertices, identified by ``tuple(sorted(id(v) for v in face))``).
    Faces with count == 1 are boundary faces; the boundary vertex set is
    the union of all vertices belonging to such faces.

    This is the recommended replacement for :meth:`Complex.boundary` on
    any complex that has been through a Delaunay re-triangulation.  The
    legacy ``Complex.boundary`` enumerates simplices combinatorially
    from the 1-skeleton (``v.nn``) and is unreliable on Delaunay outputs
    because the flag complex contains K_{dim+1} cliques that are not
    real top-dim simplices — see ``grok_1-skeleton-comment.pdf`` and the
    deferred-work note in ``hyperct/DEVELOPMENT.md``.
    """
    if dim < 1:
        raise ValueError(
            f"boundary_from_simplices requires dim >= 1; got {dim}"
        )

    simplices = getattr(HC, '_simplices', None)
    if simplices is None:
        raise ValueError(
            "HC._simplices is None — boundary_from_simplices requires an "
            "explicit top-dim simplex cache.  Populate it via "
            "hyperct.ddg.connect_and_cache_simplices, or fall back to "
            "Complex.boundary() (note: unreliable on Delaunay-derived "
            "complexes with skinny simplices)."
        )

    # face_count[id_tuple] = number of simplices containing this face.
    # face_verts[id_tuple] = the actual vertex tuple (any one is fine —
    # all instances refer to the same vertex objects).
    face_count: dict = defaultdict(int)
    face_verts: dict = {}

    n_face_verts = dim  # a dim-1 face of a dim-simplex has `dim` vertices

    for simplex in simplices:
        if len(simplex) != dim + 1:
            continue  # skip mismatched entries (e.g. ghost-dedup leftovers)
        for skip in range(dim + 1):
            face = tuple(simplex[i] for i in range(dim + 1) if i != skip)
            assert len(face) == n_face_verts
            key = tuple(sorted(id(v) for v in face))
            face_count[key] += 1
            if key not in face_verts:
                face_verts[key] = face

    boundary: set = set()
    for key, count in face_count.items():
        if count == 1:
            for v in face_verts[key]:
                boundary.add(v)

    return boundary
