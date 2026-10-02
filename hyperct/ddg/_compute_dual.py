"""
Unified dual mesh computation for hyperct simplicial complexes.

Computes the Voronoi-like dual mesh of a primal simplicial complex
using either barycentric (centroid) or circumcentric dual vertex
placement.
"""
from __future__ import annotations

from typing import Literal, Union

import numpy as np

from hyperct._vertex import VertexCacheIndex
from ._strategies import barycenter, circumcenter, DualStrategy
from ._geometry import _merge_local_duals_vector


def _batch_strategy(
    strategy: DualStrategy,
    simplex_array: np.ndarray,
) -> np.ndarray:
    """Apply a dual vertex strategy to multiple simplices at once.

    For barycentric placement, this is fully vectorized. For circumcentric
    or custom strategies, falls back to per-simplex computation.

    :param strategy: The dual vertex placement strategy.
    :param simplex_array: Array of shape (N, k+1, dim) containing N simplices.
    :return: Array of shape (N, dim) with dual vertex positions.
    """
    if strategy is barycenter:
        return np.mean(simplex_array, axis=1)
    N = simplex_array.shape[0]
    dim = simplex_array.shape[2]
    result = np.empty((N, dim))
    for i in range(N):
        result[i] = strategy(simplex_array[i])
    return result


def compute_vd(
    HC,
    method: Union[
        Literal["barycentric", "circumcentric"], DualStrategy
    ] = "barycentric",
    cdist: float = 1e-10,
    global_merge: bool = True,
    backend=None,
):
    """Compute the dual vertices of a primal complex HC.

    For each simplex in the primal mesh, a dual vertex is placed at
    either the barycenter (centroid) or circumcenter, depending on the
    chosen method. Dual vertices are connected to form the dual mesh.

    Boundary vertices must be marked with ``v.boundary = True`` before
    calling this function (e.g. via ``HC.boundary()``).

    :param HC: A hyperct Complex with triangulated vertex cache HC.V.
    :param method: ``"barycentric"`` (default), ``"circumcentric"``, or
        a custom callable taking (n_verts, dim) array -> (dim,) array.
    :param cdist: Tolerance for merging nearby dual vertices.
    :param global_merge: If True (default), perform a global merge pass
        at the end using spatial hashing to catch any remaining duplicates.
    :param backend: Optional ``BatchBackend`` instance (from
        ``hyperct._backend.get_backend``).  When provided, 2D and 3D
        dual computation uses a simplex-centric batch path that collects
        all simplices first and calls ``backend.batch_dual_positions``
        in a single vectorized/parallel operation.  Pass ``None``
        (default) to use the original per-edge sequential path.
    :return: The same Complex, with ``HC.Vd`` populated and each primal
        vertex ``v`` having ``v.vd`` (set of dual vertices).
    """
    # Resolve strategy
    if isinstance(method, str):
        strategy_map = {
            "barycentric": barycenter,
            "circumcentric": circumcenter,
        }
        if method not in strategy_map:
            raise ValueError(
                f"Unknown method {method!r}. "
                "Use 'barycentric' or 'circumcentric'."
            )
        strategy = strategy_map[method]
    else:
        strategy = method

    # Record the construction method so downstream geometry code can
    # branch on it (e.g. the exact simplex-based dual volumes in
    # hyperct.ddg._dual_volume apply only to barycentric duals).
    HC._vd_method = method if isinstance(method, str) else "custom"

    # Construct dual cache (VertexCacheIndex: pure geometry, no field overhead)
    HC.Vd = VertexCacheIndex()

    # Initialize dual neighbour sets on primal vertices
    for v in HC.V:
        v.vd = set()

    if HC.dim == 1:
        _compute_vd_1d(HC, cdist)
    elif HC.dim == 2:
        if backend is not None:
            _compute_vd_2d_batch(HC, strategy, cdist, backend)
        else:
            _compute_vd_2d(HC, strategy, cdist)
    elif HC.dim == 3:
        if backend is not None:
            _compute_vd_3d_batch(HC, strategy, cdist, backend)
        else:
            _compute_vd_3d(HC, strategy, cdist)
    else:
        if getattr(HC, '_simplices', None):
            _compute_vd_nd_simplex_aware(HC, strategy, cdist)
        else:
            _compute_vd_nd(HC, strategy, cdist)

    # Optional global merge pass using spatial hashing
    # This catches any remaining floating-point duplicates that the
    # local merge might have missed, using O(n) expected time
    if global_merge and len(HC.Vd) > 0:
        HC.Vd.merge_all(cdist)

    return HC


def _compute_vd_1d(HC, cdist: float) -> None:
    """1D dual: midpoint of each edge (identical for all methods).

    In 1D, the dual of an edge is simply the midpoint. This is the same
    regardless of whether barycentric or circumcentric placement is used.
    """
    for v1 in HC.V:
        v1_d_nn = list(v1.vd)
        for v2 in v1.nn:
            cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
            cd = _merge_local_duals_vector([cd], v1_d_nn, cdist=cdist)[0]
            vd = HC.Vd[tuple(cd)]
            v1.vd.add(vd)
            v2.vd.add(vd)
            # Update the local dual neighbourhood cache
            v1_d_nn = list(v1.vd)


def _compute_vd_2d(HC, strategy: DualStrategy, cdist: float) -> None:
    """2D dual computation parameterized by strategy.

    For each primal edge (v1, v2), finds the two triangles sharing it
    and places a dual vertex at ``strategy(triangle_vertices)`` for each.
    The two dual vertices are then connected.

    When ``HC._simplices`` is populated (list of 3-vertex tuples from a
    Delaunay triangulation), uses the simplex-aware path which correctly
    handles boundary edges and avoids the flag-complex K_3 ambiguity.
    Otherwise falls back to the legacy nn-intersection path.
    """
    if getattr(HC, '_simplices', None):
        _compute_vd_2d_simplex_aware(HC, strategy, cdist)
        return

    dim = HC.dim
    # Pre-allocate vertex array — reused across iterations to avoid
    # repeated np.zeros allocation (measurable for large meshes)
    verts = np.empty((3, dim))

    for v1 in HC.V:
        for v2 in v1.nn:
            # Update the local dual neighbourhood each iteration
            v1_d_nn = list(v1.vd)

            # Boundary edge: place dual at midpoint of boundary edge
            try:
                if v1.boundary and v2.boundary:
                    cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                    cd = _merge_local_duals_vector(
                        [cd], v1_d_nn, cdist=cdist
                    )[0]
                    vd = HC.Vd[tuple(cd)]
                    v1.vd.add(vd)
                    v2.vd.add(vd)
                    # Connect to dual of the single triangle on boundary edge
                    v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                    v3 = list(v1nn_u_v2nn)[0]
                    verts[0] = v1.x_a
                    verts[1] = v2.x_a
                    verts[2] = v3.x_a
                    cd1 = strategy(verts)
                    vd1 = HC.Vd[tuple(cd1)]
                    # Connect boundary dual edge
                    vd.connect(vd1)
                    continue
            except AttributeError:
                pass

            # Interior edge: find the two triangles sharing this edge
            v1nn_u_v2nn = v1.nn.intersection(v2.nn)
            v3_1 = list(v1nn_u_v2nn)[0]
            v3_2 = list(v1nn_u_v2nn)[1]
            if (v3_1 is v1) or (v3_2 is v1):
                continue

            # First triangle
            verts[0] = v1.x_a
            verts[1] = v2.x_a
            verts[2] = v3_1.x_a
            cd1 = strategy(verts)

            # Second triangle
            verts[2] = v3_2.x_a
            cd2 = strategy(verts)

            # Merge nearby duals to avoid floating-point duplicates
            (cd1, cd2) = _merge_local_duals_vector(
                [cd1, cd2], v1_d_nn, cdist=cdist
            )

            vd1 = HC.Vd[tuple(cd1)]
            vd2 = HC.Vd[tuple(cd2)]
            # Connect the two dual vertices
            vd1.connect(vd2)

            # Associate duals with primal vertices of first triangle
            for v in [v1, v2, v3_1]:
                v.vd.add(vd1)

            # Associate duals with primal vertices of second triangle
            for v in [v1, v2, v3_2]:
                v.vd.add(vd2)


def _compute_vd_2d_simplex_aware(HC, strategy: DualStrategy, cdist: float) -> None:
    """Simplex-aware 2D dual computation.

    Uses the explicit list of triangles ``HC._simplices`` (each element is
    a 3-tuple of primal vertex objects) to enumerate primal edges and find
    triangle adjacencies directly.  Avoids the ``v1.nn ∩ v2.nn`` heuristic
    which is incorrect when slivers near boundaries produce high mutual
    edge connectivity (flag-complex K_3 cliques that aren't real triangles).

    Algorithm:
      1. For each triangle, compute its dual vertex and associate it with
         all 3 primal vertices.
      2. Build edge -> [triangles containing it] map.
      3. For each edge, connect the dual vertices of adjacent triangles
         (2 for interior, 1 for boundary — boundary handled specially).
      4. For boundary edges, add a midpoint dual and connect.
    """
    from collections import defaultdict

    simplices = HC._simplices
    if not simplices:
        return

    dim = HC.dim
    verts_buf = np.empty((3, dim))

    # --- Step 1: compute triangle dual for each simplex ---
    simplex_to_vd = {}  # index -> dual vertex object
    for si, simplex in enumerate(simplices):
        v1, v2, v3 = simplex
        verts_buf[0] = v1.x_a
        verts_buf[1] = v2.x_a
        verts_buf[2] = v3.x_a
        cd = strategy(verts_buf)
        v1_d_nn = list(v1.vd)
        cd = _merge_local_duals_vector([cd], v1_d_nn, cdist=cdist)[0]
        vd = HC.Vd[tuple(cd)]
        simplex_to_vd[si] = vd
        for v in (v1, v2, v3):
            v.vd.add(vd)

    # --- Step 2: build edge -> [simplex_index, ...] map ---
    # (the sorted ids are only the KEY of an edge)
    edge_to_simps = defaultdict(list)
    for si, simplex in enumerate(simplices):
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted((id(simplex[i]), id(simplex[j]))))
                edge_to_simps[edge].append(si)

    # --- Step 3: connect duals across edges ---
    for edge, simp_ids in edge_to_simps.items():
        if len(simp_ids) == 2:
            # Interior edge: connect the two triangle duals
            vd_a = simplex_to_vd[simp_ids[0]]
            vd_b = simplex_to_vd[simp_ids[1]]
            vd_a.connect(vd_b)
        elif len(simp_ids) == 1:
            # Boundary edge: midpoint dual and connect to the single triangle
            # (its two vertices in the order of the triangle that owns
            # it, not in ``id()`` order: memory addresses)
            va, vb = (v for v in simplices[simp_ids[0]] if id(v) in edge)
            cd_mid = 0.5 * (va.x_a + vb.x_a)
            cd_mid = _merge_local_duals_vector(
                [cd_mid], list(va.vd), cdist=cdist
            )[0]
            vd_mid = HC.Vd[tuple(cd_mid)]
            va.vd.add(vd_mid)
            vb.vd.add(vd_mid)
            vd_mid.connect(simplex_to_vd[simp_ids[0]])


def _compute_vd_3d(HC, strategy: DualStrategy, cdist: float) -> None:
    """3D dual computation parameterized by strategy.

    For each primal face (v1, v2, v3), finds the two tetrahedra sharing
    it and places a dual vertex at ``strategy(tetrahedron_vertices)`` for
    each. The two dual vertices are then connected.

    When ``HC._simplices`` is populated (list of 4-vertex tuples from the
    Delaunay triangulation), uses the simplex-aware path which correctly
    handles boundary sliver tets.  Otherwise falls back to the legacy
    nn-intersection path.
    """
    if getattr(HC, '_simplices', None):
        _compute_vd_3d_simplex_aware(HC, strategy, cdist)
        return

    dim = HC.dim
    # Pre-allocate — reused across iterations
    verts = np.empty((dim + 1, dim))

    for v1 in HC.V:
        for v2 in v1.nn:
            # Find all vertices connected to both v1 and v2
            v1nn_u_v2nn = v1.nn.intersection(v2.nn)

            for v3 in v1nn_u_v2nn:
                # Update local dual neighbourhood
                v1_d_nn = list(v1.vd)

                if v3 is v1:
                    continue

                # Find vertices connected to v1, v2, and v3 (tetrahedra)
                v1nn_u_v2nn_u_v3nn = v1nn_u_v2nn.intersection(v3.nn)
                v4_1 = list(v1nn_u_v2nn_u_v3nn)[0]

                # Compute dual of the first tetrahedron
                verts[0] = v1.x_a
                verts[1] = v2.x_a
                verts[2] = v3.x_a
                verts[3] = v4_1.x_a
                cd1 = strategy(verts)

                # Boundary face: triangle (v1, v2, v3) is on the boundary
                if (
                    _has_boundary(v1)
                    and _has_boundary(v2)
                    and _has_boundary(v3)
                ):
                    # Boundary face dual: use strategy on boundary triangle
                    verts_b = verts[:3]
                    cd2 = strategy(verts_b)

                    # Boundary edge midpoint dual
                    cd12 = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                    cd12 = _merge_local_duals_vector(
                        [cd12], v1_d_nn, cdist=cdist
                    )[0]
                    vd12 = HC.Vd[tuple(cd12)]
                    v1.vd.add(vd12)
                    v2.vd.add(vd12)

                else:
                    # Interior face: find second tetrahedron
                    v4_2 = list(v1nn_u_v2nn_u_v3nn)[1]
                    verts[3] = v4_2.x_a
                    cd2 = strategy(verts)

                # Merge nearby duals
                (cd1, cd2) = _merge_local_duals_vector(
                    [cd1, cd2], v1_d_nn, cdist=cdist
                )

                # Create and connect dual vertices
                vd1 = HC.Vd[tuple(cd1)]
                vd2 = HC.Vd[tuple(cd2)]
                vd1.connect(vd2)

                # Associate duals with primal vertices of first tetrahedron
                for v in [v1, v2, v3, v4_1]:
                    v.vd.add(vd1)

                # Associate duals with primal vertices of second tetrahedron
                if (
                    _has_boundary(v1)
                    and _has_boundary(v2)
                    and _has_boundary(v3)
                ):
                    for v in [v1, v2, v3]:
                        v.vd.add(vd2)
                else:
                    for v in [v1, v2, v3, v4_2]:
                        v.vd.add(vd2)


def _compute_vd_3d_simplex_aware(HC, strategy: DualStrategy, cdist: float) -> None:
    """Simplex-aware 3D dual computation.

    Uses the explicit list of simplices ``HC._simplices`` (each element is
    a 4-tuple of primal vertex objects) to enumerate primal faces and find
    tet adjacencies directly.  Avoids the ``v1.nn ∩ v2.nn ∩ v3.nn``
    heuristic which is incorrect when slivers near domain boundaries
    produce high mutual edge connectivity (ghost tets).

    Algorithm:
      1. For each simplex, compute its dual vertex and associate it
         with all 4 primal vertices.
      2. Build face -> [simplices containing it] map.
      3. For each face, connect the dual vertices of adjacent simplices
         (2 for interior, 1 for boundary — boundary handled specially).
      4. For boundary faces, compute face barycenter dual and edge
         midpoint duals (same as legacy path).
    """
    from collections import defaultdict

    simplices = HC._simplices
    if not simplices:
        return

    dim = HC.dim
    verts_buf = np.empty((dim + 1, dim))

    # --- Step 1: compute tet dual vertex for each simplex ---
    simplex_to_vd = {}  # index -> dual vertex object
    for si, simplex in enumerate(simplices):
        v1, v2, v3, v4 = simplex
        verts_buf[0] = v1.x_a
        verts_buf[1] = v2.x_a
        verts_buf[2] = v3.x_a
        verts_buf[3] = v4.x_a
        cd = strategy(verts_buf)
        # Merge against v1's existing duals if any (rarely triggers on first pass)
        v1_d_nn = list(v1.vd)
        cd = _merge_local_duals_vector([cd], v1_d_nn, cdist=cdist)[0]
        vd = HC.Vd[tuple(cd)]
        simplex_to_vd[si] = vd
        # Associate with all 4 primal vertices
        for v in (v1, v2, v3, v4):
            v.vd.add(vd)

    # --- Step 2: build face -> [simplex_index, ...] map ---
    face_to_simps = defaultdict(list)
    for si, simplex in enumerate(simplices):
        for i in range(4):
            face = tuple(sorted(
                id(simplex[j]) for j in range(4) if j != i
            ))
            face_to_simps[face].append(si)

    # --- Step 3: connect duals across faces ---
    # (the sorted ids are only the KEY of a face)
    for face, simp_ids in face_to_simps.items():
        if len(simp_ids) == 2:
            # Interior face: connect the two tet duals
            vd_a = simplex_to_vd[simp_ids[0]]
            vd_b = simplex_to_vd[simp_ids[1]]
            vd_a.connect(vd_b)
        elif len(simp_ids) == 1:
            # Boundary face: create face barycenter dual and connect
            # the single tet dual to it.  Also create edge midpoint duals.
            vd_tet = simplex_to_vd[simp_ids[0]]

            # The face's vertices in the order of the simplex that owns
            # it.  The barycentre below is a sum over them; in ``id()``
            # order (memory addresses, until 2026-10-02) its last bit, and
            # with it the hash of the dual vertex and the iteration order
            # of every set that holds it, differed between processes.
            v1, v2, v3 = (v for v in simplices[simp_ids[0]] if id(v) in face)

            # Face barycenter dual
            verts_buf[0] = v1.x_a
            verts_buf[1] = v2.x_a
            verts_buf[2] = v3.x_a
            cd_face = strategy(verts_buf[:3])
            v1_d_nn = list(v1.vd)
            cd_face = _merge_local_duals_vector(
                [cd_face], v1_d_nn, cdist=cdist
            )[0]
            vd_face = HC.Vd[tuple(cd_face)]
            for v in (v1, v2, v3):
                v.vd.add(vd_face)
            vd_tet.connect(vd_face)

            # Edge midpoint duals for each of the 3 boundary face edges
            for va, vb in ((v1, v2), (v1, v3), (v2, v3)):
                cd_mid = 0.5 * (va.x_a + vb.x_a)
                cd_mid = _merge_local_duals_vector(
                    [cd_mid], list(va.vd), cdist=cdist
                )[0]
                vd_mid = HC.Vd[tuple(cd_mid)]
                va.vd.add(vd_mid)
                vb.vd.add(vd_mid)
                # Connect edge midpoint to the face dual (ring-walk support)
                vd_mid.connect(vd_face)


def _compute_vd_2d_batch(HC, strategy, cdist, backend) -> None:
    """Batch 2D dual computation using a backend for parallelism.

    Simplex-centric approach: enumerates all unique triangles, batch-
    computes their dual positions via ``backend.batch_dual_positions``,
    then wires dual connectivity per shared edge.  Local merge is
    skipped; global merge (called by ``compute_vd``) handles
    deduplication.

    When ``HC._simplices`` is populated, triangles are read directly from
    the cached list (skipping the flag-complex nn-intersection
    enumeration).  Edge classification (interior vs boundary) is also
    derived from the cache.  Otherwise falls back to the legacy
    nn-intersection path.
    """
    dim = HC.dim

    # Phase 1: Enumerate all unique triangles
    tri_dict: dict[frozenset, tuple] = {}
    cached = getattr(HC, '_simplices', None)
    if cached:
        for simplex in cached:
            v1, v2, v3 = simplex
            key = frozenset((id(v1), id(v2), id(v3)))
            if key not in tri_dict:
                tri_dict[key] = (v1, v2, v3)
    else:
        for v1 in HC.V:
            for v2 in v1.nn:
                for v3 in v1.nn.intersection(v2.nn):
                    key = frozenset((id(v1), id(v2), id(v3)))
                    if key not in tri_dict:
                        tri_dict[key] = (v1, v2, v3)

    if not tri_dict:
        return

    # Phase 2: Batch compute dual positions for all triangles
    tri_keys = list(tri_dict.keys())
    tri_verts = list(tri_dict.values())
    simplex_arr = np.empty((len(tri_keys), 3, dim))
    for i, (va, vb, vc) in enumerate(tri_verts):
        simplex_arr[i, 0] = va.x_a
        simplex_arr[i, 1] = vb.x_a
        simplex_arr[i, 2] = vc.x_a

    dual_pos = backend.batch_dual_positions(simplex_arr, strategy)

    # Phase 3: Create dual vertices and associate with primal vertices
    tri_to_vd: dict[frozenset, object] = {}
    for i, key in enumerate(tri_keys):
        vd = HC.Vd[tuple(dual_pos[i])]
        tri_to_vd[key] = vd
        for v in tri_verts[i]:
            v.vd.add(vd)

    # Phase 4: Wire dual connectivity per shared edge
    if cached:
        # Simplex-aware: derive edges directly from tri_dict, classify
        # boundary as "appears in exactly one triangle".
        from collections import defaultdict
        edge_to_tris: dict = defaultdict(list)
        for key, (v1, v2, v3) in tri_dict.items():
            for va, vb in ((v1, v2), (v1, v3), (v2, v3)):
                ek = frozenset((id(va), id(vb)))
                edge_to_tris[ek].append((tri_to_vd[key], va, vb))

        for adj_list in edge_to_tris.values():
            if len(adj_list) >= 2:
                adj_list[0][0].connect(adj_list[1][0])
            elif len(adj_list) == 1:
                vd_tri, va, vb = adj_list[0]
                cd_mid = 0.5 * (va.x_a + vb.x_a)
                vd_mid = HC.Vd[tuple(cd_mid)]
                va.vd.add(vd_mid)
                vb.vd.add(vd_mid)
                vd_mid.connect(vd_tri)
    else:
        seen_edges: set[frozenset] = set()
        for v1 in HC.V:
            for v2 in v1.nn:
                ek = frozenset((id(v1), id(v2)))
                if ek in seen_edges:
                    continue
                seen_edges.add(ek)

                # Find triangles sharing this edge
                adj = []
                for v3 in v1.nn.intersection(v2.nn):
                    tk = frozenset((id(v1), id(v2), id(v3)))
                    if tk in tri_to_vd:
                        adj.append(tri_to_vd[tk])

                if len(adj) >= 2:
                    # Interior edge: connect two triangle duals
                    adj[0].connect(adj[1])
                elif len(adj) == 1:
                    # Potential boundary edge
                    try:
                        if v1.boundary and v2.boundary:
                            cd_mid = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                            vd_mid = HC.Vd[tuple(cd_mid)]
                            v1.vd.add(vd_mid)
                            v2.vd.add(vd_mid)
                            vd_mid.connect(adj[0])
                    except AttributeError:
                        pass


def _compute_vd_3d_batch(HC, strategy, cdist, backend) -> None:
    """Batch 3D dual computation using a backend for parallelism.

    Simplex-centric approach: enumerates all unique tetrahedra, batch-
    computes their dual positions, then wires dual connectivity per
    shared face.  Boundary faces get an additional face-dual vertex
    (also batch-computed).

    When ``HC._simplices`` is populated, tets are read directly from the
    cached list (skipping the flag-complex nn-intersection enumeration
    that produces ghost K_4 cliques near domain boundaries).  Face
    classification (interior vs boundary) is also derived from the
    cached list.  Otherwise falls back to the legacy nn-intersection
    path.
    """
    dim = HC.dim

    # Phase 1: Enumerate all unique tetrahedra
    tet_dict: dict[frozenset, tuple] = {}
    cached = getattr(HC, '_simplices', None)
    if cached:
        for simplex in cached:
            v1, v2, v3, v4 = simplex
            key = frozenset((id(v1), id(v2), id(v3), id(v4)))
            if key not in tet_dict:
                tet_dict[key] = (v1, v2, v3, v4)
    else:
        for v1 in HC.V:
            for v2 in v1.nn:
                common_12 = v1.nn.intersection(v2.nn)
                for v3 in common_12:
                    common_123 = common_12.intersection(v3.nn)
                    for v4 in common_123:
                        if v4 is v1 or v4 is v2 or v4 is v3:
                            continue
                        key = frozenset((id(v1), id(v2), id(v3), id(v4)))
                        if key not in tet_dict:
                            tet_dict[key] = (v1, v2, v3, v4)

    if not tet_dict:
        return

    # Phase 2a: Batch compute tetrahedron duals
    tet_keys = list(tet_dict.keys())
    tet_verts = list(tet_dict.values())
    simplex_arr = np.empty((len(tet_keys), dim + 1, dim))
    for i, (va, vb, vc, vd_) in enumerate(tet_verts):
        simplex_arr[i, 0] = va.x_a
        simplex_arr[i, 1] = vb.x_a
        simplex_arr[i, 2] = vc.x_a
        simplex_arr[i, 3] = vd_.x_a

    tet_dual_pos = backend.batch_dual_positions(simplex_arr, strategy)

    # Create tetrahedron dual vertices and associate with primal vertices
    tet_to_vd: dict[frozenset, object] = {}
    for i, key in enumerate(tet_keys):
        vd = HC.Vd[tuple(tet_dual_pos[i])]
        tet_to_vd[key] = vd
        for v in tet_verts[i]:
            v.vd.add(vd)

    # Phase 2b: Enumerate faces, classify as interior/boundary
    boundary_faces: list[tuple] = []   # (v1, v2, v3, tet_key)
    interior_pairs: list[tuple] = []   # (tet_key1, tet_key2)

    if cached:
        # Build face -> [tet_key, (v1,v2,v3)] map directly from tet_dict.
        # A face appears in exactly 1 tet -> boundary; exactly 2 -> interior.
        from collections import defaultdict
        face_to_tets: dict = defaultdict(list)
        for tk, (v1, v2, v3, v4) in tet_dict.items():
            for face in (
                (v1, v2, v3),
                (v1, v2, v4),
                (v1, v3, v4),
                (v2, v3, v4),
            ):
                fk = frozenset(id(v) for v in face)
                face_to_tets[fk].append((tk, face))
        for entries in face_to_tets.values():
            if len(entries) == 2:
                interior_pairs.append((entries[0][0], entries[1][0]))
            elif len(entries) == 1:
                tk, (fa, fb, fc) = entries[0]
                boundary_faces.append((fa, fb, fc, tk))
    else:
        seen_faces: set[frozenset] = set()
        for v1 in HC.V:
            for v2 in v1.nn:
                common_12 = v1.nn.intersection(v2.nn)
                for v3 in common_12:
                    fk = frozenset((id(v1), id(v2), id(v3)))
                    if fk in seen_faces:
                        continue
                    seen_faces.add(fk)

                    # Find tetrahedra containing this face
                    common_123 = common_12.intersection(v3.nn)
                    adj_tets = []
                    for v4 in common_123:
                        if v4 is v1 or v4 is v2 or v4 is v3:
                            continue
                        tk = frozenset((id(v1), id(v2), id(v3), id(v4)))
                        if tk in tet_to_vd:
                            adj_tets.append(tk)

                    if len(adj_tets) >= 2:
                        interior_pairs.append((adj_tets[0], adj_tets[1]))
                    elif len(adj_tets) == 1:
                        if (
                            _has_boundary(v1)
                            and _has_boundary(v2)
                            and _has_boundary(v3)
                        ):
                            boundary_faces.append(
                                (v1, v2, v3, adj_tets[0])
                            )

    # Batch compute boundary face duals
    face_dual_pos = None
    if boundary_faces:
        face_arr = np.empty((len(boundary_faces), 3, dim))
        for i, (va, vb, vc, _) in enumerate(boundary_faces):
            face_arr[i, 0] = va.x_a
            face_arr[i, 1] = vb.x_a
            face_arr[i, 2] = vc.x_a
        face_dual_pos = backend.batch_dual_positions(face_arr, strategy)

    # Phase 3: Wire interior face connectivity
    for tk1, tk2 in interior_pairs:
        tet_to_vd[tk1].connect(tet_to_vd[tk2])

    # Phase 4: Wire boundary face connectivity
    for i, (v1, v2, v3, tk) in enumerate(boundary_faces):
        # Boundary face dual
        vd_face = HC.Vd[tuple(face_dual_pos[i])]
        v1.vd.add(vd_face)
        v2.vd.add(vd_face)
        v3.vd.add(vd_face)
        # Connect tet dual to face dual
        tet_to_vd[tk].connect(vd_face)
        # Edge midpoint duals for all 3 edges of the boundary face
        # (sequential path processes each face 6 times, creating all 3)
        for va, vb in ((v1, v2), (v1, v3), (v2, v3)):
            cd_mid = va.x_a + 0.5 * (vb.x_a - va.x_a)
            vd_mid = HC.Vd[tuple(cd_mid)]
            va.vd.add(vd_mid)
            vb.vd.add(vd_mid)


def _has_boundary(v) -> bool:
    """Check if vertex has the boundary attribute set to True."""
    try:
        return v.boundary
    except AttributeError:
        return False


def _compute_vd_nd(HC, strategy: DualStrategy, cdist: float) -> None:
    """N-D dual computation for arbitrary dimension.

    For each primal (dim-1)-simplex (codimension-1 face), finds the
    1 or 2 dim-simplices sharing it and places dual vertices at
    ``strategy(simplex_vertices)`` for each. The two dual vertices
    are then connected.

    Algorithm:
    - A (dim-1)-face is shared by 1 or 2 dim-simplices
    - For each such face, place dual vertex at strategy(simplex_verts)
    - Connect the two duals (or connect to boundary dual for boundary faces)
    - Associate duals with primal vertices of each simplex

    :param HC: Complex with ``HC.V`` populated and boundary marked.
    :param strategy: Dual vertex placement strategy (barycenter/circumcenter).
    :param cdist: Tolerance for merging nearby dual vertices.
    """
    dim = HC.dim
    processed = set()  # Track processed faces to avoid duplicates

    for v1 in HC.V:
        for v2 in v1.nn:
            # Build all (dim-1)-faces containing edge (v1, v2)
            _build_faces_and_duals(
                HC, v1, v2, dim, strategy, cdist, processed
            )


def _build_faces_and_duals(HC, v1, v2, dim, strategy, cdist, processed) -> None:
    """Build all (dim-1)-faces containing edge (v1,v2) and compute duals.

    A (dim-1)-face has dim vertices. Starting with v1, v2, we extend
    by finding common neighbors that form valid faces (all mutually connected).
    """
    # Find common neighbors of v1 and v2
    common = v1.nn.intersection(v2.nn) - {v1, v2}

    # Start extending the face from [v1, v2]
    _extend_face(HC, [v1, v2], common, dim, strategy, cdist, processed)


def _extend_face(HC, face_verts, candidates, dim, strategy, cdist, processed):
    """Recursively extend a partial face until it has `dim` vertices.

    When we have dim vertices, we have a complete (dim-1)-face.
    Then find the 1 or 2 dim-simplices containing it by looking for
    apex vertices connected to all face vertices.
    """
    if len(face_verts) == dim:
        # We have a complete (dim-1)-face
        face_key = frozenset(id(v) for v in face_verts)
        if face_key in processed:
            return
        processed.add(face_key)

        # Find vertices that complete a dim-simplex (apex vertices)
        # These are vertices connected to ALL face vertices
        apex_set = set(face_verts[0].nn)
        for v in face_verts[1:]:
            apex_set = apex_set.intersection(v.nn)
        apex_set -= set(face_verts)

        apex_list = list(apex_set)
        if len(apex_list) == 0:
            return

        # Get local dual neighborhood for merging
        v1_d_nn = list(face_verts[0].vd)

        # Pre-build face coordinate array (reused for both branches)
        face_coords = np.empty((dim, dim))
        for i, v in enumerate(face_verts):
            face_coords[i] = v.x_a

        if len(apex_list) == 1:
            # Boundary face: only one dim-simplex
            apex = apex_list[0]

            # Dual vertex at the simplex
            verts = np.empty((dim + 1, dim))
            verts[:dim] = face_coords
            verts[dim] = apex.x_a
            cd1 = strategy(verts)

            # Boundary dual: place at the (dim-1)-face itself
            cd2 = strategy(face_coords)

            # Merge nearby duals
            (cd1, cd2) = _merge_local_duals_vector(
                [cd1, cd2], v1_d_nn, cdist=cdist
            )

            vd1 = HC.Vd[tuple(cd1)]
            vd2 = HC.Vd[tuple(cd2)]
            vd1.connect(vd2)

            # Associate duals with face vertices
            for v in face_verts:
                v.vd.add(vd1)
                v.vd.add(vd2)

            # Associate simplex dual with apex
            if not hasattr(apex, 'vd'):
                apex.vd = set()
            apex.vd.add(vd1)

        elif len(apex_list) >= 2:
            # Interior face: two or more simplices (use first two)
            verts = np.empty((dim + 1, dim))
            verts[:dim] = face_coords

            # First simplex
            verts[dim] = apex_list[0].x_a
            cd1 = strategy(verts)

            # Second simplex
            verts[dim] = apex_list[1].x_a
            cd2 = strategy(verts)

            # Merge nearby duals
            (cd1, cd2) = _merge_local_duals_vector(
                [cd1, cd2], v1_d_nn, cdist=cdist
            )

            vd1 = HC.Vd[tuple(cd1)]
            vd2 = HC.Vd[tuple(cd2)]
            vd1.connect(vd2)

            # Associate duals with simplex vertices
            for v in face_verts + [apex_list[0]]:
                v.vd.add(vd1)

            for v in face_verts + [apex_list[1]]:
                v.vd.add(vd2)

        return

    # Need more vertices — extend by one from candidates
    for v_next in candidates:
        if v_next in face_verts:
            continue
        # Check v_next is connected to all current face vertices
        if all(v_next in fv.nn for fv in face_verts):
            # Recursively extend with reduced candidate set
            new_candidates = candidates.intersection(v_next.nn) - {v_next}
            _extend_face(
                HC,
                face_verts + [v_next],
                new_candidates,
                dim,
                strategy,
                cdist,
                processed,
            )


def _compute_vd_nd_simplex_aware(HC, strategy: DualStrategy, cdist: float) -> None:
    """Simplex-aware N-D dual computation (``dim > 3``).

    Uses the explicit top-dim simplex list ``HC._simplices`` (each element a
    ``dim+1``-tuple of primal vertex objects) instead of the
    ``vi.nn ∩ vj.nn`` face-extension heuristic of :func:`_compute_vd_nd`,
    which can return ghost apices on Delaunay-derived complexes with skinny
    simplices.

    Algorithm (generalises :func:`_compute_vd_3d_simplex_aware`):

    1. For each top simplex, place a dual vertex at
       ``strategy(simplex_vertices)`` and associate it with all ``dim+1``
       primal vertices.
    2. Build a ``(dim-1)``-face -> ``[simplex index, ...]`` map.
    3. For each interior face (shared by two simplices) connect the two
       simplex duals; for each boundary face (one simplex) create a face
       dual at ``strategy(face_vertices)`` and connect the simplex dual to
       it, associating it with the face vertices.
    """
    from collections import defaultdict

    simplices = HC._simplices
    if not simplices:
        return

    dim = HC.dim
    verts_buf = np.empty((dim + 1, dim))

    # --- Step 1: simplex dual vertex for each top simplex ---
    simplex_to_vd = {}
    for si, simplex in enumerate(simplices):
        for i, v in enumerate(simplex):
            verts_buf[i] = v.x_a
        cd = strategy(verts_buf)
        cd = _merge_local_duals_vector([cd], list(simplex[0].vd), cdist=cdist)[0]
        vd = HC.Vd[tuple(cd)]
        simplex_to_vd[si] = vd
        for v in simplex:
            v.vd.add(vd)

    # --- Step 2: (dim-1)-face -> [simplex index] map ---
    face_to_simps = defaultdict(list)
    for si, simplex in enumerate(simplices):
        for i in range(dim + 1):
            face = tuple(sorted(
                id(simplex[j]) for j in range(dim + 1) if j != i
            ))
            face_to_simps[face].append(si)

    # --- Step 3: connect duals across faces ---
    face_coords = np.empty((dim, dim))
    for face, simp_ids in face_to_simps.items():
        if len(simp_ids) == 2:
            simplex_to_vd[simp_ids[0]].connect(simplex_to_vd[simp_ids[1]])
        elif len(simp_ids) == 1:
            # in the order of the owning simplex, not in ``id()`` order
            face_verts = [v for v in simplices[simp_ids[0]] if id(v) in face]
            for i, v in enumerate(face_verts):
                face_coords[i] = v.x_a
            cd_face = strategy(face_coords)
            cd_face = _merge_local_duals_vector(
                [cd_face], list(face_verts[0].vd), cdist=cdist
            )[0]
            vd_face = HC.Vd[tuple(cd_face)]
            for v in face_verts:
                v.vd.add(vd_face)
            simplex_to_vd[simp_ids[0]].connect(vd_face)
