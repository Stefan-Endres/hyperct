"""
DDG discrete operators on primal/dual mesh pairs.

Functions for computing dual edge lengths (Hodge star of edges),
dual flux areas/volumes, and dual areas of primal vertices.
"""
from __future__ import annotations

import math

import numpy as np

from ._geometry import normalized, volume_of_geometric_object


def e_star(
    v_i,
    v_j,
    HC,
    n: np.ndarray | None = None,
    dim: int = 2,
) -> float | np.ndarray:
    """Compute the dual of the primary edge e_ij (Hodge star of edge).

    In 2D, returns the dual edge length (scalar). In 3D, returns an array
    of dual triangle vector areas.

    :param v_i: First endpoint of the primary edge.
    :param v_j: Second endpoint of the primary edge.
    :param HC: Complex with ``HC.Vd`` populated by ``compute_vd``.
    :param n: Directional vector for 3D orientation (optional).
    :param dim: Spatial dimension (1, 2, or 3).
    :return: Dual edge length (1D/2D) or array of vector areas (3D).
    """
    if dim == 1:
        # In 1D, the dual of an edge is the distance between dual vertices
        vdnn = v_i.vd.intersection(v_j.vd)
        if len(vdnn) == 1:
            # Boundary edge: dual length is distance from dual to boundary
            vd = list(vdnn)[0]
            return np.linalg.norm(vd.x_a - v_i.x_a)
        vd1, vd2 = list(vdnn)[:2]
        return np.linalg.norm(vd1.x_a - vd2.x_a)

    elif dim == 2:
        # Find the shared dual vertices between v_i and v_j
        vdnn = v_i.vd.intersection(v_j.vd)
        vd1 = list(vdnn)[0]
        vd2 = list(vdnn)[1]
        return np.linalg.norm(vd1.x_a - vd2.x_a)

    elif dim == 3:
        if n is None:
            n = np.array([0.0, 0.0, 0.0])

        # Find the dual vertex at the edge midpoint
        vc_12 = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a
        vc_12 = HC.Vd[tuple(vc_12)]

        # Find local dual points at the intersection of v_i and v_j duals
        dset = v_j.vd.intersection(v_i.vd)
        vd_i = list(dset)[0]

        if _has_boundary(v_i) and _has_boundary(v_j):
            # Find a boundary-starting vertex
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = 3
        else:
            iter_len = len(list(dset))

        # Walk through the dual fan around the edge
        dsetnn = vd_i.nn.intersection(dset)
        vd_j = list(dsetnn)[0]

        A_ij = []  # Triangle vector areas
        for _ in range(iter_len):
            # Compute the discrete vector area of the local triangle
            wedge_ij_ik = np.cross(
                vc_12.x_a - vd_i.x_a, vd_j.x_a - vd_i.x_a
            )
            if np.dot(normalized(wedge_ij_ik), n) < 0:
                wedge_ij_ik = np.cross(
                    vd_j.x_a - vd_i.x_a, vc_12.x_a - vd_i.x_a
                )
            A_ij.append(wedge_ij_ik / 2.0)

            # Advance to the next dual vertex in the fan
            dsetnn_k = vd_j.nn.intersection(dset)
            dsetnn_k.remove(vd_i)
            vd_i = vd_j
            try:
                vd_j = list(dsetnn_k)[0]
            except IndexError:
                pass  # Boundary edge: fan terminates

        return np.array(A_ij)

    else:
        # N-D case: compute (dim-2)-volume of dual cell around edge
        # The dual vertices shared between v_i and v_j form a (dim-2)-polytope
        vdnn = v_i.vd.intersection(v_j.vd)
        if len(vdnn) < 2:
            return 0.0

        dual_pts = np.array([vd.x_a for vd in vdnn])

        if len(dual_pts) == 2:
            # Two dual vertices: distance between them
            return np.linalg.norm(dual_pts[1] - dual_pts[0])

        # Multiple dual vertices: compute (dim-2)-volume via simplex fan
        # Use centroid-based fan triangulation
        centroid = np.mean(dual_pts, axis=0)

        # Compute (dim-2)-volume by summing (dim-2)-simplex volumes
        # Each (dim-2)-simplex is formed by the centroid and (dim-2) vertices
        total_volume = 0.0
        n_pts = len(dual_pts)

        # Generate (dim-2)-simplices using combinations of dual vertices
        # For simplicity, use a fan from centroid to consecutive vertices
        # This approximates the (dim-2)-volume of the polytope
        if n_pts >= dim - 1:
            # Build (dim-2)-simplices: centroid + (dim-2) consecutive vertices
            for i in range(n_pts):
                # Get (dim-2) vertices for the simplex
                simplex_indices = [(i + j) % n_pts for j in range(dim - 2)]
                simplex_verts = dual_pts[simplex_indices]

                # Compute (dim-2)-volume using determinant formula
                # Volume = |det(edges)| / (dim-2)!
                if len(simplex_verts) == dim - 2:
                    edges = simplex_verts - centroid
                    try:
                        # Gram matrix for volume calculation
                        gram = edges @ edges.T
                        vol = np.sqrt(
                            abs(np.linalg.det(gram))
                        ) / math.factorial(dim - 2)
                        total_volume += vol
                    except (np.linalg.LinAlgError, ValueError):
                        pass

        return total_volume if total_volume > 0 else np.linalg.norm(
            dual_pts[-1] - dual_pts[0]
        )


def v_star(
    v_i,
    v_j,
    HC,
    n: np.ndarray | None = None,
    dim: int = 2,
):
    """Compute the dual flux planes and volume of primary edge e_ij.

    In 2D, returns the dual edge length (same as e_star). In 3D, returns
    arrays of dual triangle vector areas and signed tetrahedral volumes.

    :param v_i: First endpoint of the primary edge.
    :param v_j: Second endpoint of the primary edge.
    :param HC: Complex with ``HC.Vd`` populated by ``compute_vd``.
    :param n: Directional vector for 3D orientation (optional).
    :param dim: Spatial dimension (2 or 3).
    :return: (A_ij, V_ij) tuple of vector area array and volume array (3D),
        or scalar dual edge length (2D).
    """
    if dim == 2:
        # Same as e_star in 2D
        vdnn = v_i.vd.intersection(v_j.vd)
        vd1 = list(vdnn)[0]
        vd2 = list(vdnn)[1]
        return np.linalg.norm(vd1.x_a - vd2.x_a)

    elif dim == 3:
        if n is None:
            n = np.array([0.0, 0.0, 0.0])

        # Find the dual vertex at the edge midpoint
        vc_12 = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a
        vc_12 = HC.Vd[tuple(vc_12)]

        # Find local dual points at the intersection
        dset = v_j.vd.intersection(v_i.vd)
        vd_i = list(dset)[0]

        if _has_boundary(v_i) and _has_boundary(v_j):
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = 3
        else:
            iter_len = len(list(dset))

        # Walk through the dual fan
        dsetnn = vd_i.nn.intersection(dset)
        vd_j = list(dsetnn)[0]

        A_ij = []  # Triangle vector areas
        V_ij = []  # Signed tetrahedral volumes
        for _ in range(iter_len):
            # Discrete vector area
            wedge_dij_ik = np.cross(
                vc_12.x_a - vd_i.x_a, vd_j.x_a - vd_i.x_a
            )
            if np.dot(normalized(wedge_dij_ik), n) < 0:
                wedge_dij_ik = np.cross(
                    vd_j.x_a - vd_i.x_a, vc_12.x_a - vd_i.x_a
                )
            A_ij.append(wedge_dij_ik / 2.0)

            # Signed volume of local tetrahedron
            verts = np.zeros([3, 3])
            verts[0] = vc_12.x_a
            verts[1] = vd_i.x_a
            verts[2] = vd_j.x_a
            v_dij_i = volume_of_geometric_object(verts, v_i.x_a)
            V_ij.append(v_dij_i)

            # Advance to next dual vertex
            dsetnn_k = vd_j.nn.intersection(dset)
            dsetnn_k.remove(vd_i)
            vd_i = vd_j
            try:
                vd_j = list(dsetnn_k)[0]
            except IndexError:
                pass  # Boundary edge

        return np.array(A_ij), np.array(V_ij)

    else:
        # N-D case: compute dual (dim-1)-volumes and primal (dim)-volumes
        # around the edge (v_i, v_j)

        # Edge midpoint (or use average of shared duals)
        edge_center = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a

        # Find shared dual vertices
        dset = v_j.vd.intersection(v_i.vd)
        if len(dset) < 2:
            return np.array([]), np.array([])

        dual_pts = np.array([vd.x_a for vd in dset])

        # Compute (dim-1)-volumes of dual faces
        # and (dim)-volumes of primal cells
        A_ij = []  # Dual (dim-1)-volumes
        V_ij = []  # Primal (dim)-volumes

        # For each dual vertex, compute local contribution
        # This is a simplified approach for N-D
        centroid = np.mean(dual_pts, axis=0)

        for vd in dset:
            # (dim-1)-volume: distance-based approximation
            # In general, this should be the (dim-1)-volume of the
            # dual (dim-1)-face, approximated here
            dual_dist = np.linalg.norm(vd.x_a - edge_center)
            A_ij.append(dual_dist ** (dim - 1))

            # (dim)-volume: simplex volume from primal vertex to dual
            # Approximate using distance scaling
            primal_vol = np.linalg.norm(vd.x_a - v_i.x_a) ** dim
            V_ij.append(primal_vol)

        return np.array(A_ij), np.array(V_ij)


def d_area(v) -> float:
    """Compute the dual area of a primal vertex.

    The dual area is the sum of areas of local dual triangles formed
    between the vertex, its neighbours, and their shared dual vertices.

    :param v: A vertex with ``v.nn`` (neighbours) and ``v.vd`` (dual
        vertices) populated by ``compute_vd``.
    :return: Total dual area of the vertex.
    """
    darea = 0.0
    for v2 in v.nn:
        # Find the shared dual vertices
        vdnn = v.vd.intersection(v2.vd)
        # Midpoint between v and v2
        mp = (v.x_a + v2.x_a) / 2.0
        # Height of dual triangle
        h = np.linalg.norm(mp - v.x_a)
        for vd in vdnn:
            # Base of dual triangle
            b = np.linalg.norm(vd.x_a - mp)
            darea += 0.5 * b * h
    return darea


def _has_boundary(v) -> bool:
    """Check if vertex has the boundary attribute set to True."""
    try:
        return v.boundary
    except AttributeError:
        return False


# ---------------------------------------------------------------------------
# Batch e_star — vectorized dual area computation for 3D
# ---------------------------------------------------------------------------

def _walk_fan_3d(v_i, v_j, HC):
    """Walk the dual fan around edge (v_i, v_j) and return triangle data.

    Returns a list of (mid_pos, vdi_pos, vdj_pos) coordinate triples,
    one per dual triangle in the fan.  Raises KeyError/IndexError if the
    dual connectivity is broken (degenerate tetrahedra).
    """
    vc_12_pos = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a
    vc_12 = HC.Vd[tuple(vc_12_pos)]

    dset = v_j.vd.intersection(v_i.vd)
    if not dset:
        return []

    vd_i = next(iter(dset))

    if _has_boundary(v_i) and _has_boundary(v_j):
        if len(vd_i.nn.intersection(dset)) != 1:
            for vd in dset:
                if len(vd.nn.intersection(dset)) == 1:
                    vd_i = vd
                    break
        iter_len = 3
    else:
        iter_len = len(dset)

    dsetnn = vd_i.nn.intersection(dset)
    if not dsetnn:
        return []
    vd_j = next(iter(dsetnn))

    triangles = []
    for _ in range(iter_len):
        triangles.append((vc_12.x_a.copy(), vd_i.x_a.copy(), vd_j.x_a.copy()))
        dsetnn_k = vd_j.nn.intersection(dset)
        dsetnn_k.remove(vd_i)
        vd_i = vd_j
        try:
            vd_j = next(iter(dsetnn_k))
        except StopIteration:
            pass  # boundary

    return triangles


def batch_e_star(vertices, HC, dim=3, backend=None, compute_volumes=False):
    """Compute e_star area vectors for all edges of given vertices.

    Splits the computation into a graph phase (CPU, sequential) that
    walks the dual fan to collect triangle coordinates, and a geometry
    phase (backend-dispatched) that batch-computes cross products.

    Parameters
    ----------
    vertices : iterable of vertex objects
        Primal vertices whose edges will be evaluated.  Boundary vertices
        are silently skipped.
    HC : Complex
        Simplicial complex with duals computed (``compute_vd`` already
        called).
    dim : int
        Spatial dimension (must be 3 for now).
    backend : BatchBackend or None
        Computation backend.  If None, uses vectorized numpy.
    compute_volumes : bool
        If True, also compute dual cell volumes per vertex (v_star).
        Returned as a third element ``vertex_volumes``.

    Returns
    -------
    edge_areas : dict
        ``{id(v): {id(nb): np.ndarray}}`` — per-edge area vector arrays,
        same shape as ``e_star()`` returns.
    failed_vertices : set
        Set of vertex objects where the dual fan walk failed (broken
        duals from degenerate tetrahedra).  These should be promoted
        to boundary.
    vertex_volumes : dict (only if compute_volumes=True)
        ``{id(v): float}`` — dual cell volume per vertex (sum of absolute
        tetrahedron volumes from v_star over all neighbor edges).
    """
    if dim != 3:
        raise NotImplementedError("batch_e_star only supports dim=3")

    # --- Phase 1: Graph traversal (CPU) ---
    # Walk dual fans for all edges, collect triangle coordinates.
    all_mids = []    # (N_total, 3) — edge midpoint positions
    all_vdi = []     # (N_total, 3) — first fan vertex
    all_vdj = []     # (N_total, 3) — second fan vertex
    edge_index = []  # (vid, nbid, start, count) per edge

    # For volume computation: primal vertex position per triangle
    all_primal = []  # (N_total, 3) — primal vertex v_i.x_a per triangle

    failed_vertices = set()
    offset = 0

    for v in vertices:
        if _has_boundary(v):
            continue
        v_failed = False
        for nb in v.nn:
            try:
                tris = _walk_fan_3d(v, nb, HC)
            except (KeyError, IndexError):
                v_failed = True
                break

            n_tri = len(tris)
            if n_tri == 0:
                edge_index.append((id(v), id(nb), offset, 0))
                continue

            for mid_pos, vdi_pos, vdj_pos in tris:
                all_mids.append(mid_pos)
                all_vdi.append(vdi_pos)
                all_vdj.append(vdj_pos)
                if compute_volumes:
                    all_primal.append(v.x_a[:3].copy())

            edge_index.append((id(v), id(nb), offset, n_tri))
            offset += n_tri

        if v_failed:
            failed_vertices.add(v)

    if not all_mids:
        result = ({}, failed_vertices)
        return (*result, {}) if compute_volumes else result

    # --- Phase 2: Vectorized geometry (backend) ---
    mids = np.array(all_mids)   # (N, 3)
    vdi = np.array(all_vdi)     # (N, 3)
    vdj = np.array(all_vdj)    # (N, 3)

    arm1 = mids - vdi           # (N, 3)
    arm2 = vdj - vdi            # (N, 3)

    if backend is not None:
        areas = backend.batch_cross_areas(arm1, arm2)   # (N, 3)
    else:
        areas = np.cross(arm1, arm2) / 2.0              # (N, 3)

    # Volume computation: |det([mid - apex, vdi - apex, vdj - apex])| / 6
    volumes_per_tri = None
    if compute_volumes:
        apex = np.array(all_primal)  # (N, 3)
        e1 = mids - apex
        e2 = vdi - apex
        e3 = vdj - apex
        # Scalar triple product = det of 3x3 matrix per triangle
        det = (e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
             - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
             + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]))
        volumes_per_tri = np.abs(det) / 6.0  # (N,)

    # --- Phase 3: Scatter results back to per-edge dicts ---
    edge_areas = {}
    vertex_volumes = {}
    for vid, nbid, start, count in edge_index:
        if vid not in edge_areas:
            edge_areas[vid] = {}
        if count == 0:
            edge_areas[vid][nbid] = np.empty((0, 3))
        else:
            edge_areas[vid][nbid] = areas[start:start + count]

        # Accumulate volumes per vertex
        if compute_volumes and count > 0:
            vol_sum = float(np.sum(volumes_per_tri[start:start + count]))
            if vid in vertex_volumes:
                vertex_volumes[vid] += vol_sum
            else:
                vertex_volumes[vid] = vol_sum

    if compute_volumes:
        return edge_areas, failed_vertices, vertex_volumes
    return edge_areas, failed_vertices
