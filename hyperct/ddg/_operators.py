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
