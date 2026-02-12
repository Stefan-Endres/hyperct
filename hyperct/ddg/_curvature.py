"""
Curvature computations using Heron's formula for DDG dual meshes.

This module provides functions to compute normal area, mean curvature, and
integrated curvature on triangulated surfaces represented as vertex objects
with neighbor connectivity.
"""
from __future__ import annotations

import numpy as np

from ._geometry import normalized


def HNdC_ijk(
    e_ij: np.ndarray, l_ij: float, l_jk: float, l_ik: float
) -> tuple[np.ndarray, float]:
    """
    Compute the dual edge and dual area using Heron's formula.

    Implements the cotangent weight formula for discrete differential geometry:
    w_ij = (1/2) cot(theta_i^jk) where theta_i^jk is the angle at vertex i
    in triangle ijk opposite to edge jk.

    :param e_ij: Edge vector from vertex j to vertex i (shape (3,)).
    :param l_ij: Length of edge ij.
    :param l_jk: Length of edge jk.
    :param l_ik: Length of edge ik.
    :return: Tuple of (hnda_ijk, c_ijk):
        - hnda_ijk: Curvature vector contribution from this triangle.
        - c_ijk: Dual area contribution from this triangle.
    """
    lengths = [l_ij, l_jk, l_ik]
    # Sort the list, python sorts from the smallest to largest element:
    lengths.sort()
    # We must use a >= b >= c in floating-point stable Heron's formula:
    a = lengths[2]
    b = lengths[1]
    c = lengths[0]
    A = (1 / 4.0) * np.sqrt(
        (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    )
    # Dual weights (scalar):
    w_ij = (1 / 8.0) * (l_jk**2 + l_ik**2 - l_ij**2) / A

    # Mean normal curvature (1x3 vector):
    hnda_ijk = w_ij * e_ij

    # Dual areas
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def normal_area(v, n_i: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the discrete normal area vector of vertex v_i.

    Note: This function does not handle boundary edges (single-neighbor case).
    For meshes with boundaries, use mean_curvature instead.

    Sign convention: The edge vector e_ij points FROM j TO i (i.e., vi.x_a - vj.x_a)
    rather than the natural i-to-j direction. This matches the standard DEC
    cotangent weight formula where w_ij uses the opposite edge orientation.

    :param v: Vertex object with attributes:
        - v.x or v.x_a: Vertex position array (shape (3,)).
        - v.nn: Set of neighboring vertex objects.
    :param n_i: Optional normal vector at vertex i (shape (3,)).
        If None, defaults to the vertex position vector.
    :return: Normal area vector (shape (3,)).
    """
    if n_i is None:
        n_i = v.x if hasattr(v, "x") else v.x_a

    # Initiate
    NdA_i = np.zeros(3)
    vi = v
    for vj in v.nn:
        # Compute the intersection set of vertices i and j:
        e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        # Sign convention: flip edge to point FROM j TO i
        e_ij = -e_ij
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        # Discrete vector area:
        # Simplex areas of ijk and normals
        wedge_ij_ik = np.cross(e_ij, e_ik)
        # If the wrong direction was chosen, choose the other:
        if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
            e_ij = vi.x_a - vj.x_a

        if len(e_i_int_e_j) == 1:  # boundary edge
            pass  # ignore for now

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_li = l_il
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            # Save results
            NdA_i += hnda_ijk
            NdA_i += hnda_ijl

    return NdA_i


def mean_curvature(
    v, n_i: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """
    Compute the mean normal curvature of vertex v.

    This function handles both interior and boundary edges, accumulating
    curvature contributions from all adjacent triangles.

    Sign convention: The edge vector e_ij points FROM j TO i (i.e., vi.x_a - vj.x_a)
    rather than the natural i-to-j direction. This matches the standard DEC
    cotangent weight formula where w_ij uses the opposite edge orientation.

    :param v: Vertex object with attributes:
        - v.x or v.x_a: Vertex position array (shape (3,)).
        - v.nn: Set of neighboring vertex objects.
    :param n_i: Optional normal vector at vertex i (shape (3,)).
        If None, defaults to the vertex position vector.
    :return: Tuple of (HNdA_i, C_i):
        - HNdA_i: Mean normal curvature vector (shape (3,)).
        - C_i: Total dual area around the vertex.
    """
    if n_i is None:
        n_i = v.x if hasattr(v, "x") else v.x_a

    # Initiate
    HNdA_i = np.zeros(3)
    C_i = 0.0
    vi = v
    for vj in v.nn:
        # Compute the intersection set of vertices i and j:
        e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        # Sign convention: flip edge to point FROM j TO i
        e_ij = -e_ij
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        if len(e_i_int_e_j) == 1:  # boundary edge
            vk = list(e_i_int_e_j)[0]  # Boundary edge index
            # Compute edges in triangle ijk
            e_ik = vk.x_a - vi.x_a
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Save results
            HNdA_i += hnda_ijk
            C_i += c_ijk

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_li = l_il
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            # Save results
            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return HNdA_i, C_i


def integrated_curvature(
    v, n_i: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """
    Compute the integrated mean curvature at vertex v.

    This uses interpolated curvature values from neighboring vertices
    (assumes v.hnda_i and neighbor.hnda_i attributes exist).

    Sign convention: The edge vector e_ij points FROM j TO i (i.e., vi.x_a - vj.x_a)
    rather than the natural i-to-j direction. This matches the standard DEC
    cotangent weight formula where w_ij uses the opposite edge orientation.

    :param v: Vertex object with attributes:
        - v.x or v.x_a: Vertex position array (shape (3,)).
        - v.nn: Set of neighboring vertex objects.
        - v.hnda_i: Curvature vector at this vertex (shape (3,)).
    :param n_i: Optional normal vector at vertex i (shape (3,)).
        If None, defaults to the vertex position vector.
    :return: Tuple of (HNdA_i, C_i):
        - HNdA_i: Integrated curvature vector (shape (3,)).
        - C_i: Total dual area around the vertex.
    """
    if n_i is None:
        n_i = v.x if hasattr(v, "x") else v.x_a

    # Initiate
    HNdA_i = np.zeros(3)
    C_i = 0.0
    vi = v
    for vj in v.nn:
        # Compute the intersection set of vertices i and j:
        e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        # Sign convention: flip edge to point FROM j TO i
        e_ij = -e_ij
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        if len(e_i_int_e_j) == 1:  # boundary edge
            vk = list(e_i_int_e_j)[0]  # Boundary edge index
            # Compute edges in triangle ijk
            e_ik = vk.x_a - vi.x_a
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)

            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijk, c_ijk = HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            # Save results
            HNdA_i += hnda_ijk
            C_i += c_ijk

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijk, c_ijk = HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_li = l_il
            l_jl = np.linalg.norm(e_jl)
            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijl, c_ijl = HNdC_ijk(e_hnda_ij, l_ij, l_jl, l_il)

            # Save results
            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return HNdA_i, C_i
