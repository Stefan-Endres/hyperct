"""
Dual vertex position strategies.

Each strategy is a callable that takes a (n_verts, dim) array of simplex
vertices and returns the dual vertex position as a (dim,) array.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

DualStrategy = Callable[[np.ndarray], np.ndarray]


def barycenter(verts: np.ndarray) -> np.ndarray:
    """Barycentric dual: centroid of the simplex vertices.

    :param verts: Array of shape (n, dim) with simplex vertex positions.
    :return: Centroid position as array of shape (dim,).
    """
    return np.mean(verts, axis=0)


def circumcenter(verts: np.ndarray) -> np.ndarray:
    """Circumcentric dual: circumcenter of a simplex.

    Handles both full-dimensional simplices (k+1 points in R^k) and
    embedded simplices (k+1 points in R^dim where k < dim), e.g. a
    triangle embedded in 3D space.

    Falls back to barycenter for degenerate configurations.

    :param verts: Array of shape (k+1, dim) with simplex vertex positions.
    :return: Circumcenter position as array of shape (dim,).
    """
    verts = np.asarray(verts, dtype=float)
    n_pts, dim = verts.shape
    k = n_pts - 1  # simplex dimension

    if k < 1:
        return verts[0].copy()

    if k == dim:
        # Full-dimensional simplex: use direct formulas
        return _circumcenter_full(verts, k, dim)
    elif k < dim:
        # Embedded simplex (e.g. triangle in 3D): project, solve, lift
        return _circumcenter_embedded(verts, k, dim)
    else:
        raise ValueError(
            f"Got {n_pts} points in {dim}D; need at most {dim + 1}"
        )


def _circumcenter_full(
    verts: np.ndarray, k: int, dim: int
) -> np.ndarray:
    """Circumcenter of a full-dimensional k-simplex in R^k.

    Uses direct formulas for k=1,2 and linear system for k>=3.
    """
    if k == 1:
        return np.mean(verts, axis=0)

    if k == 2:
        # 2D triangle circumcenter (determinant formula)
        A, B, C = verts
        D = 2 * (
            A[0] * (B[1] - C[1])
            + B[0] * (C[1] - A[1])
            + C[0] * (A[1] - B[1])
        )
        if abs(D) < 1e-12:
            return np.mean(verts, axis=0)
        Ux = (
            (A[0] ** 2 + A[1] ** 2) * (B[1] - C[1])
            + (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1])
            + (C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])
        ) / D
        Uy = (
            (A[0] ** 2 + A[1] ** 2) * (C[0] - B[0])
            + (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0])
            + (C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])
        ) / D
        return np.array([Ux, Uy])

    # General k-simplex: solve M @ sol = rhs
    A = verts[0]
    edges = verts[1:] - A
    M = edges  # (k, dim) where k == dim
    rhs = 0.5 * np.sum(edges ** 2, axis=1)
    try:
        sol = np.linalg.solve(M, rhs)
        return A + sol
    except np.linalg.LinAlgError:
        return np.mean(verts, axis=0)


def _circumcenter_embedded(
    verts: np.ndarray, k: int, dim: int
) -> np.ndarray:
    """Circumcenter of a k-simplex embedded in R^dim (k < dim).

    Projects the simplex into its local k-dimensional subspace,
    computes the circumcenter there, and maps back to R^dim.

    Example: circumcenter of a triangle (k=2) in 3D space (dim=3).
    """
    A = verts[0]
    edges = verts[1:] - A  # (k, dim)

    # Build orthonormal basis for the simplex subspace via QR
    Q, R = np.linalg.qr(edges.T, mode="reduced")  # Q: (dim, k)

    # Project vertices into the k-dimensional subspace
    local_coords = (verts - A) @ Q  # (k+1, k)

    # Compute circumcenter in local coordinates
    local_edges = local_coords[1:]  # (k, k)
    rhs = 0.5 * np.sum(local_edges ** 2, axis=1)
    try:
        sol = np.linalg.solve(local_edges, rhs)
    except np.linalg.LinAlgError:
        return np.mean(verts, axis=0)

    # Map back to ambient space
    return A + Q @ sol
