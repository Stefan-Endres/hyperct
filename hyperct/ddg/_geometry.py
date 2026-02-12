"""
Shared geometry utilities for DDG dual computations.

All helper functions used by both barycentric and circumcentric dual
computations live here to avoid code duplication.
"""
from __future__ import annotations

import numpy as np


def normalized(a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    """Normalize array along given axis."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def _set_boundary(v, val: bool = True) -> None:
    """Set the boundary property on a vertex."""
    v.boundary = val


def _merge_local_duals_vector(
    x_a_l: list[np.ndarray],
    Vd_cache,
    cdist: float = 1e-10,
) -> list[np.ndarray]:
    """Merge proposed dual vertex positions with existing nearby duals.

    For each proposed position in x_a_l, if a vertex in Vd_cache is
    within cdist, snap to the existing position to avoid floating-point
    duplicates.

    :param x_a_l: List of proposed vertex position arrays.
    :param Vd_cache: Iterable of existing local dual vertices.
    :param cdist: Tolerance for identifying duplicate dual vertices.
    :return: Modified list of vertex positions (merged where appropriate).
    """
    if not x_a_l or not Vd_cache:
        return x_a_l

    # Vectorized implementation for better performance
    # Stack all proposed positions into (n_proposed, dim) array
    proposed = np.stack(x_a_l, axis=0)  # (n_proposed, dim)

    # Stack all existing dual positions into (n_existing, dim) array
    existing = np.array([vd_i.x_a for vd_i in Vd_cache])  # (n_existing, dim)

    # Compute pairwise distances: (n_proposed, n_existing)
    # dist[i, j] = ||proposed[i] - existing[j]||
    diff = proposed[:, np.newaxis, :] - existing[np.newaxis, :, :]  # (n_proposed, n_existing, dim)
    distances = np.linalg.norm(diff, axis=2)  # (n_proposed, n_existing)

    # For each proposed position, find if any existing position is within cdist
    min_distances = np.min(distances, axis=1)  # (n_proposed,)
    min_indices = np.argmin(distances, axis=1)  # (n_proposed,)

    # Snap matching positions to existing positions
    for i in range(len(x_a_l)):
        if min_distances[i] < cdist:
            x_a_l[i] = existing[min_indices[i]]

    return x_a_l


def area_of_polygon(points: np.ndarray) -> float:
    """Calculate the area of a polygon in 3D space via cross products.

    :param points: Array of shape (n, 3) with polygon vertices.
    :return: Total area of the polygon.
    """
    edges = points[1:] - points[:-1]
    cross_products = np.cross(edges[:-1], edges[1:])
    triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    return np.sum(triangle_areas)


def volume_of_geometric_object(
    points: np.ndarray, extra_point: np.ndarray
) -> float:
    """Calculate volume by connecting a base polygon to an extra point.

    :param points: Array of shape (n, 3) with base polygon vertices.
    :param extra_point: Array of shape (3,), the apex point.
    :return: Volume of the geometric object.
    """
    normal_vector = np.cross(points[1] - points[0], points[2] - points[0])
    norm_sq = np.linalg.norm(normal_vector) ** 2
    projected = extra_point - (
        np.dot(extra_point - points[0], normal_vector) / norm_sq
        * normal_vector
    )
    distance = np.linalg.norm(extra_point - projected)
    base_area = area_of_polygon(points)
    return (1.0 / 3.0) * base_area * distance


def _reflect_vertex_over_edge(
    triangle: np.ndarray, target_index: int = 0
) -> np.ndarray:
    """Reflect a vertex of a triangle over the opposing edge.

    :param triangle: Array of shape (3, dim) with triangle vertices.
    :param target_index: Index of vertex to reflect.
    :return: Modified triangle array with reflected vertex.
    """
    p_o = triangle[target_index]
    p_1 = triangle[(target_index + 1) % 3]
    p_2 = triangle[(target_index + 2) % 3]
    p_midpoint = (p_1 + p_2) / 2
    triangle[target_index] = p_o + 2 * (p_midpoint - p_o)
    return triangle


def _find_intersection(
    plane1: np.ndarray, plane2: np.ndarray, plane3: np.ndarray
) -> np.ndarray:
    """Find the intersection point of 3 planes.

    Each plane is specified as [a, b, c, d] for ax + by + cz + d = 0.

    :param plane1: Coefficients of the first plane.
    :param plane2: Coefficients of the second plane.
    :param plane3: Coefficients of the third plane.
    :return: Intersection point as array of shape (3,).
    :raises ValueError: If planes are parallel or nearly parallel.
    """
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    a3, b3, c3, d3 = plane3
    A = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
    if np.linalg.det(A) == 0:
        raise ValueError(
            "The planes are parallel or nearly parallel. No unique solution."
        )
    b_vec = np.array([-d1, -d2, -d3])
    return np.linalg.solve(A, b_vec)


def _find_plane_equation(
    v_1: np.ndarray, v_2: np.ndarray, v_3: np.ndarray
) -> list[float]:
    """Find the plane equation from 3 points.

    Returns [a, b, c, d] for the equation ax + by + cz + d = 0.

    :param v_1: First point.
    :param v_2: Second point.
    :param v_3: Third point.
    :return: List [a, b, c, d] of plane coefficients.
    """
    vector1 = np.array(v_2) - np.array(v_1)
    vector2 = np.array(v_3) - np.array(v_1)
    normal_vector = np.cross(vector1, vector2)
    a, b, c = normal_vector
    d = -np.dot(normal_vector, np.array(v_1))
    return [a, b, c, d]
