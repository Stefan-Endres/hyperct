"""
Dual cell geometry extraction for hyperct simplicial complexes.

Provides ordered boundary representations of dual cells around primal
vertices.  These are needed for analytical integration over dual cells
(e.g. via the divergence theorem) and for computing exact dual cell areas.

Two 2D polygon formulations are supported:

- **"barycentric"** (default): The polygon vertices are the dual vertices
  in ``v.vd`` only (barycenters of adjacent triangles for interior
  vertices, plus edge midpoints already in ``v.vd`` for boundary edges).
  This polygon connects barycenters directly.

- **"barycentric_dual_p_ij"**: The polygon includes both the dual vertices
  AND the midpoints of all primal edges incident on ``v``.  This is the
  standard DEC barycentric dual cell whose boundary alternates between
  edge midpoints and triangle barycenters.

Both formulations also work with circumcentric duals — the polygon
vertices are circumcenters instead of barycenters, but the structure is
identical.

Currently only interior vertices are supported.  Boundary vertex dual
cells (half-cells truncated by the domain boundary) are deferred.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1D dual cell
# ---------------------------------------------------------------------------

def dual_cell_vertices_1d(v) -> tuple[float, float]:
    """Return the interval endpoints (a, b) of a 1D dual cell.

    Parameters
    ----------
    v : vertex object
        Must have ``v.vd`` populated by ``compute_vd``.

    Returns
    -------
    tuple[float, float]
        ``(a, b)`` with ``a < b``.

    Raises
    ------
    ValueError
        If the vertex has fewer than 2 dual vertices (boundary vertex).
    """
    vd_list = list(v.vd)
    if len(vd_list) < 2:
        raise ValueError(
            f"Vertex {v.x} has {len(vd_list)} dual vertices; "
            "expected 2 for an interior 1D vertex."
        )
    positions = [vd.x_a[0] for vd in vd_list]
    a, b = min(positions), max(positions)
    return (a, b)


# ---------------------------------------------------------------------------
# 2D dual cell polygon — two formulations
# ---------------------------------------------------------------------------

def dual_cell_polygon_2d(
    v,
    include_edge_midpoints: bool = True,
) -> np.ndarray:
    """Return ordered CCW polygon vertices of a 2D dual cell.

    Parameters
    ----------
    v : vertex object
        Must have ``v.vd``, ``v.nn``, and ``v.x_a`` populated.
    include_edge_midpoints : bool
        If ``True`` (default, "barycentric_dual_p_ij" formulation), the
        polygon includes midpoints of all primal edges as additional
        vertices.  The resulting polygon alternates between edge midpoints
        and dual vertices (barycenters/circumcenters).

        If ``False`` ("barycentric" formulation), the polygon uses only
        the dual vertices in ``v.vd``.

    Returns
    -------
    np.ndarray
        Polygon vertices, shape ``(N, 2)``, ordered counterclockwise.
    """
    cx, cy = v.x_a[0], v.x_a[1]

    pts = []

    # 1. Dual vertices (triangle barycenters / circumcenters)
    for vd in v.vd:
        pts.append(vd.x_a[:2].copy())

    # 2. Optionally add edge midpoints
    if include_edge_midpoints:
        for v_j in v.nn:
            mp = 0.5 * (v.x_a[:2] + v_j.x_a[:2])
            pts.append(mp)

    positions = np.array(pts)

    # Deduplicate (boundary edge midpoints may already be in v.vd)
    if len(positions) > 0:
        _, unique_idx = np.unique(
            np.round(positions, decimals=12), axis=0, return_index=True
        )
        positions = positions[np.sort(unique_idx)]

    if len(positions) < 3:
        raise ValueError(
            f"Vertex {v.x} has {len(positions)} dual cell polygon "
            "vertices; expected >= 3 for an interior 2D vertex."
        )

    # Angular sort around the primal vertex (star-convex guarantee)
    angles = np.arctan2(positions[:, 1] - cy, positions[:, 0] - cx)
    order = np.argsort(angles)
    return positions[order]


# ---------------------------------------------------------------------------
# Exact dual cell area (replaces approximate d_area)
# ---------------------------------------------------------------------------

def dual_cell_area_2d(
    v,
    include_edge_midpoints: bool = True,
) -> float:
    """Exact area of a 2D dual cell via the shoelace formula.

    This replaces the approximate ``d_area(v)`` from
    ``hyperct.ddg._operators`` which uses ``0.5 * b * h`` on
    sub-triangles (incorrect for non-right triangles).

    Parameters
    ----------
    v : vertex object
        Must have ``v.vd``, ``v.nn``, and ``v.x_a`` populated.
    include_edge_midpoints : bool
        Whether to include edge midpoints in the polygon (see
        :func:`dual_cell_polygon_2d`).

    Returns
    -------
    float
        Exact area of the dual cell (always positive).
    """
    polygon = dual_cell_polygon_2d(v, include_edge_midpoints)
    return abs(_shoelace_area(polygon))


def _shoelace_area(polygon: np.ndarray) -> float:
    """Signed area of a 2D polygon via the shoelace formula.

    Positive for counterclockwise orientation.

    Parameters
    ----------
    polygon : np.ndarray
        Polygon vertices, shape ``(N, 2)``.

    Returns
    -------
    float
        Signed area (positive if CCW).
    """
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------
# 3D dual cell
# ---------------------------------------------------------------------------

def dual_cell_faces_3d(
    v,
    HC,
    include_face_barycenters: bool = True,
) -> list[np.ndarray]:
    """Return face polygons of a 3D dual cell polyhedron.

    Each face corresponds to a primal edge incident on ``v``.

    Parameters
    ----------
    v : vertex object
        Must have ``v.vd``, ``v.nn``, and ``v.x_a`` populated.
    HC : Complex
        Simplicial complex with duals computed.
    include_face_barycenters : bool
        If True (default), build the DEC p_ij polygon by interleaving
        tet barycenters with face barycenters ``(x_i + x_j + x_k)/3``
        of the primal triangular faces shared by consecutive tetrahedra.
        This is the correct construction for linear precision on
        barycentric duals.  If False, use only tet barycenters (legacy).

    Returns
    -------
    list[np.ndarray]
        Each element is an ``(M, 3)`` array of ordered face polygon
        vertices.
    """
    faces = []
    x_i = v.x_a[:3]
    for v_j in v.nn:
        shared_vd = v.vd.intersection(v_j.vd)
        if len(shared_vd) < 3:
            continue  # Degenerate or boundary face — skip

        # --- Ring-walk via dual vertex connectivity ---
        shared_list = list(shared_vd)
        ring = [shared_list[0]]
        remaining = set(shared_list[1:])
        while remaining:
            curr = ring[-1]
            nxt = None
            for cand in remaining:
                if cand in curr.nn:
                    nxt = cand
                    break
            if nxt is None:
                break
            ring.append(nxt)
            remaining.discard(nxt)

        if len(ring) < 3:
            # Connectivity walk failed — fall back to angular sort
            ring = _angular_sort_3d(shared_list)
            if ring is None:
                continue

        ordered_pts = np.array([vd.x_a[:3] for vd in ring])

        # --- Interleave face barycenters for p_ij construction ---
        if include_face_barycenters:
            common_nbs = list(v.nn.intersection(v_j.nn))
            if common_nbs:
                x_j = v_j.x_a[:3]
                interleaved = []
                n_ring = len(ordered_pts)
                for k in range(n_ring):
                    interleaved.append(ordered_pts[k])
                    tb_k = ordered_pts[k]
                    tb_next = ordered_pts[(k + 1) % n_ring]
                    mid = 0.5 * (tb_k + tb_next)
                    best_fb = None
                    best_dist = np.inf
                    for cn in common_nbs:
                        fb = (x_i + x_j + cn.x_a[:3]) / 3.0
                        d = np.linalg.norm(fb - mid)
                        if d < best_dist:
                            best_dist = d
                            best_fb = fb
                    if best_fb is not None:
                        interleaved.append(best_fb)
                ordered_pts = np.array(interleaved)

        # Orient outward: face normal should point away from v.
        # Use the total area vector (sum of centroid-fan triangles) for
        # a robust orientation check — the first 3 points can be nearly
        # collinear for interleaved p_ij polygons.
        face_centroid = ordered_pts.mean(axis=0)
        outward = face_centroid - x_i
        face_normal = np.zeros(3)
        n_fp = len(ordered_pts)
        for k in range(n_fp):
            face_normal += np.cross(
                ordered_pts[k] - face_centroid,
                ordered_pts[(k + 1) % n_fp] - face_centroid,
            )
        if np.dot(face_normal, outward) < 0:
            ordered_pts = ordered_pts[::-1]

        faces.append(ordered_pts)

    return faces


def _angular_sort_3d(vd_list):
    """Sort dual vertices by angle around their centroid (fallback)."""
    pts = np.array([vd.x_a[:3] for vd in vd_list])
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    if len(centered) < 3:
        return None

    n_vec = np.cross(centered[1] - centered[0],
                     centered[2] - centered[0])
    n_norm = np.linalg.norm(n_vec)
    if n_norm < 1e-30:
        return None
    n_vec = n_vec / n_norm

    abs_n = np.abs(n_vec)
    if abs_n[0] <= abs_n[1] and abs_n[0] <= abs_n[2]:
        ref = np.array([1.0, 0.0, 0.0])
    elif abs_n[1] <= abs_n[2]:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])

    e1 = np.cross(n_vec, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_vec, e1)

    local_2d = np.column_stack([centered @ e1, centered @ e2])
    angles = np.arctan2(local_2d[:, 1], local_2d[:, 0])
    order = np.argsort(angles)
    return [vd_list[i] for i in order]
