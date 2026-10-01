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

In 2D the polygon of a boundary vertex is the half cell truncated by
the domain boundary: the open chain of dual vertices between the two
boundary-edge midpoints, closed through the primal vertex itself, so
the cells of all vertices tile the mesh (corners and kinked boundaries
included).  The 3D face routine still covers interior vertices only.
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

    Notes
    -----
    The polygon is ordered by walking the primal-edge adjacency around
    ``v``: each primal edge (v, v_j) owns the two dual vertices shared
    with ``v_j``, and consecutive dual vertices around the cell share a
    primal edge.  This is exact for any cell shape.  The previous
    angular sort about ``v`` assumed the cell is star-convex about the
    primal vertex, which fails for circumcentric duals once a
    circumcenter falls outside its (obtuse) triangle and produced
    mis-ordered polygons with the wrong area.  For a boundary vertex
    the walk runs along the open chain of dual vertices and the polygon
    is closed through ``v`` itself (the last polygon vertex before
    orientation).  The angular sort is kept only as a fallback for
    degenerate connectivity; it does not contain ``v`` and therefore
    under-measures a boundary cell that is not on a straight boundary.
    """
    polygon = _dual_cell_polygon_2d_walk(v, include_edge_midpoints)
    if polygon is None:
        polygon = _dual_cell_polygon_2d_angular(v, include_edge_midpoints)

    if len(polygon) < 3:
        raise ValueError(
            f"Vertex {v.x} has {len(polygon)} dual cell polygon "
            "vertices; expected >= 3 for an interior 2D vertex."
        )
    return polygon


def _dual_cell_polygon_2d_walk(v, include_edge_midpoints):
    """Order the dual cell polygon by walking primal-edge adjacency.

    Interior vertex: the dual vertices form one closed cycle.  Boundary
    vertex: they form one open chain between the midpoint duals of the
    two boundary edges, and the cell is closed through ``v`` itself.

    Returns ``None`` when the walk forms neither (degenerate
    connectivity), signalling the caller to fall back to the angular
    sort.
    """
    # Each primal edge (v, v_j) owns exactly two dual vertices; each
    # dual vertex (triangle around v) touches exactly two primal edges
    # of v, except the midpoint dual of a boundary edge, which touches
    # only that edge.
    edge_duals = {}          # id(v_j) -> (mp_j, [vd, vd])
    vd_edges = {}            # id(vd)  -> list of id(v_j)
    vd_pos = {}
    for v_j in v.nn:
        shared = list(v.vd.intersection(v_j.vd))
        if len(shared) != 2:
            return None
        mp = 0.5 * (v.x_a[:2] + v_j.x_a[:2])
        edge_duals[id(v_j)] = (mp, shared)
        for vd in shared:
            vd_edges.setdefault(id(vd), []).append(id(v_j))
            vd_pos[id(vd)] = vd.x_a[:2].copy()
    ends = [vd_id for vd_id, e in vd_edges.items() if len(e) == 1]
    if len(ends) not in (0, 2) or any(len(e) > 2 for e in vd_edges.values()):
        return None

    if ends:
        # Boundary vertex: walk the open chain end to end, then close
        # the half cell through the primal vertex.  Leaving v out (as
        # the angular fallback does) drops the triangle (end, v, end):
        # nothing on a straight boundary, 3/4 of a right-angle corner
        # cell, and most of the response to a free-surface vertex
        # moving along its normal.
        curr_vd, curr_j = ends[0], vd_edges[ends[0]][0]
        pts = [vd_pos[curr_vd]]
        for _ in range(len(edge_duals)):
            mp, pair = edge_duals[curr_j]
            if include_edge_midpoints:
                pts.append(mp)
            curr_vd = next(id(x) for x in pair if id(x) != curr_vd)
            pts.append(vd_pos[curr_vd])
            if curr_vd == ends[1]:
                break
            curr_j = next(j for j in vd_edges[curr_vd] if j != curr_j)
        n_pts = len(vd_edges) + (len(edge_duals) if include_edge_midpoints
                                 else 0)
        if curr_vd != ends[1] or len(pts) != n_pts:
            return None      # did not run through all dual vertices
        pts.append(v.x_a[:2].copy())
    else:
        # Walk the cycle: vd -> other primal edge -> other vd -> ...
        j_ids = list(edge_duals)
        start_j = j_ids[0]
        mp0, (vd_a, vd_b) = edge_duals[start_j]
        cycle = []               # list of (vd_id, j_id_leaving_it)
        curr_vd, curr_j = id(vd_b), start_j
        for _ in range(len(j_ids)):
            next_j = next(j for j in vd_edges[curr_vd] if j != curr_j)
            cycle.append((curr_vd, next_j))
            curr_j = next_j
            _, pair = edge_duals[next_j]
            curr_vd = next(id(x) for x in pair if id(x) != curr_vd)
        if (curr_vd != id(vd_b)
                or len({c[0] for c in cycle}) != len(vd_edges)):
            return None          # did not close over all dual vertices

        pts = []
        for vd_id, j_id in cycle:
            pts.append(vd_pos[vd_id])
            if include_edge_midpoints:
                pts.append(edge_duals[j_id][0])
    positions = np.array(pts)

    # Drop consecutive coincident points (a circumcenter of a right
    # triangle lies exactly at an edge midpoint)
    keep = [0]
    for k in range(1, len(positions)):
        if not np.allclose(positions[k], positions[keep[-1]], atol=1e-12):
            keep.append(k)
    if len(keep) > 1 and np.allclose(positions[keep[-1]], positions[keep[0]],
                                     atol=1e-12):
        keep.pop()
    positions = positions[keep]

    # Orient counterclockwise (shoelace sign)
    if _shoelace_area(positions) < 0:
        positions = positions[::-1]
    return positions


def _dual_cell_polygon_2d_angular(v, include_edge_midpoints):
    """Legacy ordering: angular sort about the primal vertex.

    Only valid for cells that are star-convex about ``v``; kept as a
    fallback for boundary/degenerate connectivity.
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
        return positions

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
