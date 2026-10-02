"""Element quality metrics for simplicial meshes.

Functions here are pure NumPy and operate on either vertex objects
(any object with an ``x_a`` array attribute) or raw coordinate arrays.

Conventions
-----------
- Angles are returned in **radians** unless noted otherwise.
- Aspect ratio is defined as longest_edge / (2 * inradius) for 2D (so
  equilateral triangle = 1).  For 3D tets, see ``_operations_3d`` (future).
"""

from __future__ import annotations

import math

import numpy as np


def _coords(v) -> np.ndarray:
    """Return a vertex's numpy coordinates, handling both vertex objects
    and raw arrays/tuples."""
    if hasattr(v, "x_a"):
        return np.asarray(v.x_a, dtype=float)
    return np.asarray(v, dtype=float)


def triangle_area(v0, v1, v2) -> float:
    """Signed area of a triangle in 2D, unsigned area in 3D.

    Works for vertex objects or coordinate arrays of length 2 or 3.
    """
    p0 = _coords(v0)
    p1 = _coords(v1)
    p2 = _coords(v2)
    if p0.size >= 3:
        # Unsigned area from cross product magnitude
        e1 = p1[:3] - p0[:3]
        e2 = p2[:3] - p0[:3]
        return 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
    # 2D signed area (positive = CCW)
    e1 = p1[:2] - p0[:2]
    e2 = p2[:2] - p0[:2]
    return 0.5 * float(e1[0] * e2[1] - e1[1] * e2[0])


def triangle_min_angle(v0, v1, v2) -> float:
    """Minimum interior angle of a triangle (radians).

    Returns 0.0 for degenerate triangles (zero edge length or zero area).
    """
    p0 = _coords(v0)
    p1 = _coords(v1)
    p2 = _coords(v2)
    d = max(p0.size, p1.size, p2.size)
    d = min(d, 3)
    p0 = p0[:d]
    p1 = p1[:d]
    p2 = p2[:d]

    a = float(np.linalg.norm(p1 - p2))  # opposite p0
    b = float(np.linalg.norm(p2 - p0))  # opposite p1
    c = float(np.linalg.norm(p0 - p1))  # opposite p2

    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        return 0.0

    # Law of cosines; clamp for numerical safety
    def _angle(x, y, z):
        val = (y * y + z * z - x * x) / (2.0 * y * z)
        val = max(-1.0, min(1.0, val))
        return math.acos(val)

    ang0 = _angle(a, b, c)
    ang1 = _angle(b, c, a)
    ang2 = _angle(c, a, b)
    return min(ang0, ang1, ang2)


def triangle_aspect_ratio(v0, v1, v2) -> float:
    """Aspect ratio = longest_edge / (2 * inradius).

    Equals 1.0 for an equilateral triangle, grows for slivers.
    Returns ``math.inf`` for degenerate triangles.
    """
    p0 = _coords(v0)
    p1 = _coords(v1)
    p2 = _coords(v2)
    d = min(max(p0.size, p1.size, p2.size), 3)
    p0 = p0[:d]
    p1 = p1[:d]
    p2 = p2[:d]

    a = float(np.linalg.norm(p1 - p2))
    b = float(np.linalg.norm(p2 - p0))
    c = float(np.linalg.norm(p0 - p1))
    s = 0.5 * (a + b + c)
    if s <= 0.0:
        return math.inf

    # Heron's formula for unsigned area
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0.0:
        return math.inf
    area = math.sqrt(area_sq)
    if area <= 0.0:
        return math.inf

    inradius = area / s
    if inradius <= 0.0:
        return math.inf
    return max(a, b, c) / (2.0 * inradius)


def edge_length(v_i, v_j) -> float:
    """Euclidean distance between two vertex positions."""
    p_i = _coords(v_i)
    p_j = _coords(v_j)
    d = min(p_i.size, p_j.size)
    return float(np.linalg.norm(p_i[:d] - p_j[:d]))


def iter_triangles_2d(HC):
    """Yield each triangle in a 2D simplicial complex exactly once.

    A triangle is a triple of mutually connected vertices.  To avoid
    counting a triangle 3 times, it is yielded from its first vertex in
    ``HC.V`` order, with its three vertices in that order.  (Until
    2026-10-02 the canonical order was ``id(v)``, i.e. memory addresses:
    which vertex a triangle was listed from, the order of its vertices
    and so the order of the triangles differed between processes.)

    Only vertices of the complex count.  A neighbour that is no longer
    in ``HC.V`` (``HC.V.move`` onto an occupied coordinate drops the
    displaced vertex from the cache but leaves its edges) is skipped.
    """
    verts = list(HC.V)
    rank = {id(w): i for i, w in enumerate(verts)}
    for r, v in enumerate(verts):
        # the neighbours that come later in HC.V, with their rank
        later = [(rank.get(id(w), -1), w) for w in v.nn]
        later = [rw for rw in later if rw[0] > r]
        for r2, v2 in later:
            nn2 = v2.nn
            for r3, v3 in later:
                if r3 > r2 and v3 in nn2:
                    yield v, v2, v3


def mesh_quality_histogram(HC, dim: int = 2, bins: int = 9) -> dict:
    """Compute a histogram of triangle minimum angles (2D only currently).

    Returns
    -------
    dict
        ``{'min_deg', 'mean_deg', 'max_deg', 'bins_deg', 'counts',
        'n_triangles', 'n_slivers'}`` where ``n_slivers`` counts triangles
        with min angle < 20 degrees.
    """
    if dim != 2:
        raise NotImplementedError("mesh_quality_histogram only implemented for dim=2")

    angles_deg: list[float] = []
    for v0, v1, v2 in iter_triangles_2d(HC):
        angles_deg.append(math.degrees(triangle_min_angle(v0, v1, v2)))

    if not angles_deg:
        return {
            "min_deg": 0.0,
            "mean_deg": 0.0,
            "max_deg": 0.0,
            "bins_deg": np.linspace(0, 60, bins + 1).tolist(),
            "counts": [0] * bins,
            "n_triangles": 0,
            "n_slivers": 0,
        }

    arr = np.array(angles_deg)
    edges = np.linspace(0.0, 60.0, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    return {
        "min_deg": float(arr.min()),
        "mean_deg": float(arr.mean()),
        "max_deg": float(arr.max()),
        "bins_deg": edges.tolist(),
        "counts": counts.tolist(),
        "n_triangles": int(arr.size),
        "n_slivers": int((arr < 20.0).sum()),
    }
