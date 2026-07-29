"""Adaptive remeshing driver.

Runs a sequence of local mesh operations (split, collapse, flip) plus
Laplacian smoothing to maintain mesh quality while preserving sharp
phase interfaces.

Usage
-----
>>> from hyperct.remesh import adaptive_remesh
>>> adaptive_remesh(HC, dim=2, L_min=0.5 * h, L_max=1.4 * h)

The driver does **not** recompute the dual mesh — the caller is
responsible for calling ``hyperct.ddg.compute_vd`` afterwards.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from hyperct.remesh._interface import (
    can_collapse,
    can_flip,
    is_interface_edge,
    is_interface_vertex,
    split_preserves_phase_topology,
)
from hyperct.remesh._operations_2d import (
    edge_collapse_2d,
    edge_flip_2d,
    edge_split_2d,
)
from hyperct.remesh._quality import (
    edge_length,
    iter_triangles_2d,
    triangle_min_angle,
)


def _edge_list(HC) -> list[tuple]:
    """Return a list of unique edges as ``(v_i, v_j)`` pairs, ordered by
    ``id(v)`` so that each edge appears exactly once."""
    edges = []
    seen: set = set()
    for v in HC.V:
        for nb in v.nn:
            key = (id(v), id(nb)) if id(v) < id(nb) else (id(nb), id(v))
            if key in seen:
                continue
            seen.add(key)
            edges.append((v, nb) if id(v) < id(nb) else (nb, v))
    return edges


def _estimate_h_local(HC) -> float:
    """Median edge length over the whole mesh (fallback target spacing)."""
    edges = _edge_list(HC)
    if not edges:
        return 0.0
    lengths = np.array([edge_length(vi, vj) for vi, vj in edges])
    return float(np.median(lengths))


def _edge_h_local(v_i, v_j) -> float:
    """Local length scale of edge ``(v_i, v_j)``: mean length of all
    edges incident to its two endpoints (the shared edge is counted
    once per endpoint).

    This is a cheap O(deg) proxy for the local target spacing that is
    immune to fine/coarse cross-contamination on mixed-resolution
    meshes — a *global* median drags ``L_min``/``L_max`` toward the
    dominant region and triggers unbounded splits in the other one.
    """
    total = 0.0
    n = 0
    for v in (v_i, v_j):
        for nb in v.nn:
            total += edge_length(v, nb)
            n += 1
    if n == 0:
        return 0.0
    return total / n


def _split_long_edges(HC, L_max: Optional[float] = None,
                      preserve_interface: bool = True,
                      alpha_max: Optional[float] = None) -> int:
    """Split every edge longer than the split threshold.  Returns the
    number of splits performed.

    The threshold is the absolute ``L_max`` when given; otherwise the
    per-edge local threshold ``alpha_max * _edge_h_local(v_i, v_j)``.

    When ``preserve_interface`` is True, splits that would introduce
    new cross-phase edges between the midpoint and an opposite vertex
    are rejected via :func:`split_preserves_phase_topology`.  This
    keeps bulk regions of each phase bulk: only interface edges are
    allowed to subdivide the interface, bulk-to-bulk splits are
    allowed only when their 1-ring is single-phase, and mixed
    neighbourhood splits are skipped entirely.
    """
    n_split = 0
    # Snapshot the edge list — splits introduce new edges that we do
    # NOT want to revisit in the same sweep (would explode).
    for v_i, v_j in _edge_list(HC):
        # Make sure the vertices are still live and still connected.
        if v_j not in v_i.nn:
            continue
        L = edge_length(v_i, v_j)
        if L_max is not None:
            if L <= L_max:
                continue
        else:
            if L <= alpha_max * _edge_h_local(v_i, v_j):
                continue
        if preserve_interface and not split_preserves_phase_topology(v_i, v_j):
            continue
        if edge_split_2d(HC, v_i, v_j) is not None:
            n_split += 1
    return n_split


def _collapse_short_edges(HC, L_min: Optional[float] = None,
                          alpha_min: Optional[float] = None) -> int:
    """Collapse every non-constrained edge shorter than the collapse
    threshold (absolute ``L_min`` when given, otherwise the per-edge
    local threshold ``alpha_min * _edge_h_local(v_i, v_j)``).

    Returns the number of collapses performed.
    """
    n_collapse = 0
    # Collapse invalidates incident edges — process one pass and skip
    # any pair whose vertices have been removed/disconnected.
    for v_i, v_j in _edge_list(HC):
        if v_j not in v_i.nn:
            continue
        L = edge_length(v_i, v_j)
        if L_min is not None:
            if L >= L_min:
                continue
        else:
            if L >= alpha_min * _edge_h_local(v_i, v_j):
                continue
        if not can_collapse(v_i, v_j):
            continue
        if edge_collapse_2d(HC, v_i, v_j):
            n_collapse += 1
    return n_collapse


def _flip_for_quality(HC) -> int:
    """Run one Delaunay-quality sweep over all edges.  Returns the
    number of flips performed.
    """
    n_flip = 0
    for v_i, v_j in _edge_list(HC):
        if v_j not in v_i.nn:
            continue
        if not can_flip(v_i, v_j):
            continue
        if edge_flip_2d(HC, v_i, v_j, min_quality_gain=1e-6):
            n_flip += 1
    return n_flip


def _laplacian_smooth(HC, n_iter: int = 1, relax: float = 0.5,
                      skip_boundary: bool = True,
                      skip_interface: bool = False) -> None:
    """Tangential Laplacian smoothing of interior vertex positions.

    For each interior vertex we move it fraction ``relax`` of the way
    toward the centroid of its 1-ring neighbours.  Interface vertices
    are smoothed *along the interface* (averaged over only their
    same-phase interface neighbours) to preserve the interface shape.

    Parameters
    ----------
    n_iter : int
        Number of smoothing sweeps.
    relax : float
        Blending factor in ``[0, 1]``.  0 = no motion, 1 = full jump.
    skip_boundary : bool
        If True, don't move topological boundary vertices.
    skip_interface : bool
        If True, don't move interface vertices at all.  If False, they
        are smoothed only along the interface.
    """
    for _ in range(n_iter):
        updates = {}
        for v in HC.V:
            if skip_boundary and getattr(v, "boundary", False):
                continue

            is_iface = is_interface_vertex(v)
            if is_iface and skip_interface:
                continue

            if is_iface:
                # Tangential smoothing: average only interface neighbours
                iface_nbrs = [
                    nb for nb in v.nn
                    if is_interface_edge(v, nb)
                ]
                if len(iface_nbrs) < 2:
                    continue
                centroid = np.mean(
                    [np.asarray(nb.x_a, dtype=float) for nb in iface_nbrs],
                    axis=0,
                )
            else:
                if not v.nn:
                    continue
                centroid = np.mean(
                    [np.asarray(nb.x_a, dtype=float) for nb in v.nn],
                    axis=0,
                )

            p_old = np.asarray(v.x_a, dtype=float)
            p_new = (1.0 - relax) * p_old + relax * centroid
            updates[v] = tuple(p_new)

        # Apply updates after the pass (avoid chain reactions mid-sweep).
        for v, x_new in updates.items():
            if tuple(v.x) == x_new:
                continue
            # Skip if the new position already exists in the cache.
            if x_new in HC.V.cache and HC.V.cache[x_new] is not v:
                continue
            HC.V.move(v, x_new)


def adaptive_remesh(
    HC,
    dim: int = 2,
    mps=None,
    L_min: Optional[float] = None,
    L_max: Optional[float] = None,
    alpha_min: float = 0.5,
    alpha_max: float = 1.4,
    quality_target_deg: float = 20.0,
    max_iterations: int = 3,
    smooth_iterations: int = 1,
    smooth_relax: float = 0.3,
    preserve_interface: bool = True,
    length_scale: str = "local",
    smooth_skip_interface: bool = False,
) -> dict:
    """Perform interface-preserving adaptive remeshing.

    The driver performs ``max_iterations`` passes of:

    1. Split edges longer than ``L_max``
    2. Collapse edges shorter than ``L_min`` (subject to interface /
       boundary constraints)
    3. Flip edges to improve the minimum-angle quality
    4. Tangential Laplacian smoothing

    until the worst triangle has min angle >= ``quality_target_deg`` or
    no operations were performed in the last pass.

    Parameters
    ----------
    HC : Complex
        2D or 3D simplicial complex.  Only ``dim == 2`` is implemented.
    dim : int
        Spatial dimension (must be 2).
    mps : MultiphaseSystem, optional
        Reserved for future use (per-phase length targets).  Currently
        only used to detect that the mesh has phase labels.
    L_min, L_max : float, optional
        Absolute length thresholds.  If ``None``, relative thresholds
        are used instead: per-edge ``alpha_* * h_local(edge)`` with
        ``length_scale='local'`` (default; ``h_local(edge)`` is the
        mean length of the edges incident to the two endpoints), or
        ``alpha_* * median(edge lengths)`` with
        ``length_scale='global'`` (legacy behaviour).
    alpha_min, alpha_max : float
        Relative thresholds when ``L_min`` / ``L_max`` aren't given.
    quality_target_deg : float
        Minimum-angle quality target in **degrees**.
    max_iterations : int
        Maximum number of split/collapse/flip/smooth sweeps.
    smooth_iterations : int
        Number of Laplacian smoothing sub-iterations per sweep.
    smooth_relax : float
        Smoothing relaxation factor.
    preserve_interface : bool
        If False, disables interface constraints entirely (use with
        care — single-phase meshes are unaffected regardless).
    length_scale : {'local', 'global'}
        How relative thresholds are anchored when ``L_min`` / ``L_max``
        are not given.  ``'local'`` (default) compares each edge to its
        own neighbourhood spacing, so a fine droplet region and a
        coarse outer region adapt independently.  ``'global'`` restores
        the pre-2026-07 behaviour (single mesh-wide median), which on
        mixed-resolution meshes cross-contaminates the thresholds and
        can split the coarse region without bound.  Ignored for any
        threshold that is passed explicitly.
    smooth_skip_interface : bool
        If True, Laplacian smoothing skips interface vertices entirely
        (instead of smoothing them along the interface).  Useful for
        Lagrangian solvers where tangential interface smoothing
        systematically shrinks closed interface loops.  Default False
        (legacy behaviour).

    Returns
    -------
    dict
        Stats: ``{n_splits, n_collapses, n_flips, min_angle_deg,
        iterations, n_triangles}``.
    """
    if dim != 2:
        raise NotImplementedError("adaptive_remesh currently supports dim=2 only")
    if length_scale not in ("local", "global"):
        raise ValueError(
            f"length_scale must be 'local' or 'global', got {length_scale!r}")

    if L_max is None or L_min is None:
        if length_scale == "global":
            # Legacy: absolute thresholds from the mesh-wide median.
            h_local = _estimate_h_local(HC)
            if h_local <= 0.0:
                return {
                    "n_splits": 0, "n_collapses": 0, "n_flips": 0,
                    "min_angle_deg": 0.0, "iterations": 0, "n_triangles": 0,
                }
            if L_max is None:
                L_max = alpha_max * h_local
            if L_min is None:
                L_min = alpha_min * h_local
        elif not _edge_list(HC):
            # 'local' mode on an empty/edge-free mesh: nothing to do
            # (mirrors the legacy h_local <= 0 early return).
            return {
                "n_splits": 0, "n_collapses": 0, "n_flips": 0,
                "min_angle_deg": 0.0, "iterations": 0, "n_triangles": 0,
            }
        # In 'local' mode any threshold left as None is resolved
        # per-edge inside the sweeps: split when
        # L > alpha_max * h_local(edge), collapse when
        # L < alpha_min * h_local(edge).

    # Safety: the collapse threshold must be strictly below the split
    # threshold (otherwise split and collapse fight each other).
    if L_min is not None and L_max is not None and L_min >= L_max:
        L_min = 0.5 * L_max
    if alpha_min >= alpha_max:
        alpha_min = 0.5 * alpha_max

    total_splits = 0
    total_collapses = 0
    total_flips = 0
    q_target_rad = math.radians(quality_target_deg)

    final_min_angle = 0.0
    iters_done = 0
    for it in range(max_iterations):
        iters_done = it + 1
        n_s = _split_long_edges(HC, L_max, preserve_interface=preserve_interface,
                                alpha_max=alpha_max)
        n_c = _collapse_short_edges(HC, L_min, alpha_min=alpha_min)
        n_f = _flip_for_quality(HC)
        _laplacian_smooth(
            HC,
            n_iter=smooth_iterations,
            relax=smooth_relax,
            skip_boundary=True,
            skip_interface=smooth_skip_interface,
        )

        total_splits += n_s
        total_collapses += n_c
        total_flips += n_f

        # Check quality target
        min_angle = math.pi
        n_tri = 0
        for tri in iter_triangles_2d(HC):
            ang = triangle_min_angle(*tri)
            if ang < min_angle:
                min_angle = ang
            n_tri += 1
        final_min_angle = min_angle

        if n_s == 0 and n_c == 0 and n_f == 0:
            break
        if min_angle >= q_target_rad:
            break

    # Final triangle count
    n_triangles = sum(1 for _ in iter_triangles_2d(HC))

    return {
        "n_splits": int(total_splits),
        "n_collapses": int(total_collapses),
        "n_flips": int(total_flips),
        "min_angle_deg": float(math.degrees(final_min_angle)),
        "iterations": int(iters_done),
        "n_triangles": int(n_triangles),
    }
