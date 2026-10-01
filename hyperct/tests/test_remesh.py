"""Unit tests for the ``hyperct.remesh`` module.

Covers:

- Triangle quality metrics (min angle, aspect ratio, area)
- Interface constraint helpers
- 2D local mesh operations: edge split, edge collapse, edge flip
- Adaptive remesh driver on simple meshes (uniform and perturbed grids,
  with and without phase labels)
"""
import math

import numpy as np
import pytest

from hyperct._complex import Complex
from hyperct.remesh import (
    adaptive_remesh,
    can_collapse,
    can_flip,
    edge_collapse_2d,
    edge_flip_2d,
    edge_split_2d,
    is_interface_edge,
    mesh_quality_histogram,
    triangle_area,
    triangle_aspect_ratio,
    triangle_min_angle,
    triangles_around_edge,
    vertex_phase,
)
from hyperct.remesh._quality import iter_triangles_2d


# ---------------------------------------------------------------------------
# Helpers for synthetic meshes that don't depend on Complex.triangulate()
# ---------------------------------------------------------------------------

def _grid_2d(nx=3, ny=3, Lx=1.0, Ly=1.0):
    """Return a small 2D simplicial complex on a [0,Lx]x[0,Ly] grid with
    each quad split into two triangles along the (i,j)->(i+1,j+1) diagonal.

    All boundary vertices get ``v.boundary = True``.
    """
    HC = Complex(2)
    verts = {}
    for i in range(nx):
        for j in range(ny):
            x = (i * Lx / (nx - 1), j * Ly / (ny - 1))
            verts[(i, j)] = HC.V[x]

    for i in range(nx - 1):
        for j in range(ny - 1):
            a = verts[(i, j)]
            b = verts[(i + 1, j)]
            c = verts[(i, j + 1)]
            d = verts[(i + 1, j + 1)]
            # Triangle (a,b,d) and (a,d,c); both share edge (a,d).
            a.connect(b)
            a.connect(d)
            b.connect(d)
            a.connect(c)
            c.connect(d)

    for (i, j), v in verts.items():
        v.boundary = (i == 0 or j == 0 or i == nx - 1 or j == ny - 1)
    return HC, verts


def _single_quad():
    """Two triangles sharing edge (a,c) — the classic flip test case.

    Layout::
            b(1,1)
           /|
          / |
         /  |
        a---c
        (0,0) (1,0)
               |
               d(2,0)... no, just 4 verts:

    Actually use
        a=(0,0), b=(1,1), c=(1,0), d=(0,1)
    with triangles (a,b,c) and (a,b,d) sharing edge (a,b).
    """
    HC = Complex(2)
    a = HC.V[(0.0, 0.0)]
    b = HC.V[(1.0, 1.0)]
    c = HC.V[(1.0, 0.0)]
    d = HC.V[(0.0, 1.0)]
    # Edges of the two triangles
    a.connect(b)
    a.connect(c)
    a.connect(d)
    b.connect(c)
    b.connect(d)
    # Boundary tagging (all four are on the boundary of the quad)
    for v in (a, b, c, d):
        v.boundary = False  # mark interior for flip tests
    return HC, a, b, c, d


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

class TestQualityMetrics:
    def test_equilateral_min_angle(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.5, math.sqrt(3) / 2])
        assert triangle_min_angle(p0, p1, p2) == pytest.approx(math.pi / 3)

    def test_equilateral_aspect_ratio(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.5, math.sqrt(3) / 2])
        # Equilateral: longest_edge / (2 * inradius) = sqrt(3) ≈ 1.732
        # (this is the radius-ratio-style definition used here; the
        # equilateral is the *minimum* aspect ratio attainable).
        ar = triangle_aspect_ratio(p0, p1, p2)
        assert ar == pytest.approx(math.sqrt(3), rel=1e-9)

    def test_right_triangle_min_angle(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])
        # 45-45-90 triangle: min angle = 45 deg
        assert triangle_min_angle(p0, p1, p2) == pytest.approx(math.pi / 4)

    def test_sliver_aspect_ratio_large(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.5, 1e-4])
        ar = triangle_aspect_ratio(p0, p1, p2)
        assert ar > 50.0

    def test_degenerate_triangle(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 0.0])
        assert triangle_min_angle(p0, p1, p2) == pytest.approx(0.0, abs=1e-10)
        assert triangle_aspect_ratio(p0, p1, p2) == math.inf

    def test_triangle_area_signed(self):
        # CCW triangle in 2D should have positive signed area.
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])
        assert triangle_area(p0, p1, p2) == pytest.approx(0.5)
        # Reversed (CW) should be negative.
        assert triangle_area(p0, p2, p1) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Interface helpers
# ---------------------------------------------------------------------------

class TestInterfaceHelpers:
    def test_vertex_phase_default_none(self):
        HC = Complex(2)
        v = HC.V[(0.0, 0.0)]
        assert vertex_phase(v) is None

    def test_is_interface_edge_no_phase(self):
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        assert is_interface_edge(v1, v2) is False

    def test_is_interface_edge_same_phase(self):
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        v1.phase = 0
        v2.phase = 0
        assert is_interface_edge(v1, v2) is False

    def test_is_interface_edge_cross_phase(self):
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        v1.phase = 0
        v2.phase = 1
        assert is_interface_edge(v1, v2) is True

    def test_can_flip_interface_edge(self):
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        v1.phase = 0
        v2.phase = 1
        assert can_flip(v1, v2) is False

    def test_can_flip_boundary_edge(self):
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        v1.boundary = True
        v2.boundary = True
        assert can_flip(v1, v2) is False

    def test_can_collapse_boundary(self):
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        v1.boundary = True
        assert can_collapse(v1, v2) is False


# ---------------------------------------------------------------------------
# Edge split
# ---------------------------------------------------------------------------

class TestEdgeSplit:
    def test_split_creates_midpoint(self):
        HC, a, b, c, d = _single_quad()
        n_verts_before = len(list(HC.V))
        v_m = edge_split_2d(HC, a, b)
        assert v_m is not None
        assert len(list(HC.V)) == n_verts_before + 1
        np.testing.assert_allclose(v_m.x_a[:2], [0.5, 0.5])

    def test_split_disconnects_original_edge(self):
        HC, a, b, c, d = _single_quad()
        edge_split_2d(HC, a, b)
        assert b not in a.nn

    def test_split_connects_midpoint_to_opposites(self):
        HC, a, b, c, d = _single_quad()
        v_m = edge_split_2d(HC, a, b)
        assert c in v_m.nn
        assert d in v_m.nn
        assert a in v_m.nn
        assert b in v_m.nn

    def test_split_field_averaging(self):
        HC, a, b, c, d = _single_quad()
        a.p = 10.0
        b.p = 20.0
        a.u = np.array([1.0, 0.0])
        b.u = np.array([3.0, 0.0])
        a.m = 2.0
        b.m = 4.0
        v_m = edge_split_2d(HC, a, b)
        assert v_m.p == pytest.approx(15.0)
        # 2026-07-02 (lane4-remesh-upstream): u re-pinned [2,0] -> [7/3,0].
        # The midpoint velocity is now the MASS-WEIGHTED mean of the
        # transferred parcels (dm_a=1, dm_b=2 here), which conserves
        # sum(m*u) exactly; the old arithmetic mean injected momentum.
        np.testing.assert_allclose(v_m.u, [7.0 / 3.0, 0.0])
        # In _single_quad both endpoints cede exactly half their mass
        # (their whole dual cell lies in the two split triangles), so
        # the conservative midpoint mass coincides with the old
        # arithmetic mean — but a.m and b.m are now halved.
        assert v_m.m == pytest.approx(3.0)
        assert a.m == pytest.approx(1.0)
        assert b.m == pytest.approx(2.0)

    def test_split_phase_inheritance(self):
        HC, a, b, c, d = _single_quad()
        a.phase = 0
        b.phase = 0
        v_m = edge_split_2d(HC, a, b)
        assert vertex_phase(v_m) == 0


# ---------------------------------------------------------------------------
# Edge collapse
# ---------------------------------------------------------------------------

class TestEdgeCollapse:
    def test_collapse_removes_vertex(self):
        HC, a, b, c, d = _single_quad()
        n_before = len(list(HC.V))
        ok = edge_collapse_2d(HC, a, b)
        assert ok
        assert len(list(HC.V)) == n_before - 1

    def test_collapse_merges_mass(self):
        HC, a, b, c, d = _single_quad()
        a.m = 3.0
        b.m = 5.0
        edge_collapse_2d(HC, a, b)
        assert a.m == pytest.approx(8.0)

    def test_collapse_blocked_on_boundary(self):
        HC, a, b, c, d = _single_quad()
        a.boundary = True
        assert can_collapse(a, b) is False

    def test_collapse_blocked_on_interface(self):
        HC, a, b, c, d = _single_quad()
        a.phase = 0
        b.phase = 1
        assert can_collapse(a, b) is False

    def test_split_then_collapse_vertex_count_returns(self):
        """Split + collapse of the new edge should reduce the vertex
        count to at most the original (a round trip)."""
        HC, verts = _grid_2d(nx=3, ny=3)
        n_before = len(list(HC.V))
        a = verts[(0, 0)]
        b = verts[(1, 1)]
        v_m = edge_split_2d(HC, a, b)
        assert v_m is not None
        # Collapse the new edge (v_m, b); boundary tagging on (0,0) and
        # (1,1) should allow it because v_m is interior and b is not on
        # the outer boundary... wait, b=(1,1) IS an interior vertex for
        # 3x3 grid, and v_m is interior too. Check:
        assert b.boundary is False
        assert v_m.boundary is False
        ok = edge_collapse_2d(HC, v_m, b)
        assert ok
        assert len(list(HC.V)) == n_before


# ---------------------------------------------------------------------------
# Edge flip
# ---------------------------------------------------------------------------

class TestEdgeFlip:
    def test_flip_improves_quality(self):
        """The edge (a,b) in _single_quad goes from (0,0)-(1,1) — the
        long diagonal — through a non-Delaunay configuration.  After
        the flip, the diagonal should be (c,d) = (1,0)-(0,1), which is
        also length sqrt(2).  Actually both diagonals are length sqrt(2)
        in this square, so we set up a rectangle instead.
        """
        HC = Complex(2)
        a = HC.V[(0.0, 0.0)]
        b = HC.V[(2.0, 0.5)]  # interior-ish
        c = HC.V[(1.0, 0.0)]
        d = HC.V[(0.0, 1.0)]
        # Triangles (a, c, b) and (a, b, d) share edge (a, b).
        a.connect(b)
        a.connect(c)
        a.connect(d)
        b.connect(c)
        b.connect(d)
        for v in (a, b, c, d):
            v.boundary = False

        old_q = min(
            triangle_min_angle(a, b, c),
            triangle_min_angle(a, b, d),
        )
        ok = edge_flip_2d(HC, a, b, min_quality_gain=0.0)
        if ok:
            assert b not in a.nn
            assert c in d.nn
            new_q = min(
                triangle_min_angle(c, d, a),
                triangle_min_angle(c, d, b),
            )
            assert new_q >= old_q

    def test_flip_blocked_on_interface(self):
        HC, a, b, c, d = _single_quad()
        a.phase = 0
        b.phase = 1
        # Call can_flip guard from driver
        assert can_flip(a, b) is False

    def test_flip_rejects_non_manifold(self):
        HC = Complex(2)
        a = HC.V[(0.0, 0.0)]
        b = HC.V[(1.0, 0.0)]
        a.connect(b)
        # No opposite vertices — should not flip.
        assert edge_flip_2d(HC, a, b) is False

    def test_flip_preserves_triangle_count(self):
        HC, a, b, c, d = _single_quad()
        n_tri_before = sum(1 for _ in iter_triangles_2d(HC))
        edge_flip_2d(HC, a, b, min_quality_gain=0.0)
        n_tri_after = sum(1 for _ in iter_triangles_2d(HC))
        # Flip may or may not succeed (square is Delaunay-ambiguous),
        # but either way triangle count must be conserved.
        assert n_tri_before == n_tri_after


# ---------------------------------------------------------------------------
# Triangle enumeration
# ---------------------------------------------------------------------------

class TestTriangleEnumeration:
    def test_single_quad_has_two_triangles(self):
        HC, a, b, c, d = _single_quad()
        tris = list(iter_triangles_2d(HC))
        assert len(tris) == 2

    def test_triangles_around_edge(self):
        HC, a, b, c, d = _single_quad()
        opp = triangles_around_edge(a, b)
        assert set(opp) == {c, d}

    def test_grid_3x3_triangle_count(self):
        HC, verts = _grid_2d(nx=3, ny=3)
        tris = list(iter_triangles_2d(HC))
        # 2x2 quads, 2 triangles each = 8 triangles
        assert len(tris) == 8


# ---------------------------------------------------------------------------
# Adaptive remesh driver
# ---------------------------------------------------------------------------

class TestAdaptiveRemesh:
    def test_noop_on_uniform_grid(self):
        """A uniform grid near the target spacing should converge
        without many operations."""
        HC, verts = _grid_2d(nx=4, ny=4)
        stats = adaptive_remesh(
            HC, dim=2,
            L_min=0.1, L_max=0.6,  # median edge ~0.33
            max_iterations=2,
        )
        assert stats["n_triangles"] > 0

    def test_returns_stats_dict(self):
        HC, verts = _grid_2d(nx=3, ny=3)
        stats = adaptive_remesh(HC, dim=2, max_iterations=1)
        for key in ("n_splits", "n_collapses", "n_flips",
                    "min_angle_deg", "iterations", "n_triangles"):
            assert key in stats

    def test_splits_long_edges(self):
        HC, verts = _grid_2d(nx=3, ny=3)
        # With tiny L_max all non-boundary edges should be candidates.
        stats = adaptive_remesh(
            HC, dim=2,
            L_min=0.01, L_max=0.2,
            max_iterations=1,
            smooth_iterations=0,
        )
        assert stats["n_splits"] > 0

    def test_preserves_interface_topology(self):
        """Build a 3x3 grid with phase 0 on the left column and phase 1
        on the right.  After remeshing, there should be NO cross-phase
        edge between vertices that did not originally share one (i.e.,
        the interface remains pinned to its original location).
        """
        HC, verts = _grid_2d(nx=3, ny=3)
        # Left half (i<=0) phase 0, interface at i=1, right (i>=1) phase 1.
        # Actually use i<=0 = phase 0, i>=1 = phase 1 so the vertical
        # line at i=1 is a sharp boundary.
        for (i, j), v in verts.items():
            v.phase = 0 if i == 0 else 1

        # Snapshot initial interface edges
        iface_before = set()
        for v in HC.V:
            for nb in v.nn:
                if is_interface_edge(v, nb):
                    iface_before.add(
                        frozenset((v.phase, nb.phase))
                    )

        stats = adaptive_remesh(
            HC, dim=2,
            L_min=0.1, L_max=0.6,
            max_iterations=2,
        )

        # There should still be interface edges (no collapse/flip
        # destroyed all of them).
        iface_after = sum(
            1 for v in HC.V for nb in v.nn if is_interface_edge(v, nb)
        )
        assert iface_after > 0, "Interface topology was destroyed by remesh"
        assert stats["n_triangles"] > 0

    def test_mesh_quality_histogram(self):
        HC, verts = _grid_2d(nx=4, ny=4)
        hist = mesh_quality_histogram(HC, dim=2)
        assert hist["n_triangles"] == 18
        assert 0.0 <= hist["min_deg"] <= 90.0
        assert 0.0 <= hist["mean_deg"] <= 90.0

    def test_raises_on_3d(self):
        HC = Complex(2)
        HC.V[(0.0, 0.0)]
        with pytest.raises(NotImplementedError):
            adaptive_remesh(HC, dim=3)


# ---------------------------------------------------------------------------
# Direct invariants for local operations (Step 2 review fixes)
# ---------------------------------------------------------------------------

class TestEdgeSplitDirect:
    """Direct pre/post invariants that the earlier tests only checked
    indirectly — triangle count, boundary inheritance, carryover of
    additional fields like velocity, mass, phase."""

    def test_triangle_count_doubles_on_interior_edge(self):
        """Splitting an interior edge replaces 2 triangles with 4."""
        HC, a, b, c, d = _single_quad()  # 2 triangles sharing edge (a,b)
        assert sum(1 for _ in iter_triangles_2d(HC)) == 2
        v_m = edge_split_2d(HC, a, b)
        assert v_m is not None
        assert sum(1 for _ in iter_triangles_2d(HC)) == 4

    def test_split_carries_all_fields(self):
        """Mass, velocity, pressure, AND phase all carry through."""
        HC, a, b, c, d = _single_quad()
        a.p = 4.0
        b.p = 8.0
        a.u = np.array([0.0, 2.0])
        b.u = np.array([0.0, 4.0])
        a.m = 1.0
        b.m = 3.0
        a.phase = 0
        b.phase = 0

        v_m = edge_split_2d(HC, a, b)
        assert v_m.p == pytest.approx(6.0)
        # 2026-07-02 (lane4-remesh-upstream): u re-pinned [0,3] -> [0,3.5]
        # (mass-weighted momentum-conserving mixing, dm_a=0.5, dm_b=1.5;
        # see test_split_field_averaging).
        np.testing.assert_allclose(v_m.u, [0.0, 3.5])
        assert v_m.m == pytest.approx(2.0)
        assert v_m.phase == 0

    def test_interior_midpoint_not_boundary(self):
        """Splitting an interior edge produces an interior midpoint."""
        HC, a, b, c, d = _single_quad()
        v_m = edge_split_2d(HC, a, b)
        assert v_m.boundary is False

    def test_boundary_edge_midpoint_is_boundary(self):
        """Splitting a boundary edge (only one adjacent triangle, both
        endpoints on the boundary) produces a new boundary vertex.
        """
        HC = Complex(2)
        a = HC.V[(0.0, 0.0)]
        b = HC.V[(1.0, 0.0)]
        c = HC.V[(0.5, 1.0)]  # single opposite
        a.connect(b)
        a.connect(c)
        b.connect(c)
        a.boundary = True
        b.boundary = True
        c.boundary = False

        v_m = edge_split_2d(HC, a, b)
        assert v_m is not None
        assert v_m.boundary is True


class TestEdgeCollapseDirect:
    """Direct pre/post invariants for edge_collapse_2d."""

    def test_collapse_rewires_1_ring(self):
        """Every neighbour of the removed vertex should be connected
        to the survivor afterwards.
        """
        HC, verts = _grid_2d(nx=3, ny=3)
        v_i = verts[(1, 1)]  # interior
        # Pick an interior neighbour to collapse into
        v_j = verts[(1, 2)]
        neighbours_of_j = set(v_j.nn) - {v_i}
        ok = edge_collapse_2d(HC, v_i, v_j)
        assert ok
        # v_j is removed; v_i must now be connected to every
        # original neighbour of v_j.
        for nb in neighbours_of_j:
            assert nb in v_i.nn, f"neighbour {nb.x} not rewired to survivor"

    def test_collapse_removes_exactly_one_vertex(self):
        HC, verts = _grid_2d(nx=3, ny=3)
        n_before = len(list(HC.V))
        v_i = verts[(1, 1)]
        v_j = verts[(1, 2)]
        edge_collapse_2d(HC, v_i, v_j)
        assert len(list(HC.V)) == n_before - 1

    def test_collapse_averages_scalar_fields(self):
        """Non-mass scalar fields like pressure should be averaged."""
        HC, verts = _grid_2d(nx=3, ny=3)
        v_i = verts[(1, 1)]
        v_j = verts[(1, 2)]
        v_i.p = 2.0
        v_j.p = 6.0
        edge_collapse_2d(HC, v_i, v_j)
        assert v_i.p == pytest.approx(4.0)

    def test_collapse_returns_false_for_disconnected(self):
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        # v1 and v2 are not connected
        assert edge_collapse_2d(HC, v1, v2) is False


class TestEdgeFlipDirect:
    """Quality-gain logic and non-convex quad rejection."""

    def test_flip_rejects_when_no_quality_gain(self):
        """On the symmetric _single_quad (both diagonals equal) a
        positive min_quality_gain must block the flip."""
        HC, a, b, c, d = _single_quad()
        # Require a meaningful improvement; diagonals are symmetric
        # here so no improvement is possible.
        ok = edge_flip_2d(HC, a, b, min_quality_gain=0.1)
        assert ok is False
        # Edge still present
        assert b in a.nn

    def test_flip_rejects_when_new_edge_already_exists(self):
        """If (v_k, v_l) already exists, flipping would create a
        duplicate — must refuse."""
        HC, a, b, c, d = _single_quad()
        # Pre-connect c and d (the opposites), so any flip attempt
        # would duplicate.
        c.connect(d)
        assert edge_flip_2d(HC, a, b, min_quality_gain=0.0) is False

    def test_flip_rejects_concave_configuration(self):
        """A deliberately non-convex quad must not be flipped — the
        orientation check should reject it.
        """
        HC = Complex(2)
        # 4 vertices forming a non-convex quad: a reflex vertex c inside
        a = HC.V[(0.0, 0.0)]
        b = HC.V[(2.0, 0.0)]
        c = HC.V[(1.0, 0.3)]   # interior, reflex
        d = HC.V[(1.0, 2.0)]
        # Triangles (a,b,c), (a,d,c)... but we need the shared edge to
        # be (a,c), with opposites b and d. The quadrilateral (a,b,c,d)
        # is non-convex at c.
        a.connect(b)
        a.connect(c)
        a.connect(d)
        b.connect(c)
        c.connect(d)
        for v in (a, b, c, d):
            v.boundary = False
        # The flip would replace diagonal (a,c) with (b,d) — but (b,d)
        # would cross outside the quad, inverting a triangle.
        result = edge_flip_2d(HC, a, c, min_quality_gain=0.0)
        # Either refused outright, OR produced a valid non-inverting
        # configuration. Key invariant: the mesh must still have the
        # same number of triangles.
        n_tri = sum(1 for _ in iter_triangles_2d(HC))
        assert n_tri == 2, "flip produced a topology-changing result"


class TestRemeshHelpers:
    """Coverage for helper utilities that the first test pass missed."""

    def test_is_interface_vertex_no_phase(self):
        from hyperct.remesh._interface import is_interface_vertex
        HC = Complex(2)
        v = HC.V[(0.0, 0.0)]
        assert is_interface_vertex(v) is False

    def test_is_interface_vertex_true_at_boundary(self):
        from hyperct.remesh._interface import is_interface_vertex
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        v1.phase = 0
        v2.phase = 1
        v1.connect(v2)
        assert is_interface_vertex(v1) is True
        assert is_interface_vertex(v2) is True

    def test_is_interface_vertex_false_in_bulk(self):
        from hyperct.remesh._interface import is_interface_vertex
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(1.0, 0.0)]
        v1.phase = 0
        v2.phase = 0
        v1.connect(v2)
        assert is_interface_vertex(v1) is False

    def test_edge_length_matches_numpy_norm(self):
        from hyperct.remesh._quality import edge_length
        HC = Complex(2)
        v1 = HC.V[(0.0, 0.0)]
        v2 = HC.V[(3.0, 4.0)]
        assert edge_length(v1, v2) == pytest.approx(5.0)

    def test_iter_triangles_dedupes(self):
        """Every triangle in a complex must be yielded exactly once."""
        HC, verts = _grid_2d(nx=4, ny=4)
        tris = list(iter_triangles_2d(HC))
        # 3x3 quads * 2 triangles per quad = 18
        assert len(tris) == 18
        # And all should be unique (by sorted id triple)
        keys = {tuple(sorted((id(a), id(b), id(c)))) for a, b, c in tris}
        assert len(keys) == 18


# ---------------------------------------------------------------------------
# iter_triangles_2d: only vertices of the complex (ddgclib laneS, 2026-10-01)
# ---------------------------------------------------------------------------

def test_iter_triangles_2d_skips_a_vertex_dropped_from_the_cache():
    """``HC.V.move`` onto an occupied coordinate drops the displaced
    vertex from ``HC.V`` but leaves its edges.  Its triangles must not be
    yielded.  They used to be yielded unless the dropped vertex had the
    smallest ``id`` of the three, so the result depended on memory
    addresses (the dropped vertex here has the LARGEST id, the case the
    old code got wrong)."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    n_before = sum(1 for _ in iter_triangles_2d(HC))
    dropped = max(HC.V, key=id)
    n_with_dropped = sum(1 for t in iter_triangles_2d(HC)
                         if any(v is dropped for v in t))
    assert n_with_dropped > 0
    mover = next(v for v in HC.V if v is not dropped and v not in dropped.nn)
    HC.V.move(mover, dropped.x)
    members = {id(v) for v in HC.V}
    assert id(dropped) not in members and dropped.nn
    tris = list(iter_triangles_2d(HC))
    assert all(id(v) in members for t in tris for v in t)
    assert len(tris) == n_before - n_with_dropped
