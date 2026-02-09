"""Tests for star-based refinement, edge splitting, and boundary methods."""
import pytest
import numpy
from hyperct._complex import Complex


class TestSplitEdge:
    """Tests for Complex.split_edge() — splits an edge by creating a midpoint."""

    def test_split_creates_midpoint(self):
        """Splitting edge (0,0)-(1,1) creates vertex at (0.5,0.5)."""
        HC = Complex(2)
        HC.triangulate()
        vc = HC.split_edge((0.0, 0.0), (1.0, 1.0))
        assert vc.x == (0.5, 0.5)

    def test_split_disconnects_original_edge(self):
        """After split, original vertices are no longer directly connected."""
        HC = Complex(2)
        HC.triangulate()
        v1 = (0.0, 0.0)
        v2 = (0.5, 0.5)  # centroid — connected to all corners
        # Verify the edge exists before splitting
        assert HC.V[v2] in HC.V[v1].nn
        HC.split_edge(v1, v2)
        assert HC.V[v2] not in HC.V[v1].nn
        assert HC.V[v1] not in HC.V[v2].nn

    def test_split_connects_midpoint_to_endpoints(self):
        """Midpoint vertex is connected to both original endpoints."""
        HC = Complex(2)
        HC.triangulate()
        vc = HC.split_edge((0.0, 0.0), (1.0, 1.0))
        assert HC.V[(0.0, 0.0)] in vc.nn
        assert HC.V[(1.0, 1.0)] in vc.nn

    def test_split_increases_vertex_count_by_one(self):
        """Splitting one edge adds exactly one vertex to the cache."""
        HC = Complex(2)
        HC.triangulate()
        initial = len(HC.V.cache)
        HC.split_edge((0.0, 0.0), (1.0, 0.0))
        assert len(HC.V.cache) == initial + 1

    def test_split_3d_edge(self):
        """split_edge works correctly in 3D."""
        HC = Complex(3)
        HC.triangulate()
        vc = HC.split_edge((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert vc.x == (0.5, 0.5, 0.5)
        assert HC.V[(0.0, 0.0, 0.0)] in vc.nn
        assert HC.V[(1.0, 1.0, 1.0)] in vc.nn

    def test_split_midpoint_coordinates(self):
        """Midpoint is at exact arithmetic mean of endpoints."""
        HC = Complex(2, domain=[(0.0, 4.0), (0.0, 6.0)])
        HC.triangulate()
        vc = HC.split_edge((0.0, 0.0), (4.0, 6.0))
        assert vc.x == (2.0, 3.0)


class TestStar:
    """Tests for Complex.st() — returns star domain of a vertex."""

    def test_star_contains_vertex_itself(self):
        """Star domain of v should contain v."""
        HC = Complex(2)
        HC.triangulate()
        v_x = (0.5, 0.5)
        st = HC.st(v_x)
        assert HC.V[v_x] in st

    def test_star_contains_all_neighbours(self):
        """Star domain of v should contain all neighbours of v."""
        HC = Complex(2)
        HC.triangulate()
        v_x = (0.5, 0.5)
        st = HC.st(v_x)
        for vn in HC.V[v_x].nn:
            assert vn in st

    def test_star_size_2d_centroid(self):
        """Initial 2D unit cube centroid has 4 neighbours, star = 5."""
        HC = Complex(2)
        HC.triangulate()
        # After triangulate, centroid (0.5,0.5) connected to 4 corners
        st = HC.st((0.5, 0.5))
        assert len(st) >= 5  # centroid + 4 corners


class TestRefineStar:
    """Tests for Complex.refine_star() — refines star domain of a vertex."""

    def test_refine_star_increases_vertex_count(self):
        """Star refinement should add new midpoint vertices."""
        HC = Complex(2)
        HC.triangulate()
        initial = len(HC.V.cache)
        v = HC.V[(0.5, 0.5)]
        HC.refine_star(v)
        assert len(HC.V.cache) > initial

    def test_refine_star_midpoints_connected(self):
        """All vertices should remain connected after star refinement."""
        HC = Complex(2)
        HC.triangulate()
        v = HC.V[(0.5, 0.5)]
        HC.refine_star(v)
        for vk in HC.V.cache:
            assert len(HC.V[vk].nn) >= 1, (
                f"Vertex {vk} has no connections after star refinement"
            )

    def test_refine_star_preserves_existing_vertices(self):
        """Original vertices should still exist after star refinement."""
        HC = Complex(2)
        HC.triangulate()
        original_coords = set(HC.V.cache.keys())
        v = HC.V[(0.5, 0.5)]
        HC.refine_star(v)
        for coord in original_coords:
            assert coord in HC.V.cache

    def test_refine_star_corner_vertex(self):
        """Star refinement works on a corner vertex (fewer neighbours)."""
        HC = Complex(2)
        HC.triangulate()
        initial = len(HC.V.cache)
        v = HC.V[(0.0, 0.0)]
        HC.refine_star(v)
        assert len(HC.V.cache) > initial

    def test_refine_star_3d(self):
        """Star refinement works in 3D."""
        HC = Complex(3)
        HC.triangulate()
        initial = len(HC.V.cache)
        v = HC.V[(0.5, 0.5, 0.5)]
        HC.refine_star(v)
        assert len(HC.V.cache) > initial


class TestRefineAllStar:
    """Tests for Complex.refine_all_star() — refines all vertex star domains."""

    def test_refine_all_star_increases_vertex_count(self):
        """Refining all stars should add many new vertices."""
        HC = Complex(2)
        HC.triangulate()
        initial = len(HC.V.cache)
        HC.refine_all_star()
        assert len(HC.V.cache) > initial

    def test_refine_all_star_3d(self):
        """refine_all_star works in 3D."""
        HC = Complex(3)
        HC.triangulate()
        initial = len(HC.V.cache)
        HC.refine_all_star()
        assert len(HC.V.cache) > initial

    def test_refine_all_star_connectivity(self):
        """All vertices should have connections after refine_all_star."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all_star()
        for vk in HC.V.cache:
            assert len(HC.V[vk].nn) >= 1


class TestBoundary:
    """Tests for Complex.boundary() — computes boundary of vertex set."""

    def test_boundary_2d_contains_corners(self):
        """In a 2D unit cube, all 4 corner vertices should be on boundary."""
        HC = Complex(2)
        HC.triangulate()
        dV = HC.boundary()
        boundary_coords = {v.x for v in dV}
        for corner in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            assert corner in boundary_coords, (
                f"Corner {corner} not found in boundary"
            )

    def test_boundary_excludes_interior_after_refine(self):
        """Interior vertices should not be on boundary after refinement."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        dV = HC.boundary()
        boundary_coords = {v.x for v in dV}
        # Centroid (0.5, 0.5) should be interior
        assert (0.5, 0.5) not in boundary_coords

    def test_boundary_with_custom_vertex_set(self):
        """boundary(V) computes boundary of given vertex subset."""
        HC = Complex(2)
        HC.triangulate()
        # Pass entire vertex cache
        dV = HC.boundary(V=HC.V)
        assert len(dV) > 0

    def test_boundary_3d_contains_corners(self):
        """3D unit cube corners should be on boundary."""
        HC = Complex(3)
        HC.triangulate()
        dV = HC.boundary()
        boundary_coords = {v.x for v in dV}
        assert (0.0, 0.0, 0.0) in boundary_coords
        assert (1.0, 1.0, 1.0) in boundary_coords
