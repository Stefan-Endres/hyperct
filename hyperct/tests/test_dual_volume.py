"""
Unit tests for exact barycentric dual volumes (hyperct.ddg._dual_volume).

Vol_i = (1/(dim+1)) * sum_{T ∋ i} |T| over incident top simplices —
exact to machine precision and tiles the domain (partition of unity),
including boundary/corner cells.

Run with:
    pytest hyperct/tests/test_dual_volume.py
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import Delaunay

from hyperct._complex import Complex
from hyperct.ddg import (
    compute_vd,
    dual_cell_area_2d,
    simplex_dual_volumes,
    vertex_dual_volume,
)


def _build_delaunay_mesh(dim, n_refine=2, jitter=0.0, seed=42):
    """Unit-cube mesh re-Delaunay'd with HC._simplices cached.

    Mirrors the ``_retopologize`` pattern (and
    ``test_stress.py::test_p_ij_linear_precision_jittered_3d``):
    structured triangulate + refine, optional interior jitter, global
    disconnect, scipy Delaunay, edge connect, simplex cache.
    """
    HC = Complex(dim, domain=[(0.0, 1.0)] * dim)
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = set()
    for v in HC.V:
        v.boundary = any(
            abs(v.x_a[d]) < 1e-14 or abs(v.x_a[d] - 1.0) < 1e-14
            for d in range(dim)
        )
        if v.boundary:
            bV.add(v)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        for v in list(HC.V):
            if v not in bV and v.nn:
                el = min(np.linalg.norm(v.x_a - vn.x_a) for vn in v.nn)
                off = rng.uniform(-jitter * el, jitter * el, size=dim)
                HC.V.move(v, tuple(v.x_a[d] + off[d] for d in range(dim)))

    verts = list(HC.V)
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)
    coords = np.array([v.x_a[:dim] for v in verts])
    tri = Delaunay(coords)
    for s in tri.simplices:
        for i in range(dim + 1):
            for j in range(i + 1, dim + 1):
                verts[s[i]].connect(verts[s[j]])
    HC._simplices = [
        tuple(verts[s[i]] for i in range(dim + 1)) for s in tri.simplices
    ]
    compute_vd(HC, method='barycentric', cdist=1e-10)
    return HC, bV


class TestPartitionOfUnity:
    """sum_i Vol_i must equal the domain volume to machine precision."""

    @pytest.mark.parametrize("jitter", [0.0, 0.05])
    def test_2d(self, jitter):
        HC, _ = _build_delaunay_mesh(2, n_refine=3, jitter=jitter)
        vols = simplex_dual_volumes(HC, 2)
        total = sum(vols.values())
        npt.assert_allclose(total, 1.0, rtol=1e-12)

    @pytest.mark.parametrize("jitter", [0.0, 0.05])
    def test_3d(self, jitter):
        HC, _ = _build_delaunay_mesh(3, n_refine=2, jitter=jitter)
        vols = simplex_dual_volumes(HC, 3)
        total = sum(vols.values())
        npt.assert_allclose(total, 1.0, rtol=1e-12)

    def test_matches_simplex_total(self):
        """Total equals the summed simplex volumes (self-consistency)."""
        HC, _ = _build_delaunay_mesh(3, n_refine=1, jitter=0.05)
        vols = simplex_dual_volumes(HC, 3)
        tet_total = 0.0
        for s in HC._simplices:
            pts = np.array([v.x_a[:3] for v in s])
            tet_total += abs(np.linalg.det(pts[1:] - pts[0])) / 6.0
        npt.assert_allclose(sum(vols.values()), tet_total, rtol=1e-13)


class TestAgreementWithGeometricPath:
    """Exact rule must match the legacy geometric reconstruction where
    the latter is known-exact: 2D interior dual cell areas."""

    def test_2d_interior_structured(self):
        HC, bV = _build_delaunay_mesh(2, n_refine=3, jitter=0.0)
        vols = simplex_dual_volumes(HC, 2)
        n_checked = 0
        for v in HC.V:
            if v in bV:
                continue
            area_geom = dual_cell_area_2d(v, include_edge_midpoints=True)
            npt.assert_allclose(
                vols[v], area_geom, rtol=1e-12,
                err_msg=f"1/3-rule vs dual_cell_area_2d mismatch at {v.x}",
            )
            n_checked += 1
        assert n_checked > 0

    def test_2d_interior_jittered(self):
        HC, bV = _build_delaunay_mesh(2, n_refine=3, jitter=0.05)
        vols = simplex_dual_volumes(HC, 2)
        for v in HC.V:
            if v in bV:
                continue
            area_geom = dual_cell_area_2d(v, include_edge_midpoints=True)
            npt.assert_allclose(vols[v], area_geom, rtol=1e-11)


class TestVertexDualVolume:

    def test_matches_batch_2d(self):
        HC, _ = _build_delaunay_mesh(2, n_refine=2, jitter=0.05)
        vols = simplex_dual_volumes(HC, 2)
        for v in HC.V:
            npt.assert_allclose(
                vertex_dual_volume(HC, v, 2), vols[v], rtol=1e-13,
            )

    def test_matches_batch_3d(self):
        HC, _ = _build_delaunay_mesh(3, n_refine=1, jitter=0.05)
        vols = simplex_dual_volumes(HC, 3)
        for v in HC.V:
            npt.assert_allclose(
                vertex_dual_volume(HC, v, 3), vols[v], rtol=1e-13,
            )

    def test_boundary_and_corner_included(self):
        """Boundary/corner vertices get their true (nonzero) share."""
        HC, bV = _build_delaunay_mesh(2, n_refine=2, jitter=0.0)
        assert len(bV) > 0
        for v in bV:
            assert vertex_dual_volume(HC, v, 2) > 0.0


class TestNoSimplexCache:

    def test_raises_value_error(self):
        HC = Complex(2, domain=[(0.0, 1.0)] * 2)
        HC.triangulate()
        assert getattr(HC, '_simplices', None) is None
        with pytest.raises(ValueError, match="_simplices"):
            simplex_dual_volumes(HC, 2)
        v = next(iter(HC.V))
        with pytest.raises(ValueError, match="_simplices"):
            vertex_dual_volume(HC, v, 2)


# ---------------------------------------------------------------------------
# Boundary half cells of the geometric 2D path, and the simplex cache built
# from existing connectivity (ddgclib laneS, 2026-10-01)
# ---------------------------------------------------------------------------

def _build_structured_mesh(dim, n_refine, domain=None, jitter=0.0, seed=7):
    """Structured mesh as hyperct builds it: no Delaunay, no simplex cache.

    With ``jitter`` every vertex except the box corners is displaced,
    boundary vertices included, so the boundary is no longer straight.
    """
    domain = domain or [(0.0, 1.0)] * dim
    HC = Complex(dim, domain=domain)
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()
    on_face = {
        v: [abs(v.x_a[d] - domain[d][0]) < 1e-14
            or abs(v.x_a[d] - domain[d][1]) < 1e-14 for d in range(dim)]
        for v in HC.V
    }
    for v in HC.V:
        v.boundary = any(on_face[v])
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        for v in sorted(HC.V, key=lambda w: w.x):
            if all(on_face[v]):
                continue
            el = min(np.linalg.norm(v.x_a - vn.x_a) for vn in v.nn)
            off = rng.uniform(-jitter * el, jitter * el, size=dim)
            HC.V.move(v, tuple(v.x_a[d] + off[d] for d in range(dim)))
    return HC


def _simplex_rule_2d(HC):
    """Vol_i = sum |T| / 3 from the triangles of the 1-skeleton, computed
    without touching ``HC._simplices``."""
    from hyperct.ddg import rebuild_simplex_cache_2d
    saved = HC._simplices
    rebuild_simplex_cache_2d(HC)
    vols = simplex_dual_volumes(HC, 2)
    HC._simplices = saved
    return vols


class TestDualCellArea2dBoundary:
    """``dual_cell_area_2d`` must give the half cell of a boundary vertex,
    closed through the vertex itself.  Without the vertex the cell of a
    right-angle corner is 4x too small (rectangle total 0.96875 at
    refinement 2) and a kinked boundary vertex loses most of its cell."""

    @pytest.mark.parametrize("with_cache", [False, True])
    def test_rectangle_with_corners_sums_to_exact_area(self, with_cache):
        HC = _build_structured_mesh(2, 3, domain=[(0.0, 2.0), (0.0, 1.0)])
        if with_cache:
            from hyperct.ddg import rebuild_simplex_cache_2d
            rebuild_simplex_cache_2d(HC)
        compute_vd(HC, method='barycentric', cdist=1e-10)
        areas = {v: dual_cell_area_2d(v, include_edge_midpoints=True)
                 for v in HC.V}
        npt.assert_allclose(sum(areas.values()), 2.0, rtol=1e-12)
        exact = _simplex_rule_2d(HC)
        corners = [v for v in HC.V
                   if v.x_a[0] in (0.0, 2.0) and v.x_a[1] in (0.0, 1.0)]
        assert len(corners) == 4
        for v in corners:
            npt.assert_allclose(areas[v], exact[v], rtol=1e-12)

    @pytest.mark.parametrize("with_cache", [False, True])
    def test_kinked_boundary_matches_simplex_rule(self, with_cache):
        HC = _build_structured_mesh(2, 3, jitter=0.1)
        if with_cache:
            from hyperct.ddg import rebuild_simplex_cache_2d
            rebuild_simplex_cache_2d(HC)
        compute_vd(HC, method='barycentric', cdist=1e-10)
        exact = _simplex_rule_2d(HC)
        areas = {v: dual_cell_area_2d(v, include_edge_midpoints=True)
                 for v in HC.V}
        n_boundary = 0
        for v in HC.V:
            if not v.boundary:
                continue
            n_boundary += 1
            npt.assert_allclose(areas[v], exact[v], rtol=1e-11,
                                err_msg=f"boundary vertex {v.x}")
        assert n_boundary == 32
        npt.assert_allclose(sum(areas.values()), sum(exact.values()),
                            rtol=1e-12)

    def test_free_surface_vertex_feels_its_own_motion(self):
        """Moving a boundary vertex along its outward normal by ds grows
        its own cell by ds * (adjacent boundary edge lengths) / 3 (laneK:
        the polygon without the vertex credited it with a quarter of
        this, which made a free surface unstable)."""
        ds = 1e-3

        def area_of_top_vertex(shift):
            HC = _build_structured_mesh(2, 3)
            top = HC.V[(0.5, 1.0)]
            if shift:
                HC.V.move(top, (0.5, 1.0 + shift))
            compute_vd(HC, method='barycentric', cdist=1e-10)
            return dual_cell_area_2d(top, include_edge_midpoints=True)

        gain = area_of_top_vertex(ds) - area_of_top_vertex(0.0)
        # two boundary edges of length 1/8 -> the two boundary triangles
        # (and the triangles between them) gain ds * (1/8 + 1/8) / 2,
        # a third of which belongs to the moved vertex
        npt.assert_allclose(gain, ds * 0.125 / 3.0, rtol=1e-9)


class TestRebuildSimplexCache:
    """Simplex cache enumerated from the existing connectivity."""

    def test_2d_enumeration_follows_vertex_cache_order(self):
        from hyperct.ddg import rebuild_simplex_cache_2d
        HC = _build_structured_mesh(2, 2)
        n = rebuild_simplex_cache_2d(HC)
        rank = {v: i for i, v in enumerate(HC.V)}
        ranks = [tuple(rank[v] for v in s) for s in HC._simplices]
        assert n == len(ranks) == 64
        assert all(r[0] < r[1] < r[2] for r in ranks)
        assert ranks == sorted(ranks)

    @pytest.mark.parametrize("n_refine", [1, 2])
    def test_3d_structured_cube(self, n_refine):
        from hyperct.ddg import rebuild_simplex_cache_3d
        HC = _build_structured_mesh(
            3, n_refine, domain=[(0.0, 2.0), (0.0, 1.0), (0.0, 1.0)])
        edges_before = {frozenset((id(v), id(nb)))
                        for v in HC.V for nb in v.nn}
        n = rebuild_simplex_cache_3d(HC)
        assert n == len(HC._simplices) > 0
        # nothing re-triangulated: the cache spans exactly the old edges
        edges_after = {frozenset((id(v), id(nb)))
                       for v in HC.V for nb in v.nn}
        cached_edges = {frozenset((id(a), id(b)))
                        for s in HC._simplices for a in s for b in s
                        if a is not b}
        assert edges_after == edges_before == cached_edges
        vols = simplex_dual_volumes(HC, 3)
        npt.assert_allclose(sum(vols.values()), 2.0, rtol=1e-12)
        assert min(vols.values()) > 0.0
        rank = {v: i for i, v in enumerate(HC.V)}
        ranks = [tuple(rank[v] for v in s) for s in HC._simplices]
        assert ranks == sorted(ranks)

    def test_3d_clique_split_by_interior_vertex_is_not_a_cell(self):
        from hyperct.ddg import rebuild_simplex_cache_3d
        HC = Complex(3, domain=[(0.0, 1.0)] * 3)
        corners = [HC.V[x] for x in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                     (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
        centre = HC.V[(0.25, 0.25, 0.25)]
        verts = corners + [centre]
        for i, a in enumerate(verts):
            for b in verts[i + 1:]:
                a.connect(b)
        assert rebuild_simplex_cache_3d(HC) == 4
        assert all(centre in s for s in HC._simplices)
        vols = simplex_dual_volumes(HC, 3)
        npt.assert_allclose(sum(vols.values()), 1.0 / 6.0, rtol=1e-12)

    def test_3d_non_manifold_cliques_leave_cache_empty(self):
        from hyperct.ddg import rebuild_simplex_cache_3d
        HC = Complex(3, domain=[(-1.0, 1.0)] * 3)
        tri = [HC.V[x] for x in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                 (0.0, 1.0, 0.0))]
        apexes = [HC.V[x] for x in ((0.2, 0.2, 1.0), (0.2, 0.2, -1.0),
                                    (-1.0, -1.0, 0.5))]
        for i, a in enumerate(tri):
            for b in tri[i + 1:]:
                a.connect(b)
        for apex in apexes:
            for a in tri:
                apex.connect(a)
        # three "tetrahedra" on one triangle: not a tetrahedralisation
        assert rebuild_simplex_cache_3d(HC) == 0
        assert HC._simplices is None
