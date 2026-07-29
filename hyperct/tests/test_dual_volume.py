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
