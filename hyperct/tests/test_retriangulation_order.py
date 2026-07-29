"""Input-order invariance of ``connect_and_cache_simplices`` (3D).

qhull tie-breaking on degenerate (cospherical / cocircular) point sets
depends on the input ORDER, so the same point SET triangulated from two
different vertex orderings can yield two different (both valid)
Delaunay triangulations.  Since 2026-07-29 the 3D path of
:func:`hyperct.ddg.connect_and_cache_simplices` canonicalizes the qhull
input by a lexicographic coordinate sort, making the triangulation a
function of the point set only — this eliminated the order-dependent
retopology settle-step artifact in the ddgclib 3D static-droplet floor
(plateau 7.616854e-5 -> 7.274172e-5; see
``NOTE(laneA-canonical-order)`` in ``ddg/_retriangulation.py``).

2D is intentionally NOT canonicalized: downstream pinned 2D baselines
must stay bit-identical.
"""
import numpy as np
import pytest

from hyperct import Complex

try:
    from scipy.spatial import ConvexHull
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

pytestmark = pytest.mark.skipif(not HAVE_SCIPY, reason="scipy required")


def _cospherical_cloud():
    """26 points on the unit sphere (deliberately cospherical —
    degenerate for qhull) + the centre + 4 jittered interior points."""
    pts = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                if x == y == z == 0.0:
                    continue
                v = np.array([x, y, z])
                pts.append(v / np.linalg.norm(v))
    pts.append(np.zeros(3))
    rng = np.random.default_rng(7)
    for _ in range(4):
        pts.append(rng.uniform(-0.4, 0.4, size=3))
    return np.array(pts)


def _simplex_key_set(coords, order):
    """Triangulate ``coords[order]`` (vertices created in that order)
    and return the cached simplices as coordinate-tuple frozensets."""
    from hyperct.ddg import connect_and_cache_simplices

    HC = Complex(3)
    verts = []
    for i in order:
        verts.append(HC.V[tuple(coords[i])])
    connect_and_cache_simplices(HC, verts, 3, coords=coords[order])
    return {frozenset(v.x for v in s) for s in HC._simplices}


class TestCanonicalOrder3D:
    def test_3d_cospherical_input_order_invariant(self):
        """Any input permutation yields the identical simplex set."""
        coords = _cospherical_cloud()
        n = len(coords)
        base = _simplex_key_set(coords, np.arange(n))
        rng = np.random.default_rng(0)
        for _ in range(3):
            perm = rng.permutation(n)
            assert _simplex_key_set(coords, perm) == base

    def test_3d_index_remap_tiles_convex_hull(self):
        """The order->original index remap keeps vertex correspondence:
        the cached tets tile the convex hull volume exactly."""
        from hyperct.ddg import connect_and_cache_simplices

        coords = _cospherical_cloud()
        rng = np.random.default_rng(1)
        order = rng.permutation(len(coords))
        HC = Complex(3)
        verts = [HC.V[tuple(coords[i])] for i in order]
        connect_and_cache_simplices(HC, verts, 3, coords=coords[order])
        vol = 0.0
        for s in HC._simplices:
            p = np.array([v.x_a for v in s])
            vol += abs(np.linalg.det(p[1:] - p[0])) / 6.0
        hull = ConvexHull(coords)
        np.testing.assert_allclose(vol, hull.volume, rtol=1e-12)
