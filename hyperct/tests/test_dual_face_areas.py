"""Exact barycentric dual face areas from the top-simplex cache
(``hyperct.ddg.simplex_dual_face_areas``, lane Q 2026-10-05).

The dual face of the edge ``(i, j)`` is the sum over the incident top
simplices of ``|T| (grad phi_j - grad phi_i) / (dim + 1)``, the two
barycentric-subdivision triangles (edge midpoint, face barycentre, cell
barycentre) of the edge in each tetrahedron.  Checked here:

- antisymmetry ``A_ij = -A_ji`` to the bit;
- closure ``sum_j A_ij = 0`` at every interior vertex, on a mesh WITH
  flat tetrahedra (qhull's triangulated output on the cube lattice);
- the half cell of a hull vertex closes with its hull facets;
- linear precision ``1/2 sum_j (x_j - x_i) (x) A_ij = Vol_i I`` with the
  exact simplex volumes;
- equality with an independent per-edge construction: the DEC ``p_ij``
  polygon of tet barycentres interleaved with face barycentres, read
  from the link of the edge;
- 2D parity: equal to the segment between the two triangle barycentres
  (or barycentre and edge midpoint on the hull), oriented by the edge.

Run with:
    pytest hyperct/tests/test_dual_face_areas.py
"""
import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex
from hyperct.ddg import (
    connect_and_cache_simplices,
    simplex_dual_face_areas,
    simplex_dual_volumes,
)
from hyperct.ddg._dual_volume import _orientation_signs


def _cloud_mesh(dim, n, seed):
    pts = np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, dim))
    HC = Complex(dim)
    verts = [HC.V[tuple(p)] for p in pts]
    connect_and_cache_simplices(HC, verts, dim, coords=pts)
    return HC


def _lattice_mesh(dim, n_refine):
    """Unit cube lattice re-triangulated by scipy: cospherical points, so
    the triangulated output holds flat simplices in 3D."""
    HC = Complex(dim, domain=[(0.0, 1.0)] * dim)
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()
    verts = list(HC.V)
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)
    coords = np.array([v.x_a[:dim] for v in verts])
    connect_and_cache_simplices(HC, verts, dim, coords=coords)
    return HC


def _tops(HC, dim):
    return [s for s in HC._simplices if len(s) == dim + 1]


def _signed_volumes(HC, dim):
    P = np.array([[w.x_a[:dim] for w in s] for s in _tops(HC, dim)])
    return np.linalg.det(P[:, 1:] - P[:, :1])


def _hull_pieces(HC, dim):
    """``{id(v): sum of |f| n_f / dim}`` over the hull facets at v, outward
    normal from the geometry (meshes without flat hull simplices)."""
    from collections import defaultdict
    tops = _tops(HC, dim)
    owners = defaultdict(list)
    for t, s in enumerate(tops):
        for a in range(dim + 1):
            owners[frozenset(id(w) for b, w in enumerate(s) if b != a)].append((t, a))
    pieces = defaultdict(lambda: np.zeros(dim))
    for f, own in owners.items():
        if len(own) != 1:
            continue
        t, a = own[0]
        others = [w for b, w in enumerate(tops[t]) if b != a]
        x = np.array([w.x_a[:dim] for w in others])
        if dim == 3:
            n = 0.5 * np.cross(x[1] - x[0], x[2] - x[0])
        else:
            e = x[1] - x[0]
            n = np.array([e[1], -e[0]])
        if n @ (x.mean(axis=0) - tops[t][a].x_a[:dim]) < 0:
            n = -n
        for w in others:
            pieces[id(w)] += n / dim
    return pieces


def _pij_polygon(v_i, v_j, HC):
    """Independent reference: the DEC p_ij polygon (tet barycentres
    interleaved with the face barycentres (x_i + x_j + x_k) / 3) read from
    the link of the edge; open chain through the edge midpoint on a hull
    edge.  Oriented by the edge."""
    tets = [s for s in HC._simplices if any(w is v_i for w in s)
            and any(w is v_j for w in s)]
    others = [tuple(w for w in T if w is not v_i and w is not v_j) for T in tets]
    count = {}
    for o in others:
        for w in o:
            count[id(w)] = count.get(id(w), 0) + 1
    ends = [w for w in count if count[w] == 1]
    x_i, x_j = v_i.x_a[:3], v_j.x_a[:3]
    pts = []
    if ends:
        t = next(k for k in range(len(tets)) if any(id(w) == ends[0] for w in others[k]))
        cur = next(w for w in others[t] if id(w) == ends[0])
        pts += [0.5 * (x_i + x_j), (x_i + x_j + cur.x_a[:3]) / 3.0]
    else:
        t, cur = 0, others[0][0]
    used = set()
    while t is not None:
        used.add(t)
        pts.append(np.mean([w.x_a[:3] for w in tets[t]], axis=0))
        cur = next(w for w in others[t] if w is not cur)
        pts.append((x_i + x_j + cur.x_a[:3]) / 3.0)
        t = next((k for k in range(len(tets)) if k not in used
                  and any(w is cur for w in others[k])), None)
    assert len(used) == len(tets)
    P = np.array(pts)
    c = P.mean(axis=0)
    A = 0.5 * np.cross(P - c, np.roll(P, -1, axis=0) - c).sum(axis=0)
    return A if A @ (x_j - x_i) >= 0 else -A


class TestInvariants3D:
    @pytest.fixture(scope='class', params=['cloud', 'lattice'])
    def mesh(self, request):
        if request.param == 'cloud':
            return _cloud_mesh(3, 250, seed=7)
        return _lattice_mesh(3, 2)

    def test_lattice_has_flat_tetrahedra(self):
        det = _signed_volumes(_lattice_mesh(3, 2), 3)
        assert int((det == 0).sum()) >= 10      # 13 measured

    def test_orientation_equals_sign_det_on_valid_simplices(self, mesh):
        tops = _tops(mesh, 3)
        index = {id(v): k for k, v in enumerate(mesh.V)}
        idx = np.array([[index[id(w)] for w in s] for s in tops])
        det = _signed_volumes(mesh, 3)
        s = _orientation_signs(idx, det)
        nz = det != 0
        assert np.all(s[nz] == np.sign(det[nz]))
        assert set(np.unique(s)) <= {-1.0, 1.0}

    def test_antisymmetric(self, mesh):
        A = simplex_dual_face_areas(mesh, 3)
        for i, row in A.items():
            for j, a in row.items():
                assert np.array_equal(A[j][i], -a)

    def test_interior_cells_close_and_are_linearly_precise(self, mesh):
        A = simplex_dual_face_areas(mesh, 3)
        vols = simplex_dual_volumes(mesh, 3)
        hull = _hull_pieces(mesh, 3)
        verts = {id(v): v for v in mesh.V}
        n = 0
        for v in mesh.V:
            if id(v) in hull or id(v) not in A:
                continue
            row = A[id(v)]
            S = np.array(list(row.values()))
            scale = np.linalg.norm(S, axis=1).sum()
            assert np.linalg.norm(S.sum(axis=0)) < 1e-14 * scale
            M = sum(0.5 * np.outer(verts[j].x_a[:3] - v.x_a[:3], a)
                    for j, a in row.items())
            npt.assert_allclose(M, vols[v] * np.eye(3), atol=1e-14 * vols[v],
                                rtol=0.0)
            n += 1
        assert n > 50

    def test_equal_to_the_pij_polygon(self, mesh):
        A = simplex_dual_face_areas(mesh, 3)
        worst = 0.0
        for v in mesh.V:
            for nb in v.nn:
                ref = _pij_polygon(v, nb, mesh)
                worst = max(worst, np.linalg.norm(A[id(v)][id(nb)] - ref)
                            / max(np.linalg.norm(ref), 1e-300))
        assert worst < 1e-13


def _box_hull_pieces(HC):
    """``{id(v): sum of |f| n_f / 3}`` over the hull facets at v of the unit
    cube lattice, outward normal of the box face the facet lies on.  The
    geometric rule of :func:`_hull_pieces` is undefined on a flat hull
    tetrahedron (its fourth vertex lies in the facet plane)."""
    from collections import defaultdict
    tops = _tops(HC, 3)
    owners = defaultdict(list)
    for t, s in enumerate(tops):
        for a in range(4):
            owners[frozenset(id(w) for b, w in enumerate(s) if b != a)].append((t, a))
    pieces = defaultdict(lambda: np.zeros(3))
    for f, own in owners.items():
        if len(own) != 1:
            continue
        t, a = own[0]
        others = [w for b, w in enumerate(tops[t]) if b != a]
        x = np.array([w.x_a[:3] for w in others])
        n = 0.5 * np.cross(x[1] - x[0], x[2] - x[0])
        const = [k for k in range(3) if np.ptp(x[:, k]) == 0.0]
        assert const, 'hull facet of the cube lattice off every box face'
        outward = np.zeros(3)
        outward[const[0]] = 1.0 if x[0, const[0]] == 1.0 else -1.0
        if n @ outward < 0:
            n = -n
        for w in others:
            pieces[id(w)] += n / 3.0
    return pieces


class TestHullHalfCells3D:
    def test_hull_half_cell_closes_with_its_facets(self):
        mesh = _cloud_mesh(3, 250, seed=7)        # no flat hull simplices
        A = simplex_dual_face_areas(mesh, 3)
        hull = _hull_pieces(mesh, 3)
        assert len(hull) > 20
        for v in mesh.V:
            if id(v) not in hull:
                continue
            S = np.array(list(A[id(v)].values()))
            scale = np.linalg.norm(S, axis=1).sum()
            assert np.linalg.norm(S.sum(axis=0) + hull[id(v)]) < 1e-13 * scale

    def test_hull_half_cell_closes_beside_flat_hull_tetrahedra(self):
        """The lattice's flat tetrahedra lie in the box faces, so a hull
        vertex next to one has a dual face piece that no geometric sign
        rule can orient (a per-piece rule such as ``quad . d_ij > 0`` is
        undefined there); the combinatorial orientation still closes the
        half cell against the box-face normals."""
        mesh = _lattice_mesh(3, 2)
        A = simplex_dual_face_areas(mesh, 3)
        hull = _box_hull_pieces(mesh)
        tops = _tops(mesh, 3)
        det = _signed_volumes(mesh, 3)
        flat_touch = {id(w) for t in np.flatnonzero(det == 0) for w in tops[t]}
        n_hull_flat = sum(1 for v in mesh.V
                          if id(v) in hull and id(v) in flat_touch)
        assert n_hull_flat >= 10                  # 36 measured
        for v in mesh.V:
            if id(v) not in hull:
                continue
            S = np.array(list(A[id(v)].values()))
            scale = np.linalg.norm(S, axis=1).sum()
            assert np.linalg.norm(S.sum(axis=0) + hull[id(v)]) < 1e-13 * scale


class TestParity2D:
    def test_matches_the_dual_segment(self):
        mesh = _cloud_mesh(2, 120, seed=3)
        A = simplex_dual_face_areas(mesh, 2)
        vols = simplex_dual_volumes(mesh, 2)
        hull = _hull_pieces(mesh, 2)
        tri_at = {}
        for s in mesh._simplices:
            for w in s:
                tri_at.setdefault(id(w), []).append(s)
        for v in mesh.V:
            for nb in v.nn:
                tris = [s for s in tri_at[id(v)] if any(w is nb for w in s)]
                x_i, x_j = v.x_a[:2], nb.x_a[:2]
                bary = [np.mean([w.x_a[:2] for w in s], axis=0) for s in tris]
                if len(bary) == 1:
                    seg = bary[0] - 0.5 * (x_i + x_j)
                else:
                    seg = bary[1] - bary[0]
                ref = np.array([-seg[1], seg[0]])
                if ref @ (x_j - x_i) < 0:
                    ref = -ref
                npt.assert_allclose(A[id(v)][id(nb)], ref, atol=1e-15, rtol=0)
            S = np.array(list(A[id(v)].values()))
            closing = hull.get(id(v), np.zeros(2))
            assert np.linalg.norm(S.sum(axis=0) + closing) < 1e-14
            if id(v) not in hull:
                M = sum(0.5 * np.outer(w.x_a[:2] - v.x_a[:2], A[id(v)][id(w)])
                        for w in v.nn)
                npt.assert_allclose(M, vols[v] * np.eye(2),
                                    atol=1e-14 * vols[v], rtol=0.0)


class TestErrors:
    def test_requires_the_simplex_cache(self):
        HC = Complex(3)
        for p in np.random.default_rng(1).uniform(size=(8, 3)):
            HC.V[tuple(p)]
        HC._simplices = None
        with pytest.raises(ValueError, match='_simplices'):
            simplex_dual_face_areas(HC, 3)

    def test_dimension(self):
        mesh = _cloud_mesh(2, 30, seed=5)
        with pytest.raises(NotImplementedError):
            simplex_dual_face_areas(mesh, 4)
