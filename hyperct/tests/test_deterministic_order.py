"""Results do not depend on the memory addresses of the vertex objects.

Until 2026-10-02 three routines ordered vertices by ``id()``:

- ``compute_vd`` (3D simplex-aware path) summed each boundary face
  barycentre in ``id()`` order of the face's vertices.  The last bit of
  the dual vertex position, and with it its hash and every set walk that
  starts from it, differed between interpreters and between two builds in
  one interpreter (ddgclib: 3D runs reproducible to two digits only).
- ``hyperct.remesh._driver._edge_list`` oriented each edge by ``id()``,
  and a collapse keeps the first vertex of the pair.
- ``hyperct.remesh._quality.iter_triangles_2d`` listed each triangle from
  its lowest-``id()`` vertex.

They now follow ``HC.V`` order (or the order of the first simplex that
lists a face).  The tests build the same complex twice with the
small-object heap fragmented in between, so that the address order of the
second build differs from the first.
"""
import random

import numpy as np
import pytest

from hyperct import Complex
from hyperct.ddg import (
    boundary_from_simplices,
    compute_vd,
    connect_and_cache_simplices,
)
from hyperct.remesh import adaptive_remesh
from hyperct.remesh._driver import _edge_list
from hyperct.remesh._quality import iter_triangles_2d


class _Filler:
    """Same allocation size class as a vertex (instance with a dict)."""

    def __init__(self, i):
        self.x = (float(i),)


def _fragment_heap(seed, n=40_000):
    """Allocate ``n`` small objects and free a random subset: objects
    created afterwards fill the holes, at addresses that are not monotone
    in creation order.  Returns the survivors (keep them alive)."""
    rng = random.Random(seed)
    objs = [_Filler(i) for i in range(n)]
    rng.shuffle(objs)
    return objs[rng.randint(n // 4, 3 * n // 4):]


def _cloud(dim, n, seed=11):
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, dim))


def _build(dim, coords):
    HC = Complex(dim)
    verts = [HC.V[tuple(p)] for p in coords]
    connect_and_cache_simplices(HC, verts, dim, coords=coords)
    dV = boundary_from_simplices(HC, dim)
    for v in HC.V:
        v.boundary = v in dV
    return HC


def _address_order(HC):
    ids = [id(v) for v in HC.V]
    return tuple(sorted(range(len(ids)), key=ids.__getitem__))


@pytest.fixture(scope='module')
def builds_3d():
    """The same 3D complex three times, at other addresses each time."""
    coords = _cloud(3, 60)
    out, keep = [], []
    for seed in (0, 1, 2):
        if seed:
            keep.append(_fragment_heap(seed))
        HC = _build(3, coords)
        compute_vd(HC, method='barycentric')
        out.append(HC)
    return out


class TestDualVerticesDoNotDependOnAddresses:
    def test_the_builds_sit_at_other_address_orders(self, builds_3d):
        assert len({_address_order(HC) for HC in builds_3d}) > 1

    def test_neighbour_sets_iterate_in_the_same_order(self, builds_3d):
        """``v.nn`` is a set, but a vertex hashes by its coordinate tuple
        (``VertexBase.__hash__``), not by its address: the iteration
        order is a function of the coordinates and of the order of the
        ``connect`` calls.  (Held before 2026-10-02 as well; everything
        that walks ``v.nn``, ``v.vd`` or their intersections relies on
        it.)"""
        order = [[[w.x for w in v.nn] for v in HC.V] for HC in builds_3d]
        assert order[0] == order[1] == order[2]

    def test_dual_vertex_positions_equal_to_the_bit(self, builds_3d):
        keys = [sorted(vd.x for vd in HC.Vd) for HC in builds_3d]
        assert keys[0] == keys[1] == keys[2]

    def test_dual_cache_order_equal(self, builds_3d):
        order = [[vd.x for vd in HC.Vd] for HC in builds_3d]
        assert order[0] == order[1] == order[2]

    def test_vertex_dual_sets_iterate_in_the_same_order(self, builds_3d):
        """The fan walks start at ``next(iter(v_i.vd & v_j.vd))``."""
        walks = []
        for HC in builds_3d:
            walks.append([[vd.x for vd in v.vd.intersection(nb.vd)]
                          for v in HC.V for nb in v.nn])
        assert walks[0] == walks[1] == walks[2]

    def test_boundary_face_dual_is_summed_in_simplex_order(self, builds_3d):
        """The barycentre of a boundary face is the mean of its vertices
        in the order of the one simplex that owns the face."""
        HC = builds_3d[0]
        owners = {}
        for s in HC._simplices:
            for skip in range(4):
                face = tuple(v for i, v in enumerate(s) if i != skip)
                owners.setdefault(frozenset(face), []).append(face)
        n_checked = 0
        for faces in owners.values():
            if len(faces) != 1:
                continue
            bary = np.mean(np.array([v.x_a for v in faces[0]]), axis=0)
            assert tuple(bary) in HC.Vd.cache
            n_checked += 1
        assert n_checked > 10


@pytest.fixture(scope='module')
def builds_2d():
    coords = _cloud(2, 80, seed=5)
    out, keep = [], []
    for seed in (0, 3, 4):
        if seed:
            keep.append(_fragment_heap(seed))
        out.append(_build(2, coords))
    return out


class TestRemeshEnumerationFollowsCacheOrder:
    def test_the_builds_sit_at_other_address_orders(self, builds_2d):
        assert len({_address_order(HC) for HC in builds_2d}) > 1

    def test_edge_list_is_oriented_by_cache_order(self, builds_2d):
        lists = []
        for HC in builds_2d:
            rank = {v: i for i, v in enumerate(HC.V)}
            edges = _edge_list(HC)
            assert all(rank[a] < rank[b] for a, b in edges)
            assert len({frozenset((a.x, b.x)) for a, b in edges}) == len(edges)
            assert len(edges) == sum(len(v.nn) for v in HC.V) // 2
            lists.append([(a.x, b.x) for a, b in edges])
        assert lists[0] == lists[1] == lists[2]

    def test_triangles_are_listed_in_cache_order(self, builds_2d):
        lists = []
        for HC in builds_2d:
            rank = {v: i for i, v in enumerate(HC.V)}
            tris = list(iter_triangles_2d(HC))
            assert all(rank[a] < rank[b] < rank[c] for a, b, c in tris)
            assert ({frozenset(v.x for v in t) for t in tris}
                    == {frozenset(v.x for v in s) for s in HC._simplices})
            assert len(tris) == len(HC._simplices)
            lists.append([tuple(v.x for v in t) for t in tris])
        assert lists[0] == lists[1] == lists[2]

    def test_adaptive_remesh_gives_the_same_complex(self, builds_2d):
        """Splits, collapses and flips of a random Delaunay mesh: the
        same vertices in the same cache order with the same edges and
        the same carried masses."""
        states = []
        for HC in builds_2d:
            for i, v in enumerate(HC.V):
                v.m = 1.0 + 0.01 * i
            stats = adaptive_remesh(HC, dim=2, alpha_min=0.6, alpha_max=1.3,
                                    max_iterations=2, smooth_iterations=1)
            assert stats['n_splits'] + stats['n_collapses'] > 0
            states.append((
                [(v.x, float(v.m)) for v in HC.V],
                [(a.x, b.x) for a, b in _edge_list(HC)],
            ))
        assert states[0] == states[1] == states[2]
