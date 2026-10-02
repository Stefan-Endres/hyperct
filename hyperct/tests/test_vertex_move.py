"""``VertexCacheBase.move`` onto an occupied coordinate key, and
``move_all``.

The cache is keyed by coordinate tuple, so two vertices cannot share a
position.  Until 2026-10-01 ``move`` onto the key of another vertex took
the key silently: the occupant dropped out of the cache with its edges
left in place, and when the occupant was moved later it removed the mover
from the cache in turn.  A loop that shifts a structured mesh by half its
width lost one vertex per collision that way (ddgclib audit 2026-09-25,
F10 C2).
"""
import pytest

from hyperct._complex import Complex
from hyperct._vertex import VertexCollisionError


def _grid(refine=1):
    """Unit square, (2**refine + 1)**2 lattice vertices plus cell centres."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    for _ in range(refine):
        HC.refine_all()
    return HC


def _state(HC):
    return {v.x: frozenset(nb.x for nb in v.nn) for v in HC.V}


def _consistent(HC):
    """Every vertex sits under its own key and every edge joins two
    vertices of the cache."""
    members = {id(v) for v in HC.V}
    for key, v in HC.V.cache.items():
        assert key == v.x and hash(v) == hash(key)
        for nb in v.nn:
            assert id(nb) in members and v in nb.nn


class TestMoveCollision:
    def test_move_onto_an_occupied_key_is_refused(self):
        HC = _grid()
        before = _state(HC)
        v, w = HC.V[(0.0, 0.0)], HC.V[(0.5, 0.5)]
        with pytest.raises(VertexCollisionError, match='held by another'):
            HC.V.move(v, w.x)
        # nothing changed: keys, edges, and both vertices are still there
        assert _state(HC) == before
        assert HC.V.cache[(0.0, 0.0)] is v and HC.V.cache[(0.5, 0.5)] is w
        _consistent(HC)

    def test_move_onto_its_own_key_is_allowed(self):
        HC = _grid()
        v = HC.V[(0.5, 0.5)]
        before = _state(HC)
        assert HC.V.move(v, (0.5, 0.5)) is v
        assert _state(HC) == before

    def test_move_to_a_free_key_is_unchanged(self):
        HC = _grid()
        v = HC.V[(0.5, 0.5)]
        nn = set(v.nn)
        assert HC.V.move(v, (0.55, 0.45)) is v
        assert v.x == (0.55, 0.45) and set(v.nn) == nn
        assert (0.5, 0.5) not in HC.V.cache
        _consistent(HC)

    def test_evict_is_the_old_behaviour(self):
        """Documented, not endorsed: the occupant leaves the cache with
        its edges in place, and moving it later removes the mover."""
        HC = _grid()
        n = len(HC.V)
        v, w = HC.V[(0.0, 0.0)], HC.V[(0.5, 0.5)]
        HC.V.move(v, w.x, on_collision='evict')
        assert len(HC.V) == n - 1
        assert HC.V.cache[(0.5, 0.5)] is v
        assert all(u is not w for u in HC.V) and w.nn
        HC.V.move(w, (0.9, 0.9), on_collision='evict')
        assert len(HC.V) == n - 1
        assert all(u is not v for u in HC.V)

    def test_unknown_policy_raises(self):
        HC = _grid()
        with pytest.raises(ValueError, match='on_collision'):
            HC.V.move(HC.V[(0.0, 0.0)], (0.1, 0.1), on_collision='merge')

    def test_shift_loop_that_lost_vertices_now_raises(self):
        """The loop of the audit: shift the square by half its width."""
        HC = _grid(refine=2)
        with pytest.raises(VertexCollisionError):
            for v in list(HC.V):
                HC.V.move(v, (v.x[0] - 0.5, v.x[1] - 0.5))

    def test_the_same_loop_with_evict_loses_vertices(self):
        HC = _grid(refine=2)
        n = len(HC.V)
        for v in list(HC.V):
            HC.V.move(v, (v.x[0] - 0.5, v.x[1] - 0.5), on_collision='evict')
        assert len(HC.V) < n


class TestMoveAll:
    def test_shift_keeps_every_vertex_and_edge(self):
        HC = _grid(refine=2)
        before = _state(HC)
        verts = list(HC.V)
        HC.V.move_all([(v, (v.x[0] - 0.5, v.x[1] - 0.5)) for v in verts])
        assert len(HC.V) == len(verts)
        shift = lambda x: (x[0] - 0.5, x[1] - 0.5)  # noqa: E731
        assert _state(HC) == {shift(k): frozenset(shift(n) for n in nn)
                              for k, nn in before.items()}
        _consistent(HC)

    def test_rescale_keeps_every_vertex(self):
        HC = _grid(refine=2)
        verts = list(HC.V)
        HC.V.move_all([(v, (2.0 * v.x[0], v.x[1])) for v in verts])
        assert len(HC.V) == len(verts)
        assert max(v.x[0] for v in HC.V) == 2.0
        _consistent(HC)

    def test_cache_order_is_that_of_a_move_loop(self):
        A, B = _grid(), _grid()
        shift = lambda x: (x[0] + 0.03, x[1])  # noqa: E731  (no collision)
        subset = [x for x in list(A.V.cache)[::2]]
        for x in subset:
            A.V.move(A.V[x], shift(x))
        B.V.move_all([(B.V[x], shift(x)) for x in subset])
        assert list(A.V.cache) == list(B.V.cache)

    def test_two_movers_on_one_key_are_refused(self):
        HC = _grid()
        before = _state(HC)
        a, b = HC.V[(0.0, 0.0)], HC.V[(1.0, 1.0)]
        with pytest.raises(VertexCollisionError, match='share'):
            HC.V.move_all([(a, (0.3, 0.3)), (b, (0.3, 0.3))])
        assert _state(HC) == before

    def test_mover_onto_a_vertex_that_stays_is_refused(self):
        HC = _grid()
        before = _state(HC)
        a = HC.V[(0.0, 0.0)]
        with pytest.raises(VertexCollisionError):
            HC.V.move_all([(a, (0.5, 0.5))])
        assert _state(HC) == before
        _consistent(HC)

    def test_swap_of_two_vertices(self):
        HC = _grid()
        a, b = HC.V[(0.0, 0.0)], HC.V[(1.0, 1.0)]
        HC.V.move_all([(a, (1.0, 1.0)), (b, (0.0, 0.0))])
        assert HC.V.cache[(1.0, 1.0)] is a and HC.V.cache[(0.0, 0.0)] is b
        _consistent(HC)

    def test_vertex_listed_twice_is_refused_before_anything_changes(self):
        HC = _grid()
        before = _state(HC)
        a, b = HC.V[(0.0, 0.0)], HC.V[(1.0, 1.0)]
        with pytest.raises(ValueError, match='more than once'):
            HC.V.move_all([(b, (0.2, 0.2)), (a, (0.3, 0.3)), (a, (0.4, 0.4))])
        assert _state(HC) == before
        _consistent(HC)

    def test_vertex_not_in_the_cache_is_refused_before_anything_changes(self):
        HC = _grid()
        a, b = HC.V[(0.0, 0.0)], HC.V[(1.0, 1.0)]
        HC.V.remove(b)
        before = _state(HC)
        with pytest.raises(ValueError, match='not in this cache'):
            HC.V.move_all([(a, (0.3, 0.3)), (b, (0.4, 0.4))])
        assert _state(HC) == before
        _consistent(HC)
