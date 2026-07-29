"""Regression tests for mass-conservative 2D remesh operations and the
per-edge local length scale of the adaptive driver.

Background (ddgclib debugging lane 4 / open problem A.4):

- ``edge_split_2d`` used to ASSIGN the midpoint the arithmetic mean of
  the endpoint masses, inflating total mass on every split (documented
  blow-up: mass 9.7 -> 187 over 100 steps under adaptive remesh).  It
  now transfers a conservative share FROM the endpoints, sized by the
  barycentric dual-area fraction each endpoint cedes, so ``sum(m)`` is
  invariant and a uniform density field stays exactly uniform.
- ``edge_collapse_2d`` summed ``v.m`` but let per-phase masses
  (``v.m_phase``) fall through the averaging branch of ``_carry_attrs``,
  silently destroying half of the merged per-phase mass.  Both are now
  additive.  A position collision at the merged midpoint used to be
  detected only after the connectivity merge (returning True with the
  survivor left in place); it now aborts cleanly BEFORE any mutation.
- ``adaptive_remesh`` used a single mesh-wide median edge length to
  derive ``L_min``/``L_max``, so a fine droplet region and a coarse
  outer region cross-contaminated the thresholds (unbounded splits in
  the coarse region).  The default is now a per-edge local length scale
  (``length_scale='local'``); ``length_scale='global'`` restores the
  legacy behaviour.
"""
import numpy as np
import pytest

from hyperct._complex import Complex
from hyperct.remesh import (
    adaptive_remesh,
    can_collapse,
    edge_collapse_2d,
    edge_split_2d,
)
from hyperct.remesh._operations_2d import _one_ring_area
from hyperct.remesh._quality import edge_length


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _grid_2d(nx=3, ny=3, Lx=1.0, Ly=1.0):
    """Uniform triangulated grid (same construction as test_remesh.py)."""
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
            a.connect(b)
            a.connect(d)
            b.connect(d)
            a.connect(c)
            c.connect(d)
    for (i, j), v in verts.items():
        v.boundary = (i == 0 or j == 0 or i == nx - 1 or j == ny - 1)
    return HC, verts


def _mixed_scale_mesh(h_coarse=1.0, h_fine=0.12, R_disc=0.6, L_box=4.0):
    """Fine disc (spacing ``h_fine``, phase 1) embedded in a coarse box
    (spacing ``h_coarse``, phase 0), Delaunay-triangulated.  This is the
    configuration that blew up under the legacy global-median driver.
    """
    from scipy.spatial import Delaunay

    pts = []
    c = L_box / 2.0
    n = int(round(L_box / h_coarse)) + 1
    for i in range(n):
        for j in range(n):
            p = (i * h_coarse, j * h_coarse)
            if (p[0] - c) ** 2 + (p[1] - c) ** 2 > (R_disc + 0.4 * h_coarse) ** 2:
                pts.append(p)
    m = int(round(2 * R_disc / h_fine)) + 1
    for i in range(m):
        for j in range(m):
            p = (c - R_disc + i * h_fine, c - R_disc + j * h_fine)
            if (p[0] - c) ** 2 + (p[1] - c) ** 2 <= R_disc ** 2:
                pts.append(p)
    pts = np.array(pts)
    tri = Delaunay(pts)
    HC = Complex(2)
    verts = [HC.V[tuple(p)] for p in pts]
    for s in tri.simplices:
        a, b, cc = (verts[k] for k in s)
        a.connect(b)
        b.connect(cc)
        a.connect(cc)
    for v in HC.V:
        x, y = v.x_a[:2]
        v.boundary = bool(x <= 0 or y <= 0 or x >= L_box or y >= L_box)
        v.phase = 1 if (x - c) ** 2 + (y - c) ** 2 <= R_disc ** 2 else 0
    return HC


def _assign_uniform_density_mass(HC, rho=1.0, n_phases=2):
    """v.m = rho * (barycentric dual area) so density is exactly uniform;
    m_phase books the whole mass on the vertex's own phase."""
    for v in HC.V:
        v.m = rho * _one_ring_area(v) / 3.0
        mp = np.zeros(n_phases)
        mp[getattr(v, "phase", 0) % n_phases] = v.m
        v.m_phase = mp


def _total_mass(HC) -> float:
    return sum(float(v.m) for v in HC.V)


def _total_m_phase(HC) -> np.ndarray:
    return sum(np.asarray(v.m_phase, dtype=float) for v in HC.V)


def _longest_splittable_edge(HC):
    best, best_L = None, -1.0
    seen = set()
    for v in HC.V:
        for nb in v.nn:
            key = frozenset((id(v), id(nb)))
            if key in seen:
                continue
            seen.add(key)
            L = edge_length(v, nb)
            if L > best_L:
                best, best_L = (v, nb), L
    return best


def _shortest_collapsible_edge(HC):
    best, best_L = None, np.inf
    seen = set()
    for v in HC.V:
        for nb in v.nn:
            key = frozenset((id(v), id(nb)))
            if key in seen:
                continue
            seen.add(key)
            if not can_collapse(v, nb):
                continue
            L = edge_length(v, nb)
            if L < best_L:
                best, best_L = (v, nb), L
    return best


# ---------------------------------------------------------------------------
# Split conservation
# ---------------------------------------------------------------------------

class TestSplitMassConservation:
    def test_single_split_conserves_total_mass(self):
        HC, verts = _grid_2d(nx=4, ny=4)
        for k, v in enumerate(HC.V):
            v.phase = k % 2
        _assign_uniform_density_mass(HC)
        m0 = _total_mass(HC)
        v_m = edge_split_2d(HC, verts[(1, 1)], verts[(2, 2)])
        assert v_m is not None
        assert _total_mass(HC) == pytest.approx(m0, rel=1e-13)

    def test_repeated_splits_conserve_total_mass(self):
        HC, _ = _grid_2d(nx=4, ny=4)
        for v in HC.V:
            v.phase = 0
        _assign_uniform_density_mass(HC)
        m0 = _total_mass(HC)
        mp0 = _total_m_phase(HC)
        n_ok = 0
        for _ in range(25):
            edge = _longest_splittable_edge(HC)
            if edge is None:
                break
            if edge_split_2d(HC, *edge) is not None:
                n_ok += 1
        assert n_ok >= 20  # the operation must actually run
        assert _total_mass(HC) == pytest.approx(m0, rel=1e-12)
        np.testing.assert_allclose(_total_m_phase(HC), mp0, rtol=1e-12)

    def test_split_conserves_m_phase_and_consistency(self):
        """Per-phase mass is conserved componentwise and the midpoint
        keeps v.m == sum(v.m_phase)."""
        HC, verts = _grid_2d(nx=3, ny=3)
        for v in HC.V:
            v.phase = 0
        _assign_uniform_density_mass(HC, n_phases=2)
        mp0 = _total_m_phase(HC)
        v_m = edge_split_2d(HC, verts[(0, 0)], verts[(1, 1)])
        assert v_m is not None
        np.testing.assert_allclose(_total_m_phase(HC), mp0, rtol=1e-13)
        assert float(v_m.m) == pytest.approx(float(np.sum(v_m.m_phase)),
                                             rel=1e-13)

    def test_split_uniform_density_stays_uniform(self):
        """With v.m = rho * dual_area, the midpoint and both endpoints
        must all end at density rho w.r.t. their POST-split dual areas
        (exact property of the barycentric-fraction transfer)."""
        rho = 3.7
        HC, verts = _grid_2d(nx=4, ny=4)
        for v in HC.V:
            v.phase = 0
        _assign_uniform_density_mass(HC, rho=rho)
        v_i, v_j = verts[(1, 1)], verts[(2, 2)]
        v_m = edge_split_2d(HC, v_i, v_j)
        assert v_m is not None
        for v in (v_m, v_i, v_j):
            area = _one_ring_area(v) / 3.0
            assert float(v.m) == pytest.approx(rho * area, rel=1e-12)

    def test_split_transfers_from_endpoints(self):
        """The endpoints must LOSE exactly what the midpoint gains."""
        HC, verts = _grid_2d(nx=3, ny=3)
        a, b = verts[(0, 0)], verts[(1, 1)]
        a.m, b.m = 2.0, 4.0
        v_m = edge_split_2d(HC, a, b)
        assert v_m is not None
        assert (float(a.m) + float(b.m) + float(v_m.m)
                ) == pytest.approx(6.0, rel=1e-13)
        assert float(a.m) < 2.0 and float(b.m) < 4.0
        assert float(v_m.m) > 0.0


# ---------------------------------------------------------------------------
# Collapse conservation
# ---------------------------------------------------------------------------

class TestCollapseMassConservation:
    def test_collapse_conserves_total_mass(self):
        HC, verts = _grid_2d(nx=4, ny=4)
        for v in HC.V:
            v.phase = 0
        _assign_uniform_density_mass(HC)
        m0 = _total_mass(HC)
        assert edge_collapse_2d(HC, verts[(1, 1)], verts[(2, 2)])
        assert _total_mass(HC) == pytest.approx(m0, rel=1e-13)

    def test_collapse_sums_m_phase(self):
        """m_phase must be ADDED (was averaged by _carry_attrs before,
        destroying half of the merged per-phase mass)."""
        HC, verts = _grid_2d(nx=4, ny=4)
        v_i, v_j = verts[(1, 1)], verts[(2, 2)]
        v_i.m, v_j.m = 3.0, 5.0
        v_i.m_phase = np.array([1.0, 2.0])
        v_j.m_phase = np.array([4.0, 1.0])
        assert edge_collapse_2d(HC, v_i, v_j)
        np.testing.assert_allclose(v_i.m_phase, [5.0, 3.0], rtol=1e-13)
        assert float(v_i.m) == pytest.approx(8.0, rel=1e-13)

    def test_collapse_keeps_mass_when_only_removed_vertex_has_it(self):
        """Mass of the removed vertex must never be dropped, even when
        the survivor has no mass attribute yet."""
        HC, verts = _grid_2d(nx=4, ny=4)
        v_i, v_j = verts[(1, 1)], verts[(2, 2)]
        v_j.m = 5.0  # v_i has no m
        assert edge_collapse_2d(HC, v_i, v_j)
        assert float(v_i.m) == pytest.approx(5.0, rel=1e-13)

    def test_collapse_position_collision_aborts_before_mutation(self):
        """If a third vertex already occupies the merged midpoint, the
        collapse must return False and leave the mesh untouched
        (previously it merged the connectivity, skipped the move, and
        returned True)."""
        HC = Complex(2)
        a = HC.V[(0.0, 0.0)]
        b = HC.V[(1.0, 1.0)]
        c = HC.V[(1.0, 0.0)]
        d = HC.V[(0.0, 1.0)]
        for e in (b, c, d):
            a.connect(e)
        b.connect(c)
        b.connect(d)
        for v in (a, b, c, d):
            v.boundary = False
        blocker = HC.V[(0.5, 0.5)]  # occupies the midpoint of (a, b)
        a.m, b.m = 2.0, 4.0
        n_before = len(list(HC.V))
        nn_a_before = set(a.nn)

        ok = edge_collapse_2d(HC, a, b)

        assert ok is False
        assert len(list(HC.V)) == n_before
        assert b in a.nn
        assert set(a.nn) == nn_a_before
        assert float(a.m) == pytest.approx(2.0)
        assert float(b.m) == pytest.approx(4.0)
        assert blocker in set(HC.V)


# ---------------------------------------------------------------------------
# Momentum conservation
# ---------------------------------------------------------------------------

def _total_momentum(HC) -> np.ndarray:
    return sum(float(v.m) * np.asarray(v.u, dtype=float) for v in HC.V)


def _total_ke(HC) -> float:
    return sum(0.5 * float(v.m) * float(np.dot(v.u, v.u)) for v in HC.V)


class TestMomentumConservation:
    def _seeded_mesh(self):
        HC, verts = _grid_2d(nx=4, ny=4)
        rng = np.random.default_rng(42)
        for v in HC.V:
            v.phase = 0
        _assign_uniform_density_mass(HC)
        for v in HC.V:
            v.m = float(v.m) * float(rng.uniform(0.5, 2.0))
            v.u = rng.normal(size=2)
        return HC, verts

    def test_split_conserves_momentum_and_dissipates_ke(self):
        HC, verts = self._seeded_mesh()
        p0 = _total_momentum(HC)
        ke0 = _total_ke(HC)
        v_m = edge_split_2d(HC, verts[(1, 1)], verts[(2, 2)])
        assert v_m is not None
        np.testing.assert_allclose(_total_momentum(HC), p0,
                                   rtol=0, atol=1e-13)
        assert _total_ke(HC) <= ke0 + 1e-13

    def test_collapse_conserves_momentum_and_dissipates_ke(self):
        HC, verts = self._seeded_mesh()
        p0 = _total_momentum(HC)
        ke0 = _total_ke(HC)
        assert edge_collapse_2d(HC, verts[(1, 1)], verts[(2, 2)])
        np.testing.assert_allclose(_total_momentum(HC), p0,
                                   rtol=0, atol=1e-13)
        assert _total_ke(HC) <= ke0 + 1e-13


# ---------------------------------------------------------------------------
# Split/collapse round trips
# ---------------------------------------------------------------------------

class TestSplitCollapseCycle:
    def test_cycle_mass_invariant(self):
        """Alternating splits and collapses on a static mesh must keep
        sum(m) and sum(m_phase) invariant to round-off."""
        HC, _ = _grid_2d(nx=5, ny=5)
        for v in HC.V:
            v.phase = 0
        _assign_uniform_density_mass(HC)
        m0 = _total_mass(HC)
        mp0 = _total_m_phase(HC)
        n_ops = 0
        for _ in range(15):
            edge = _longest_splittable_edge(HC)
            if edge is not None and edge_split_2d(HC, *edge) is not None:
                n_ops += 1
            edge = _shortest_collapsible_edge(HC)
            if edge is not None and edge_collapse_2d(HC, *edge):
                n_ops += 1
        assert n_ops >= 15
        assert _total_mass(HC) == pytest.approx(m0, rel=1e-12)
        np.testing.assert_allclose(_total_m_phase(HC), mp0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Adaptive driver: local length scale + driver-level conservation
# ---------------------------------------------------------------------------

class TestLocalLengthScale:
    def test_mixed_scale_mesh_stays_within_vertex_budget(self):
        """Fine disc in a coarse box: repeated adaptive_remesh calls
        (default length_scale='local') must NOT blow up the vertex
        count.  Probe baseline 2026-07-02: 95 -> 106 vertices over 10
        calls; the legacy global-median mode explodes 95 -> 554 on the
        identical mesh."""
        HC = _mixed_scale_mesh()
        _assign_uniform_density_mass(HC)
        n0 = len(list(HC.V))
        m0 = _total_mass(HC)
        for _ in range(10):
            adaptive_remesh(HC, dim=2, max_iterations=1)
        n_final = len(list(HC.V))
        assert n_final <= 1.5 * n0, (
            f"vertex budget blown: {n0} -> {n_final}")
        # Driver-level mass conservation (splits + collapses + flips +
        # smoothing all together).
        assert _total_mass(HC) == pytest.approx(m0, rel=1e-12)

    def test_local_beats_global_on_mixed_scale_mesh(self):
        """Documents the cross-contamination fix: on the same mixed
        mesh the legacy global-median thresholds grow the vertex count
        far faster than the per-edge local thresholds."""
        HC_loc = _mixed_scale_mesh()
        HC_glob = _mixed_scale_mesh()
        n0 = len(list(HC_loc.V))
        for _ in range(3):
            adaptive_remesh(HC_loc, dim=2, max_iterations=1,
                            length_scale="local")
            adaptive_remesh(HC_glob, dim=2, max_iterations=1,
                            length_scale="global")
        n_loc = len(list(HC_loc.V))
        n_glob = len(list(HC_glob.V))
        assert n_loc <= 1.5 * n0
        assert n_glob > n_loc

    def test_explicit_thresholds_unaffected_by_length_scale(self):
        """Explicit L_min/L_max remain absolute regardless of
        length_scale (public API preserved)."""
        HC_a, _ = _grid_2d(nx=3, ny=3)
        HC_b, _ = _grid_2d(nx=3, ny=3)
        kw = dict(L_min=0.01, L_max=0.2, max_iterations=1,
                  smooth_iterations=0)
        s_a = adaptive_remesh(HC_a, dim=2, length_scale="local", **kw)
        s_b = adaptive_remesh(HC_b, dim=2, length_scale="global", **kw)
        assert s_a["n_splits"] == s_b["n_splits"] > 0
        assert s_a["n_collapses"] == s_b["n_collapses"]

    def test_uniform_grid_local_mode_no_ops(self):
        """A uniform grid has no local outliers — the local length
        scale must not trigger splits or collapses."""
        HC, _ = _grid_2d(nx=5, ny=5)
        stats = adaptive_remesh(HC, dim=2, max_iterations=1,
                                smooth_iterations=0)
        assert stats["n_splits"] == 0
        assert stats["n_collapses"] == 0

    def test_invalid_length_scale_raises(self):
        HC, _ = _grid_2d(nx=3, ny=3)
        with pytest.raises(ValueError):
            adaptive_remesh(HC, dim=2, length_scale="nonsense")

    def test_smooth_skip_interface_pins_interface(self):
        """With smooth_skip_interface=True, smoothing must not move
        interface vertices (tangential smoothing systematically shrinks
        closed interface loops in Lagrangian runs)."""
        HC, verts = _grid_2d(nx=5, ny=5)
        for (i, j), v in verts.items():
            v.phase = 0 if i <= 1 else 1
        iface = [v for v in HC.V
                 if any(getattr(nb, "phase", None) != v.phase
                        for nb in v.nn)]
        pos_before = {id(v): tuple(v.x) for v in iface}
        adaptive_remesh(HC, dim=2, L_min=1e-6, L_max=10.0,
                        max_iterations=1, smooth_iterations=2,
                        smooth_relax=0.5, smooth_skip_interface=True)
        for v in iface:
            assert tuple(v.x) == pos_before[id(v)]

    def test_empty_mesh_local_mode(self):
        HC = Complex(2)
        HC.V[(0.0, 0.0)]
        stats = adaptive_remesh(HC, dim=2)
        assert stats["n_splits"] == 0
        assert stats["n_triangles"] == 0


# ---------------------------------------------------------------------------
# Simplex-cache rebuild after local operations
# ---------------------------------------------------------------------------

class TestRebuildSimplexCache2D:
    def test_grid_triangle_count(self):
        from hyperct.ddg import rebuild_simplex_cache_2d
        HC, _ = _grid_2d(nx=4, ny=4)
        n = rebuild_simplex_cache_2d(HC)
        assert n == 18  # 3x3 quads * 2 triangles
        assert len(HC._simplices) == 18

    def test_ghost_k3_filtered(self):
        """A triangle subdivided by an interior vertex connected to all
        three corners must cache the 3 sub-triangles, NOT the outer
        K_3 clique (the flag-complex ambiguity)."""
        from hyperct.ddg import rebuild_simplex_cache_2d
        HC = Complex(2)
        a = HC.V[(0.0, 0.0)]
        b = HC.V[(2.0, 0.0)]
        c = HC.V[(1.0, 2.0)]
        m = HC.V[(1.0, 0.7)]  # strictly inside (a, b, c)
        a.connect(b)
        b.connect(c)
        a.connect(c)
        for v in (a, b, c):
            m.connect(v)
        n = rebuild_simplex_cache_2d(HC)
        assert n == 3
        keys = {frozenset(id(v) for v in tri) for tri in HC._simplices}
        assert frozenset((id(a), id(b), id(c))) not in keys

    def test_partition_of_unity_after_ops(self):
        """After a split + collapse + rebuild, the exact dual volumes
        over the rebuilt cache must partition the total mesh area."""
        from hyperct.ddg import rebuild_simplex_cache_2d, simplex_dual_volumes
        from hyperct.remesh._quality import triangle_area

        HC, verts = _grid_2d(nx=4, ny=4)
        assert edge_split_2d(HC, verts[(1, 1)], verts[(2, 2)]) is not None
        edge = _shortest_collapsible_edge(HC)
        assert edge is not None and edge_collapse_2d(HC, *edge)
        rebuild_simplex_cache_2d(HC)
        total_area = sum(abs(triangle_area(*tri)) for tri in HC._simplices)
        vols = simplex_dual_volumes(HC, 2)
        assert sum(vols.values()) == pytest.approx(total_area, rel=1e-12)
