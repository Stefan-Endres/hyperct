"""Tests covering code paths exercised by the examples/ scripts.

These fill coverage gaps for: refine(n) lazy refinement, minimiser/maximiser
on scalar field vertices, non-parallel constraint processing through Complex,
and process_pools with field + constraints.
"""
import numpy
import pytest

from hyperct._complex import Complex


def simple_field(x):
    return numpy.sum(numpy.array(x) ** 2)


def eggholder(x):
    return (-(x[1] + 47.0)
            * numpy.sin(numpy.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0))))
            - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0)))))


def circle_constraint(x):
    """g(x) <= 0 keeps points inside the unit circle centered at (0.5, 0.5)."""
    return (x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.25


# --- refine(n) lazy refinement ---

class TestRefineN:
    """Tests for Complex.refine(n) (lazy target-count refinement)."""

    def test_refine_n_reaches_target(self):
        """refine(n) should produce at least n total vertices."""
        H = Complex(2)
        H.refine(n=20)
        assert len(H.V.cache) >= 20

    def test_refine_n_from_scratch(self):
        """refine(n) on a fresh complex triggers triangulate internally."""
        H = Complex(2)
        H.refine(n=5)
        assert len(H.V.cache) >= 5

    def test_refine_n_incremental(self):
        """Successive refine(n) calls add more vertices."""
        H = Complex(2)
        H.refine(n=10)
        count1 = len(H.V.cache)
        H.refine(n=20)
        count2 = len(H.V.cache)
        assert count2 > count1

    def test_refine_none_triggers_triangulate(self):
        """refine(None) on a fresh complex calls triangulate."""
        H = Complex(2)
        H.refine(n=None)
        assert len(H.V.cache) > 0

    def test_refine_none_after_triangulate(self):
        """refine(None) after triangulate calls refine_all."""
        H = Complex(2)
        H.triangulate()
        count_before = len(H.V.cache)
        H.refine(n=None)
        assert len(H.V.cache) > count_before


# --- minimiser / maximiser on scalar field vertices ---

class TestMinimiserMaximiser:
    """Tests for VertexScalarField.minimiser() and maximiser()."""

    def test_minimiser_found(self):
        """A refined complex with a convex field should have a minimiser."""
        H = Complex(2, sfield=simple_field)
        H.triangulate()
        H.refine_all()
        H.V.process_pools()

        minimisers = [v for v in H.V.cache if H.V[v].minimiser()]
        assert len(minimisers) >= 1
        # The origin (0,0) should be the minimiser for sum-of-squares
        assert (0.0, 0.0) in minimisers

    def test_maximiser_found(self):
        """A refined complex with a convex field should have a maximiser."""
        H = Complex(2, sfield=simple_field)
        H.triangulate()
        H.refine_all()
        H.V.process_pools()

        maximisers = [v for v in H.V.cache if H.V[v].maximiser()]
        assert len(maximisers) >= 1

    def test_minimiser_caching(self):
        """Calling minimiser() twice returns the same result (cached)."""
        H = Complex(2, sfield=simple_field)
        H.triangulate()
        H.V.process_pools()

        v = H.V[(0.0, 0.0)]
        result1 = v.minimiser()
        result2 = v.minimiser()
        assert result1 == result2
        assert v.check_min is False  # Should be cached

    def test_maximiser_caching(self):
        """Calling maximiser() twice returns the same result (cached)."""
        H = Complex(2, sfield=simple_field)
        H.triangulate()
        H.V.process_pools()

        v = H.V[(1.0, 1.0)]
        result1 = v.maximiser()
        result2 = v.maximiser()
        assert result1 == result2
        assert v.check_max is False

    def test_field_values_stored(self):
        """After process_pools, each vertex has a finite .f value."""
        H = Complex(2, sfield=simple_field)
        H.triangulate()
        H.V.process_pools()

        for v in H.V.cache:
            assert numpy.isfinite(H.V[v].f)

    def test_proc_minimisers_via_process_pools(self):
        """process_pools calls proc_minimisers, setting _min/_max on all."""
        H = Complex(2, sfield=simple_field)
        H.triangulate()
        H.V.process_pools()

        for v in H.V.cache:
            vertex = H.V[v]
            assert hasattr(vertex, '_min')
            assert hasattr(vertex, '_max')


# --- Non-parallel constraint processing ---

class TestConstraints:
    """Tests for constraint processing without parallel workers."""

    def test_single_constraint_dict(self):
        """A single constraint dict (not in a list) is accepted."""
        cons = {'type': 'ineq', 'fun': circle_constraint}
        H = Complex(2, sfield=simple_field, constraints=cons)
        H.triangulate()
        H.V.process_pools()

        # At least some vertices should be infeasible (outside circle)
        feasible = [v for v in H.V.cache if H.V[v].feasible]
        infeasible = [v for v in H.V.cache if not H.V[v].feasible]
        assert len(feasible) > 0
        assert len(infeasible) > 0

    def test_multiple_constraints(self):
        """Multiple constraint dicts process correctly."""
        def g1(x):
            return x[0] - 0.3  # x0 >= 0.3

        def g2(x):
            return x[1] - 0.3  # x1 >= 0.3

        cons = [
            {'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2},
        ]
        H = Complex(2, sfield=simple_field, constraints=cons)
        H.triangulate()
        H.V.process_pools()

        # Origin should be infeasible (violates both constraints)
        assert H.V[(0.0, 0.0)].feasible is False
        # (1, 1) should be feasible
        assert H.V[(1.0, 1.0)].feasible is True

    def test_constraint_with_args(self):
        """Constraint functions receive extra args correctly."""
        def g_with_args(x, threshold):
            return x[0] - threshold

        cons = {'type': 'ineq', 'fun': g_with_args, 'args': (0.5,)}
        H = Complex(2, sfield=simple_field, constraints=cons)
        H.triangulate()
        H.V.process_pools()

        assert H.V[(0.0, 0.0)].feasible is False
        assert H.V[(1.0, 1.0)].feasible is True

    def test_infeasible_vertex_has_inf_field(self):
        """Infeasible vertices get f=inf when field + constraints both set."""
        def g(x):
            return x[0] - 0.5

        cons = {'type': 'ineq', 'fun': g}
        H = Complex(2, sfield=simple_field, constraints=cons)
        H.triangulate()
        H.V.process_pools()

        # (0, 0) is infeasible, should have f=inf
        assert H.V[(0.0, 0.0)].f == numpy.inf

    def test_constraints_only_no_field(self):
        """Constraints without a scalar field still mark feasibility."""
        def g(x):
            return x[0] - 0.5

        cons = {'type': 'ineq', 'fun': g}
        H = Complex(2, constraints=cons)
        H.triangulate()
        H.V.process_pools()

        assert H.V[(0.0, 0.0)].feasible is False
        assert H.V[(1.0, 1.0)].feasible is True


# --- Custom domain bounds ---

class TestCustomDomain:
    """Tests for non-default domain bounds."""

    def test_custom_bounds_vertices(self):
        """Vertices respect the specified domain bounds."""
        H = Complex(2, domain=[(-5.0, 5.0), (-10.0, 10.0)])
        H.triangulate()

        for v in H.V.cache:
            assert -5.0 <= v[0] <= 5.0
            assert -10.0 <= v[1] <= 10.0

    def test_3d_custom_domain(self):
        """3D complex with custom domain triangulates correctly."""
        H = Complex(3, domain=[(-1.0, 1.0)] * 3)
        H.triangulate()
        H.refine_all()

        assert len(H.V.cache) > 9  # More than initial triangulation
        for v in H.V.cache:
            for vi in v:
                assert -1.0 <= vi <= 1.0
