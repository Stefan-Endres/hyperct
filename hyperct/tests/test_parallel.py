"""Tests for parallel processing via multiprocessing pools."""
import pytest
import numpy
import multiprocessing as mp

from hyperct._complex import Complex
from hyperct._vertex import VertexCacheField, FieldWraper, ConstraintWraper


def simple_field(x):
    """Simple scalar field: sum of squares."""
    return numpy.sum(numpy.array(x) ** 2)


def simple_constraint(x):
    """Constraint: x[0] >= 0.5 (g(x) = x[0] - 0.5, feasible when g >= 0)."""
    return x[0] - 0.5


@pytest.fixture
def pool_cleanup():
    """Track pools created during test and terminate them after."""
    pools = []
    yield pools
    for p in pools:
        try:
            p.terminate()
            p.join(timeout=5)
        except Exception:
            pass


# --- FieldWraper / ConstraintWraper unit tests ---

class TestFieldWraper:
    """Tests for the FieldWraper pickling helper."""

    def test_evaluates_correctly(self):
        """FieldWraper.func evaluates the wrapped field function."""
        fw = FieldWraper(simple_field, ())
        result = fw.func(numpy.array([1.0, 2.0]))
        assert result == 5.0

    def test_handles_nan_as_inf(self):
        """NaN field values are converted to inf."""
        def nan_field(x):
            return float('nan')
        fw = FieldWraper(nan_field, ())
        result = fw.func(numpy.array([1.0]))
        assert result == numpy.inf

    def test_handles_exception_as_inf(self):
        """Field function exceptions produce inf."""
        def bad_field(x):
            raise ValueError("bad")
        fw = FieldWraper(bad_field, ())
        result = fw.func(numpy.array([1.0]))
        assert result == numpy.inf


class TestConstraintWraper:
    """Tests for the ConstraintWraper pickling helper."""

    def test_feasible_point(self):
        """Point satisfying constraint returns True."""
        cw = ConstraintWraper((simple_constraint,), ((),))
        assert cw.gcons(numpy.array([1.0, 0.0])) is True

    def test_infeasible_point(self):
        """Point violating constraint returns False."""
        cw = ConstraintWraper((simple_constraint,), ((),))
        assert cw.gcons(numpy.array([0.0, 0.0])) is False

    def test_boundary_point(self):
        """Point exactly on constraint boundary (g=0) is feasible."""
        cw = ConstraintWraper((simple_constraint,), ((),))
        # g(0.5, 0) = 0.0, which is NOT < 0, so feasible
        assert cw.gcons(numpy.array([0.5, 0.0])) is True


# --- VertexCacheField parallel setup tests ---

class TestVertexCacheFieldParallel:
    """Tests for parallel pool creation in VertexCacheField."""

    def test_pool_created_with_workers(self, pool_cleanup):
        """Setting workers creates a multiprocessing pool."""
        vcf = VertexCacheField(field=simple_field, workers=2)
        pool_cleanup.append(vcf.pool)
        assert hasattr(vcf, 'pool')
        assert isinstance(vcf.pool, mp.pool.Pool)

    def test_no_pool_without_workers(self):
        """Without workers param, no pool is created."""
        vcf = VertexCacheField(field=simple_field)
        assert not hasattr(vcf, 'pool')

    def test_parallel_dispatch_no_constraints(self, pool_cleanup):
        """With workers and no constraints, parallel fpool method is used."""
        vcf = VertexCacheField(field=simple_field, workers=2)
        pool_cleanup.append(vcf.pool)
        assert vcf.process_fpool == vcf.pproc_fpool_nog
        assert vcf.process_gpool == vcf.pproc_gpool

    def test_parallel_dispatch_with_constraints(self, pool_cleanup):
        """With workers and constraints, parallel constraint fpool is used."""
        vcf = VertexCacheField(
            field=simple_field,
            g_cons=(simple_constraint,),
            g_cons_args=((),),
            workers=2
        )
        pool_cleanup.append(vcf.pool)
        assert vcf.process_fpool == vcf.pproc_fpool_g

    def test_workers_count_stored(self, pool_cleanup):
        """Workers count is stored on the cache object."""
        vcf = VertexCacheField(field=simple_field, workers=3)
        pool_cleanup.append(vcf.pool)
        assert vcf.workers == 3


# --- Integration tests through Complex ---

class TestComplexParallel:
    """Integration tests for parallel processing through the Complex class."""

    def test_complex_with_workers_creates_pool(self, pool_cleanup):
        """Complex with workers param creates pool on vertex cache."""
        HC = Complex(2, sfield=simple_field, workers=2)
        pool_cleanup.append(HC.V.pool)
        assert hasattr(HC.V, 'pool')

    def test_parallel_triangulation_matches_serial(self, pool_cleanup):
        """Parallel and serial triangulation produce identical vertices."""
        HC_serial = Complex(2, sfield=simple_field)
        HC_serial.triangulate()
        HC_serial.V.process_pools()

        HC_parallel = Complex(2, sfield=simple_field, workers=2)
        pool_cleanup.append(HC_parallel.V.pool)
        HC_parallel.triangulate()
        HC_parallel.V.process_pools()

        # Same vertex set
        assert set(HC_serial.V.cache.keys()) == set(HC_parallel.V.cache.keys())
        # Same field values
        for v in HC_serial.V.cache:
            numpy.testing.assert_allclose(
                HC_serial.V[v].f, HC_parallel.V[v].f,
                err_msg=f"Field mismatch at vertex {v}"
            )

    def test_parallel_with_constraints(self, pool_cleanup):
        """Parallel processing with constraints marks feasibility correctly."""
        cons = {'type': 'ineq', 'fun': simple_constraint}
        HC = Complex(2, sfield=simple_field, constraints=cons, workers=2)
        pool_cleanup.append(HC.V.pool)
        HC.triangulate()
        HC.V.process_pools()
        # Origin (0,0): x[0]=0 < 0.5, g(x)=-0.5 < 0 → infeasible
        assert HC.V[(0.0, 0.0)].feasible is False
        # (1.0, 1.0): x[0]=1 > 0.5, g(x)=0.5 > 0 → feasible
        assert HC.V[(1.0, 1.0)].feasible is True

    def test_parallel_with_refinement(self, pool_cleanup):
        """Parallel processing works after refinement adds more vertices."""
        HC_serial = Complex(2, sfield=simple_field)
        HC_serial.triangulate()
        HC_serial.refine_all()
        HC_serial.V.process_pools()

        HC_parallel = Complex(2, sfield=simple_field, workers=2)
        pool_cleanup.append(HC_parallel.V.pool)
        HC_parallel.triangulate()
        HC_parallel.refine_all()
        HC_parallel.V.process_pools()

        assert set(HC_serial.V.cache.keys()) == set(HC_parallel.V.cache.keys())
        for v in HC_serial.V.cache:
            numpy.testing.assert_allclose(
                HC_serial.V[v].f, HC_parallel.V[v].f,
                err_msg=f"Field mismatch at vertex {v} after refinement"
            )
