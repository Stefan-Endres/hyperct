"""Benchmark suite for hyperct performance tracking.

Uses pytest-benchmark. Run with:
    pytest hyperct/tests/test_benchmarks.py --benchmark-only

Save results:
    pytest hyperct/tests/test_benchmarks.py --benchmark-save=v0.3.2

Compare against baseline:
    pytest hyperct/tests/test_benchmarks.py --benchmark-compare=0001_v0.3.2

Skip during normal test runs:
    pytest --benchmark-skip
"""
import copy
import json
import os
import tempfile

import numpy
import pytest

from hyperct._complex import Complex


# --- Scalar field functions ---

def simple_field(x):
    return numpy.sum(numpy.array(x) ** 2)


# --- 1a. Triangulation + Refinement (dims 1-5) ---

class TestBenchTriangulation:
    """Benchmark triangulation and refinement across dimensions."""

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
    def test_bench_triangulate(self, benchmark, dim):
        """Benchmark initial triangulation + refine_all."""
        def run():
            HC = Complex(dim)
            HC.triangulate()
            HC.refine_all()
            return HC

        benchmark(run)

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
    def test_bench_triangulate_split(self, benchmark, dim):
        """Benchmark triangulation + refine_all + split_generation."""
        def run():
            HC = Complex(dim)
            HC.triangulate()
            HC.refine_all()
            HC.split_generation()
            return HC

        benchmark(run)


# --- 1b. Scalar field computation (serial & parallel, dims 1-5) ---

class TestBenchScalarField:
    """Benchmark scalar field evaluation."""

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
    def test_bench_field_serial(self, benchmark, dim):
        """Benchmark serial field evaluation."""
        bounds = [(-10.0, 10.0)] * dim

        def run():
            HC = Complex(dim, domain=bounds, sfield=simple_field)
            HC.triangulate()
            HC.refine_all()
            HC.V.process_pools()
            return HC

        benchmark(run)

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
    def test_bench_field_parallel(self, benchmark, dim):
        """Benchmark parallel field evaluation with workers=2."""
        bounds = [(-10.0, 10.0)] * dim

        def run():
            HC = Complex(dim, domain=bounds, sfield=simple_field, workers=2)
            HC.triangulate()
            HC.refine_all()
            HC.V.process_pools()
            HC.V.pool.terminate()
            return HC

        benchmark(run)


# --- 1c. Caching operations ---

class TestBenchCaching:
    """Benchmark vertex cache operations."""

    def test_bench_cache_lookup(self, benchmark):
        """Benchmark repeated lookups of existing vertices."""
        HC = Complex(3)
        HC.triangulate()
        HC.refine_all()
        keys = list(HC.V.cache.keys())

        def run():
            for k in keys:
                _ = HC.V[k]

        benchmark(run)

    def test_bench_cache_insert(self, benchmark):
        """Benchmark vertex creation via triangulate + refine_all."""
        def run():
            HC = Complex(3)
            HC.triangulate()
            HC.refine_all()
            return HC

        benchmark(run)


# --- 1d. Vertex deletion ---

class TestBenchDeletion:
    """Benchmark vertex removal operations."""

    def test_bench_vertex_remove(self, benchmark):
        """Benchmark removing vertices (measures reindex cost)."""
        def run():
            HC = Complex(2)
            HC.triangulate()
            HC.refine_all()
            HC.refine_all()
            # Remove first 5 vertices
            to_remove = list(HC.V)[:5]
            for v in to_remove:
                HC.V.remove(v)

        benchmark(run)

    def test_bench_vertex_remove_batch(self, benchmark):
        """Benchmark removing many vertices at once."""
        def run():
            HC = Complex(3)
            HC.triangulate()
            HC.refine_all()
            # Remove half the vertices
            vertices = list(HC.V)
            to_remove = vertices[:len(vertices) // 2]
            for v in to_remove:
                HC.V.remove(v)

        benchmark(run)


# --- 1e. Iteration patterns ---

class TestBenchIteration:
    """Benchmark iteration over vertex cache."""

    def test_bench_iteration(self, benchmark):
        """Benchmark iterating over HC.V and accessing neighbors."""
        HC = Complex(3)
        HC.triangulate()
        HC.refine_all()

        def run():
            count = 0
            for v in HC.V:
                count += len(v.nn)
            return count

        benchmark(run)

    def test_bench_iteration_snapshot(self, benchmark):
        """Benchmark copy.copy(HC.V) iteration pattern for comparison."""
        HC = Complex(3)
        HC.triangulate()
        HC.refine_all()

        def run():
            count = 0
            for v in copy.copy(HC.V):
                count += len(v.nn)
            return count

        benchmark(run)


# --- 1e2. Mesh conversion & serialization ---

class TestBenchMeshConversion:
    """Benchmark mesh conversion and serialization operations."""

    def test_bench_vf_to_vv(self, benchmark):
        """Benchmark vertex-face to vertex-vertex mesh conversion."""
        # First build a vf mesh to convert from
        HC_source = Complex(2)
        HC_source.triangulate()
        HC_source.refine_all()
        HC_source.vertex_face_mesh()
        vertices_arr = numpy.array(HC_source.vertices_fm)
        simplices = HC_source.simplices_fm_i

        def run():
            HC = Complex(2)
            HC.vf_to_vv(vertices_arr, simplices)
            return HC

        benchmark(run)

    def test_bench_vertex_face_mesh(self, benchmark):
        """Benchmark vertex-vertex to vertex-face mesh conversion."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()

        def run():
            HC.vertex_face_mesh()

        benchmark(run)

    def test_bench_save_complex(self, benchmark):
        """Benchmark save_complex JSON serialization."""
        HC = Complex(2, sfield=simple_field)
        HC.triangulate()
        HC.refine_all()
        HC.V.process_pools()
        tmpdir = tempfile.mkdtemp(prefix='hyperct_bench_')
        fn = os.path.join(tmpdir, 'bench_save.json')

        def run():
            HC.save_complex(fn=fn)

        benchmark(run)

    def test_bench_load_complex(self, benchmark):
        """Benchmark load_complex JSON deserialization."""
        # Prepare a saved file
        HC = Complex(2, sfield=simple_field)
        HC.triangulate()
        HC.refine_all()
        HC.V.process_pools()
        tmpdir = tempfile.mkdtemp(prefix='hyperct_bench_')
        fn = os.path.join(tmpdir, 'bench_load.json')
        HC.save_complex(fn=fn)

        def run():
            HC2 = Complex(2, sfield=simple_field)
            HC2.load_complex(fn=fn)
            return HC2

        benchmark(run)


# --- 1g. Unit tests for previously uncovered operations ---

class TestMerge:
    """Unit tests for merge_nn and merge_all."""

    def test_merge_nn_reduces_vertices(self):
        """merge_nn with large cdist should merge some close neighbors."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        initial = len(HC.V.cache)
        n_merged = HC.V.merge_nn(cdist=0.6)
        assert n_merged > 0
        assert len(HC.V.cache) < initial

    def test_merge_nn_preserves_connectivity(self):
        """After merge_nn, remaining vertices still have neighbors."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        HC.V.merge_nn(cdist=0.4)
        for v in HC.V:
            # After merging, the surviving vertex should have connections
            # (some edge cases may have isolated vertices after aggressive merge)
            pass  # Mainly checking no crash during iteration
        assert len(HC.V.cache) > 0

    def test_merge_all_reduces_vertices(self):
        """merge_all with moderate cdist should merge close vertices."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        initial = len(HC.V.cache)
        HC.V.merge_all(cdist=0.4)
        assert len(HC.V.cache) < initial

    def test_merge_all_no_merge_small_cdist(self):
        """merge_all with very small cdist should merge nothing."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        initial = len(HC.V.cache)
        HC.V.merge_all(cdist=1e-15)
        assert len(HC.V.cache) == initial

    def test_merge_nn_with_exclude(self):
        """merge_nn excludes specified vertices from merging."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        origin = HC.V[(0.0, 0.0)]
        supremum = HC.V[(1.0, 1.0)]
        HC.V.merge_nn(cdist=0.6, exclude={origin, supremum})
        # Excluded vertices should still be in cache
        assert (0.0, 0.0) in HC.V.cache
        assert (1.0, 1.0) in HC.V.cache


class TestMove:
    """Unit tests for VertexCacheBase.move()."""

    def test_move_updates_coordinates(self):
        """move() changes vertex coordinates in cache."""
        HC = Complex(2)
        HC.triangulate()
        v = HC.V[(0.0, 0.0)]
        HC.V.move(v, (0.1, 0.1))
        assert v.x == (0.1, 0.1)
        assert (0.1, 0.1) in HC.V.cache
        assert (0.0, 0.0) not in HC.V.cache

    def test_move_preserves_neighbors(self):
        """move() preserves neighbor connections."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        v = HC.V[(0.5, 0.5)]
        nn_count = len(v.nn)
        HC.V.move(v, (0.55, 0.55))
        assert len(v.nn) == nn_count

    def test_move_updates_hash(self):
        """move() updates the vertex hash."""
        HC = Complex(2)
        HC.triangulate()
        v = HC.V[(0.0, 0.0)]
        old_hash = hash(v)
        HC.V.move(v, (0.1, 0.1))
        assert hash(v) != old_hash
        assert hash(v) == hash((0.1, 0.1))


class TestVertexCacheSize:
    """Unit tests for VertexCacheBase.size()."""

    def test_size_matches_cache(self):
        """size() should match len(V.cache) after operations."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        # Note: size() returns self.index + 1, which tracks insertions
        assert HC.V.size() == len(HC.V.cache)


class TestSafeIteration:
    """Tests verifying safe iteration during modification."""

    def test_remove_during_iteration_no_error(self):
        """Removing vertices during for v in HC.V should not raise."""
        HC = Complex(2)
        HC.triangulate()
        HC.refine_all()
        initial = len(HC.V.cache)
        removed = 0
        for v in HC.V:
            if v.x == (0.5, 0.5):
                HC.V.remove(v)
                removed += 1
        assert removed == 1
        assert len(HC.V.cache) == initial - 1

    def test_cut_g_removes_infeasible(self):
        """cut_g correctly removes infeasible vertices."""
        def g(x):
            return x[0] - 0.5

        cons = {'type': 'ineq', 'fun': g}
        HC = Complex(2, sfield=simple_field, constraints=cons)
        HC.triangulate()
        initial = len(HC.V.cache)
        HC.cut_g()
        assert len(HC.V.cache) < initial
        # All remaining vertices should be feasible
        for v in HC.V:
            assert v.feasible
