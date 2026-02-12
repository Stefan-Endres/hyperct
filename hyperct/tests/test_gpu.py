"""
Backend and GPU tests for hyperct.

PyTorch tests run on CPU when torch is installed; CUDA-specific tests are
skipped when no GPU driver is available.

Run all backend tests:            pytest hyperct/tests/test_gpu.py
Run GPU-only tests:               pytest hyperct/tests/test_gpu.py -m gpu
Skip GPU tests:                   pytest hyperct/tests/test_gpu.py -m "not gpu"
"""
import numpy as np
import numpy.testing as npt
import pytest

from hyperct._backend import (
    NumpyBackend,
    MultiprocessingBackend,
    TorchBackend,
    get_backend,
)
from hyperct._complex import Complex

# ---------------------------------------------------------------------------
# Library / hardware availability detection
# ---------------------------------------------------------------------------
try:
    import torch as _torch  # noqa: F401
    HAS_TORCH = True
    HAS_TORCH_CUDA = _torch.cuda.is_available()
except (ImportError, Exception):
    HAS_TORCH = False
    HAS_TORCH_CUDA = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed"
)
requires_torch_cuda = pytest.mark.skipif(
    not HAS_TORCH_CUDA, reason="PyTorch CUDA not available (no driver or no GPU)"
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------
def _simple_field(x, *args):
    """Test field: f(x) = sum((x_i - 1)^2)."""
    return float(np.sum((x - 1.0) ** 2))


def _constraint_inside_ball(x, *args):
    """Constraint g(x) = 0.5 - ||x||^2.  Feasible when ||x|| < sqrt(0.5)."""
    return 0.5 - np.sum(x ** 2)


def _make_test_coords(n=50, dim=3, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))


def _make_test_matrices(k=20, m=4, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((k, m, m))


# ---------------------------------------------------------------------------
# TestNumpyBackend — always runs
# ---------------------------------------------------------------------------
class TestNumpyBackend:
    """Validates the vectorized NumpyBackend against direct computation."""

    def setup_method(self):
        self.backend = NumpyBackend()

    def test_batch_field_eval(self):
        coords = _make_test_coords(n=30, dim=3)
        result = self.backend.batch_field_eval(coords, _simple_field, ())
        expected = np.array([_simple_field(coords[i]) for i in range(30)])
        npt.assert_allclose(result, expected)

    def test_batch_field_eval_nan_handling(self):
        def bad_field(x):
            return np.nan
        coords = _make_test_coords(n=5, dim=2)
        result = self.backend.batch_field_eval(coords, bad_field, ())
        assert np.all(np.isinf(result))

    def test_batch_field_eval_exception_handling(self):
        def failing_field(x):
            raise ValueError("boom")
        coords = _make_test_coords(n=5, dim=2)
        result = self.backend.batch_field_eval(coords, failing_field, ())
        assert np.all(np.isinf(result))

    def test_batch_feasibility(self):
        coords = _make_test_coords(n=40, dim=3)
        g_cons = (_constraint_inside_ball,)
        g_cons_args = ((),)
        result = self.backend.batch_feasibility(coords, g_cons, g_cons_args)
        expected = np.array([
            not np.any(_constraint_inside_ball(coords[i]) < 0.0)
            for i in range(40)
        ])
        npt.assert_array_equal(result, expected)

    def test_batch_distance_matrix(self):
        coords = _make_test_coords(n=20, dim=3)
        dist = self.backend.batch_distance_matrix(coords)
        assert dist.shape == (20, 20)
        npt.assert_allclose(np.diag(dist), 0.0, atol=1e-15)
        npt.assert_allclose(dist, dist.T)
        expected_01 = np.linalg.norm(coords[0] - coords[1])
        npt.assert_allclose(dist[0, 1], expected_01)

    def test_batch_determinants(self):
        matrices = _make_test_matrices(k=15, m=4)
        result = self.backend.batch_determinants(matrices)
        expected = np.array([np.linalg.det(matrices[i]) for i in range(15)])
        npt.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestMultiprocessingBackend — always runs
# ---------------------------------------------------------------------------
class TestMultiprocessingBackend:
    """Validates MultiprocessingBackend matches NumpyBackend results."""

    def setup_method(self):
        self.mp_backend = MultiprocessingBackend(workers=2)
        self.np_backend = NumpyBackend()

    def teardown_method(self):
        self.mp_backend.terminate()

    def test_field_eval_matches_numpy(self):
        coords = _make_test_coords(n=30, dim=3)
        mp_result = self.mp_backend.batch_field_eval(coords, _simple_field, ())
        np_result = self.np_backend.batch_field_eval(coords, _simple_field, ())
        npt.assert_allclose(mp_result, np_result)

    def test_feasibility_matches_numpy(self):
        coords = _make_test_coords(n=30, dim=3)
        g_cons = (_constraint_inside_ball,)
        g_cons_args = ((),)
        mp_result = self.mp_backend.batch_feasibility(coords, g_cons, g_cons_args)
        np_result = self.np_backend.batch_feasibility(coords, g_cons, g_cons_args)
        npt.assert_array_equal(mp_result, np_result)


# ---------------------------------------------------------------------------
# TestTorchBackend — runs when PyTorch is installed (CPU or GPU)
# ---------------------------------------------------------------------------
@requires_torch
class TestTorchBackend:
    """Validates TorchBackend matches NumpyBackend on all operations.

    Runs on CPU when no CUDA driver is loaded; automatically uses GPU when
    CUDA is available.
    """

    def setup_method(self):
        self.torch_backend = TorchBackend()
        self.np_backend = NumpyBackend()

    def test_device_detection(self):
        assert self.torch_backend.name == "torch"
        assert self.torch_backend.device.type in ("cpu", "cuda")

    def test_field_eval_matches_numpy(self):
        coords = _make_test_coords(n=50, dim=3)
        torch_result = self.torch_backend.batch_field_eval(
            coords, _simple_field, ()
        )
        np_result = self.np_backend.batch_field_eval(coords, _simple_field, ())
        npt.assert_allclose(torch_result, np_result, rtol=1e-10)

    def test_field_eval_nan_handling(self):
        def bad_field(x):
            return np.nan
        coords = _make_test_coords(n=5, dim=2)
        result = self.torch_backend.batch_field_eval(coords, bad_field, ())
        assert np.all(np.isinf(result))

    def test_field_eval_exception_handling(self):
        def failing_field(x):
            raise ValueError("boom")
        coords = _make_test_coords(n=5, dim=2)
        result = self.torch_backend.batch_field_eval(coords, failing_field, ())
        assert np.all(np.isinf(result))

    def test_feasibility_matches_numpy(self):
        coords = _make_test_coords(n=50, dim=3)
        g_cons = (_constraint_inside_ball,)
        g_cons_args = ((),)
        torch_result = self.torch_backend.batch_feasibility(
            coords, g_cons, g_cons_args
        )
        np_result = self.np_backend.batch_feasibility(
            coords, g_cons, g_cons_args
        )
        npt.assert_array_equal(torch_result, np_result)

    def test_distance_matrix_matches_numpy(self):
        coords = _make_test_coords(n=30, dim=4)
        torch_result = self.torch_backend.batch_distance_matrix(coords)
        np_result = self.np_backend.batch_distance_matrix(coords)
        # torch.cdist uses a different algorithm than manual broadcasting,
        # so we allow slightly looser tolerance (typically <1e-7 abs diff)
        npt.assert_allclose(torch_result, np_result, atol=1e-7, rtol=1e-10)

    def test_distance_matrix_properties(self):
        coords = _make_test_coords(n=20, dim=3)
        dist = self.torch_backend.batch_distance_matrix(coords)
        assert dist.shape == (20, 20)
        npt.assert_allclose(np.diag(dist), 0.0, atol=1e-14)
        npt.assert_allclose(dist, dist.T, atol=1e-14)

    def test_determinants_match_numpy(self):
        matrices = _make_test_matrices(k=20, m=4)
        torch_result = self.torch_backend.batch_determinants(matrices)
        np_result = self.np_backend.batch_determinants(matrices)
        npt.assert_allclose(torch_result, np_result, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestTorchBackendIntegration — full Complex workflow with torch backend
# ---------------------------------------------------------------------------
@requires_torch
class TestTorchBackendIntegration:
    """Full Complex workflow with backend='torch' vs serial reference."""

    def test_scalar_field_2d(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]

        HC_ref = Complex(2, domain=bounds, sfield=_simple_field)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_torch = Complex(2, domain=bounds, sfield=_simple_field,
                           backend="torch")
        HC_torch.triangulate()
        HC_torch.refine_all()
        HC_torch.V.process_pools()

        assert len(HC_ref.V) == len(HC_torch.V)
        for v_ref in HC_ref.V:
            v_test = HC_torch.V[v_ref.x]
            npt.assert_allclose(v_ref.f, v_test.f, rtol=1e-12,
                                err_msg=f"Field mismatch at {v_ref.x}")

    def test_constrained_2d(self):
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        constraints = ({'type': 'ineq', 'fun': _constraint_inside_ball},)

        HC_ref = Complex(2, domain=bounds, sfield=_simple_field,
                         constraints=constraints)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_torch = Complex(2, domain=bounds, sfield=_simple_field,
                           constraints=constraints, backend="torch")
        HC_torch.triangulate()
        HC_torch.refine_all()
        HC_torch.V.process_pools()

        assert len(HC_ref.V) == len(HC_torch.V)
        for v_ref in HC_ref.V:
            v_test = HC_torch.V[v_ref.x]
            assert v_ref.feasible == v_test.feasible
            npt.assert_allclose(v_ref.f, v_test.f, rtol=1e-12)

    def test_3d_multi_generation(self):
        HC_ref = Complex(3, sfield=_simple_field)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_torch = Complex(3, sfield=_simple_field, backend="torch")
        HC_torch.triangulate()
        HC_torch.refine_all()
        HC_torch.refine_all()
        HC_torch.V.process_pools()

        assert len(HC_ref.V) == len(HC_torch.V)
        for v_ref in HC_ref.V:
            v_test = HC_torch.V[v_ref.x]
            npt.assert_allclose(v_ref.f, v_test.f, rtol=1e-12)

    def test_4d_with_field(self):
        HC_ref = Complex(4, sfield=_simple_field)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_torch = Complex(4, sfield=_simple_field, backend="torch")
        HC_torch.triangulate()
        HC_torch.refine_all()
        HC_torch.V.process_pools()

        assert len(HC_ref.V) == len(HC_torch.V)
        for v_ref in HC_ref.V:
            v_test = HC_torch.V[v_ref.x]
            npt.assert_allclose(v_ref.f, v_test.f, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestTorchCUDA — only runs when CUDA is actually available
# ---------------------------------------------------------------------------
@requires_torch_cuda
@pytest.mark.gpu
class TestTorchCUDA:
    """Validates TorchBackend is actually on CUDA when a GPU is present."""

    def setup_method(self):
        self.backend = TorchBackend()

    def test_device_is_cuda(self):
        assert self.backend.device.type == "cuda"
        assert self.backend.has_cuda

    def test_field_eval_on_gpu(self):
        coords = _make_test_coords(n=100, dim=5)
        result = self.backend.batch_field_eval(coords, _simple_field, ())
        expected = np.array([_simple_field(coords[i]) for i in range(100)])
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_distance_matrix_on_gpu(self):
        coords = _make_test_coords(n=50, dim=5)
        dist = self.backend.batch_distance_matrix(coords)
        assert dist.shape == (50, 50)
        npt.assert_allclose(np.diag(dist), 0.0, atol=1e-6)

    def test_determinants_on_gpu(self):
        matrices = _make_test_matrices(k=30, m=5)
        result = self.backend.batch_determinants(matrices)
        expected = np.linalg.det(matrices)
        npt.assert_allclose(result, expected, rtol=1e-8)

    def test_complex_workflow_on_gpu(self):
        """Full 3D Complex workflow actually on CUDA."""
        HC_ref = Complex(3, sfield=_simple_field)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_gpu = Complex(3, sfield=_simple_field, backend="gpu")
        HC_gpu.triangulate()
        HC_gpu.refine_all()
        HC_gpu.V.process_pools()

        assert HC_gpu._backend.has_cuda
        assert len(HC_ref.V) == len(HC_gpu.V)
        for v_ref in HC_ref.V:
            v_gpu = HC_gpu.V[v_ref.x]
            npt.assert_allclose(v_ref.f, v_gpu.f, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestBackendFallback — always runs
# ---------------------------------------------------------------------------
class TestBackendFallback:
    """Verifies get_backend() gracefully handles various inputs."""

    def test_numpy_backend_default(self):
        backend = get_backend(None)
        assert isinstance(backend, NumpyBackend)

    def test_numpy_backend_explicit(self):
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)

    def test_gpu_auto_detect_returns_valid_backend(self):
        backend = get_backend("gpu")
        assert hasattr(backend, "batch_field_eval")
        assert hasattr(backend, "batch_feasibility")
        assert hasattr(backend, "batch_distance_matrix")
        assert hasattr(backend, "batch_determinants")

    @requires_torch
    def test_gpu_auto_detect_prefers_torch(self):
        backend = get_backend("gpu")
        assert isinstance(backend, TorchBackend)

    @requires_torch
    def test_torch_backend_explicit(self):
        backend = get_backend("torch")
        assert isinstance(backend, TorchBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent_backend")


# ---------------------------------------------------------------------------
# TestNumpyBackendIntegration — always runs, full Complex workflow
# ---------------------------------------------------------------------------
class TestNumpyBackendIntegration:
    """Full Complex workflow with backend='numpy' vs default (no backend)."""

    def test_scalar_field_2d(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]

        HC_ref = Complex(2, domain=bounds, sfield=_simple_field)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_test = Complex(2, domain=bounds, sfield=_simple_field,
                          backend="numpy")
        HC_test.triangulate()
        HC_test.refine_all()
        HC_test.V.process_pools()

        assert len(HC_ref.V) == len(HC_test.V)
        for v_ref in HC_ref.V:
            v_test = HC_test.V[v_ref.x]
            npt.assert_allclose(v_ref.f, v_test.f, rtol=1e-12)

    def test_constrained_2d(self):
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        constraints = ({'type': 'ineq', 'fun': _constraint_inside_ball},)

        HC_ref = Complex(2, domain=bounds, sfield=_simple_field,
                         constraints=constraints)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_test = Complex(2, domain=bounds, sfield=_simple_field,
                          constraints=constraints, backend="numpy")
        HC_test.triangulate()
        HC_test.refine_all()
        HC_test.V.process_pools()

        assert len(HC_ref.V) == len(HC_test.V)
        for v_ref in HC_ref.V:
            v_test = HC_test.V[v_ref.x]
            assert v_ref.feasible == v_test.feasible
            npt.assert_allclose(v_ref.f, v_test.f, rtol=1e-12)

    def test_no_field_backend_ignored(self):
        HC = Complex(2, backend="numpy")
        HC.triangulate()
        HC.refine_all()
        assert len(HC.V) > 0


# ---------------------------------------------------------------------------
# TestGPUAutoDetectIntegration — when torch is installed
# ---------------------------------------------------------------------------
@requires_torch
class TestGPUAutoDetectIntegration:
    """Full Complex workflow with backend='gpu' auto-detection."""

    def test_scalar_field_2d_gpu_autodetect(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]

        HC_ref = Complex(2, domain=bounds, sfield=_simple_field)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_gpu = Complex(2, domain=bounds, sfield=_simple_field,
                         backend="gpu")
        HC_gpu.triangulate()
        HC_gpu.refine_all()
        HC_gpu.V.process_pools()

        assert len(HC_ref.V) == len(HC_gpu.V)
        for v_ref in HC_ref.V:
            v_gpu = HC_gpu.V[v_ref.x]
            npt.assert_allclose(v_ref.f, v_gpu.f, rtol=1e-10)

    def test_constrained_3d_gpu_autodetect(self):
        bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        constraints = ({'type': 'ineq', 'fun': _constraint_inside_ball},)

        HC_ref = Complex(3, domain=bounds, sfield=_simple_field,
                         constraints=constraints)
        HC_ref.triangulate()
        HC_ref.refine_all()
        HC_ref.V.process_pools()

        HC_gpu = Complex(3, domain=bounds, sfield=_simple_field,
                         constraints=constraints, backend="gpu")
        HC_gpu.triangulate()
        HC_gpu.refine_all()
        HC_gpu.V.process_pools()

        assert len(HC_ref.V) == len(HC_gpu.V)
        for v_ref in HC_ref.V:
            v_gpu = HC_gpu.V[v_ref.x]
            assert v_ref.feasible == v_gpu.feasible
            npt.assert_allclose(v_ref.f, v_gpu.f, rtol=1e-10)
