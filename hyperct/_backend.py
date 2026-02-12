"""
Batch computation backends for hyperct.

Provides a unified interface for batch field evaluation, constraint checking,
distance computation, and determinant calculation. Backends:

- ``NumpyBackend``: Vectorized numpy (default, always available)
- ``MultiprocessingBackend``: CPU parallelism via ``multiprocessing.Pool``
- ``TorchBackend``: PyTorch tensors — auto-uses CUDA when available,
  falls back to CPU (requires ``torch``)

Usage::

    from hyperct._backend import get_backend

    backend = get_backend("numpy")      # explicit
    backend = get_backend("gpu")        # auto-detect GPU, fallback to numpy
    backend = get_backend("torch")      # PyTorch (GPU if available, else CPU)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any, Callable, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Protocol (structural typing interface)
# ---------------------------------------------------------------------------
@runtime_checkable
class BatchBackend(Protocol):
    """Protocol for batch computation backends."""

    name: str

    def batch_field_eval(
        self,
        coords: np.ndarray,
        field_fn: Callable,
        field_args: tuple,
    ) -> np.ndarray:
        """Evaluate a scalar field at N points.

        Parameters
        ----------
        coords : ndarray of shape (N, dim)
        field_fn : callable, f(x, *args) -> scalar for a single point
        field_args : tuple of extra arguments

        Returns
        -------
        ndarray of shape (N,)
        """
        ...

    def batch_feasibility(
        self,
        coords: np.ndarray,
        g_cons: Sequence[Callable],
        g_cons_args: Sequence[tuple],
    ) -> np.ndarray:
        """Check constraint feasibility at N points.

        Parameters
        ----------
        coords : ndarray of shape (N, dim)
        g_cons : sequence of constraint functions g(x, *args) -> value
        g_cons_args : sequence of argument tuples

        Returns
        -------
        boolean ndarray of shape (N,), True = feasible
        """
        ...

    def batch_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix.

        Parameters
        ----------
        coords : ndarray of shape (N, dim)

        Returns
        -------
        ndarray of shape (N, N)
        """
        ...

    def batch_determinants(self, matrices: np.ndarray) -> np.ndarray:
        """Compute determinants of a batch of square matrices.

        Parameters
        ----------
        matrices : ndarray of shape (K, M, M)

        Returns
        -------
        ndarray of shape (K,)
        """
        ...

    def batch_dual_positions(
        self,
        simplices: np.ndarray,
        strategy_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Compute dual vertex positions for a batch of simplices.

        Parameters
        ----------
        simplices : ndarray of shape (N, k+1, dim)
            N simplices, each with k+1 vertices in dim-dimensional space.
        strategy_fn : callable
            Maps (k+1, dim) array -> (dim,) array for single simplex.

        Returns
        -------
        ndarray of shape (N, dim)
            Dual vertex positions.
        """
        ...


# ---------------------------------------------------------------------------
# Numpy backend (always available)
# ---------------------------------------------------------------------------
class NumpyBackend:
    """Vectorized numpy backend. Default for all installations."""

    name = "numpy"

    def batch_field_eval(
        self,
        coords: np.ndarray,
        field_fn: Callable,
        field_args: tuple,
    ) -> np.ndarray:
        n = coords.shape[0]
        result = np.empty(n)
        for i in range(n):
            try:
                val = field_fn(coords[i], *field_args)
            except Exception:
                val = np.inf
            if np.isnan(val):
                val = np.inf
            result[i] = val
        return result

    def batch_feasibility(
        self,
        coords: np.ndarray,
        g_cons: Sequence[Callable],
        g_cons_args: Sequence[tuple],
    ) -> np.ndarray:
        n = coords.shape[0]
        feasible = np.ones(n, dtype=bool)
        for g, args in zip(g_cons, g_cons_args):
            for i in range(n):
                if not feasible[i]:
                    continue
                if np.any(g(coords[i], *args) < 0.0):
                    feasible[i] = False
        return feasible

    def batch_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def batch_determinants(self, matrices: np.ndarray) -> np.ndarray:
        return np.linalg.det(matrices)

    def batch_dual_positions(
        self,
        simplices: np.ndarray,
        strategy_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        return np.array([strategy_fn(s) for s in simplices])


# ---------------------------------------------------------------------------
# Multiprocessing backend (wraps existing pool pattern)
# ---------------------------------------------------------------------------
class MultiprocessingBackend:
    """CPU-parallel backend using multiprocessing.Pool."""

    name = "multiprocessing"

    def __init__(self, workers: int = 2):
        import multiprocessing as mp
        self.workers = workers
        self.pool = mp.Pool(processes=workers)

    def _get_chunksize(self, n: int) -> int:
        return max(1, n // (4 * self.workers))

    def batch_field_eval(
        self,
        coords: np.ndarray,
        field_fn: Callable,
        field_args: tuple,
    ) -> np.ndarray:
        wrapper = _FieldEvalWorker(field_fn, field_args)
        chunksize = self._get_chunksize(coords.shape[0])
        results = self.pool.map(wrapper, [coords[i] for i in range(coords.shape[0])],
                                chunksize=chunksize)
        return np.array(results)

    def batch_feasibility(
        self,
        coords: np.ndarray,
        g_cons: Sequence[Callable],
        g_cons_args: Sequence[tuple],
    ) -> np.ndarray:
        wrapper = _FeasibilityWorker(g_cons, g_cons_args)
        chunksize = self._get_chunksize(coords.shape[0])
        results = self.pool.map(wrapper, [coords[i] for i in range(coords.shape[0])],
                                chunksize=chunksize)
        return np.array(results, dtype=bool)

    def batch_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def batch_determinants(self, matrices: np.ndarray) -> np.ndarray:
        return np.linalg.det(matrices)

    def batch_dual_positions(
        self,
        simplices: np.ndarray,
        strategy_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        wrapper = _DualPositionWorker(strategy_fn)
        chunksize = self._get_chunksize(simplices.shape[0])
        results = self.pool.map(
            wrapper, [simplices[i] for i in range(len(simplices))], chunksize=chunksize
        )
        return np.array(results)

    def terminate(self):
        self.pool.terminate()

    def __del__(self):
        try:
            self.pool.terminate()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('pool', None)
        return state


class _FieldEvalWorker:
    """Pickleable callable for multiprocessing field evaluation."""
    def __init__(self, field_fn: Callable, field_args: tuple):
        self.field_fn = field_fn
        self.field_args = field_args

    def __call__(self, x: np.ndarray) -> float:
        try:
            val = self.field_fn(x, *self.field_args)
        except Exception:
            val = np.inf
        if np.isnan(val):
            val = np.inf
        return val


class _FeasibilityWorker:
    """Pickleable callable for multiprocessing feasibility checking."""
    def __init__(self, g_cons: Sequence[Callable], g_cons_args: Sequence[tuple]):
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def __call__(self, x: np.ndarray) -> bool:
        for g, args in zip(self.g_cons, self.g_cons_args):
            if np.any(g(x, *args) < 0.0):
                return False
        return True


class _DualPositionWorker:
    """Pickleable callable for multiprocessing dual position computation."""
    def __init__(self, strategy_fn: Callable[[np.ndarray], np.ndarray]):
        self.strategy_fn = strategy_fn

    def __call__(self, simplex: np.ndarray) -> np.ndarray:
        return self.strategy_fn(simplex)


# ---------------------------------------------------------------------------
# PyTorch backend (optional GPU, with CPU fallback)
# ---------------------------------------------------------------------------
class TorchBackend:
    """PyTorch backend. Uses CUDA when available, otherwise CPU tensors.

    This is the recommended GPU backend for conda environments with PyTorch
    installed.  Even without a working CUDA driver, PyTorch's vectorized
    tensor operations on CPU are competitive with plain numpy for batched
    workloads.

    Requires ``torch`` to be installed.
    """

    name = "torch"

    def __init__(self):
        import torch
        self.torch = torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @property
    def has_cuda(self) -> bool:
        return self.device.type == "cuda"

    def batch_field_eval(
        self,
        coords: np.ndarray,
        field_fn: Callable,
        field_args: tuple,
    ) -> np.ndarray:
        torch = self.torch
        # Try a fully-vectorized call first (works if field_fn is
        # written with numpy/torch broadcasting).
        try:
            coords_t = torch.as_tensor(coords, dtype=torch.float64,
                                       device=self.device)
            result = field_fn(coords_t, *field_args)
            result = torch.as_tensor(result, device=self.device).cpu().numpy().ravel()
            result[np.isnan(result)] = np.inf
            return result
        except Exception:
            pass

        # Fallback: row-by-row on CPU (for user field functions that
        # don't support tensor input or batched (N, dim) arrays).
        n = coords.shape[0]
        result = np.empty(n)
        for i in range(n):
            try:
                val = field_fn(coords[i], *field_args)
            except Exception:
                val = np.inf
            if np.isnan(val):
                val = np.inf
            result[i] = val
        return result

    def batch_feasibility(
        self,
        coords: np.ndarray,
        g_cons: Sequence[Callable],
        g_cons_args: Sequence[tuple],
    ) -> np.ndarray:
        torch = self.torch
        n = coords.shape[0]
        feasible = np.ones(n, dtype=bool)
        for g, args in zip(g_cons, g_cons_args):
            try:
                coords_t = torch.as_tensor(coords, dtype=torch.float64,
                                           device=self.device)
                g_vals = g(coords_t, *args)
                g_vals_np = torch.as_tensor(g_vals, device=self.device).cpu().numpy()
                if g_vals_np.ndim == 1:
                    feasible &= ~(g_vals_np < 0.0)
                else:
                    feasible &= ~np.any(g_vals_np < 0.0, axis=1)
            except Exception:
                for i in range(n):
                    if not feasible[i]:
                        continue
                    if np.any(g(coords[i], *args) < 0.0):
                        feasible[i] = False
        return feasible

    def batch_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        torch = self.torch
        coords_t = torch.as_tensor(coords, dtype=torch.float64,
                                   device=self.device)
        dist = torch.cdist(coords_t.unsqueeze(0), coords_t.unsqueeze(0)).squeeze(0)
        return dist.cpu().numpy()

    def batch_determinants(self, matrices: np.ndarray) -> np.ndarray:
        torch = self.torch
        mat_t = torch.as_tensor(matrices, dtype=torch.float64,
                                device=self.device)
        dets = torch.linalg.det(mat_t)
        return dets.cpu().numpy()

    def batch_dual_positions(
        self,
        simplices: np.ndarray,
        strategy_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        torch = self.torch
        try:
            # Try vectorized: for barycenter, this is just mean
            simplices_t = torch.as_tensor(
                simplices, dtype=torch.float64, device=self.device
            )
            # Try applying strategy to all at once
            result = strategy_fn(simplices_t)
            return result.cpu().numpy()
        except Exception:
            # Fallback to per-simplex
            return np.array([strategy_fn(s) for s in simplices])


# ---------------------------------------------------------------------------
# Backend registry and auto-detection
# ---------------------------------------------------------------------------
_BACKENDS: dict[str, type] = {
    "numpy": NumpyBackend,
    "multiprocessing": MultiprocessingBackend,
    "torch": TorchBackend,
}


def _detect_gpu_backend() -> BatchBackend:
    """Auto-detect the best available GPU backend, falling back to numpy.

    Preference: PyTorch+CUDA > PyTorch+CPU > numpy.
    """
    try:
        backend = TorchBackend()
        if backend.has_cuda:
            return backend
    except (ImportError, Exception):
        pass
    # PyTorch on CPU is still vectorized and better than serial numpy
    try:
        return TorchBackend()
    except (ImportError, Exception):
        pass
    return NumpyBackend()


def get_backend(name: str | None = None, **kwargs: Any) -> BatchBackend:
    """Get a computation backend by name.

    Parameters
    ----------
    name : str or None
        Backend name: ``"numpy"``, ``"multiprocessing"``, ``"torch"``,
        ``"gpu"`` (auto-detect), or ``None`` (numpy default).
    **kwargs
        Passed to the backend constructor (e.g. ``workers=4`` for
        multiprocessing).

    Returns
    -------
    BatchBackend
        An instance satisfying the :class:`BatchBackend` protocol.
    """
    if name is None or name == "numpy":
        return NumpyBackend()
    if name == "gpu":
        return _detect_gpu_backend()
    if name == "multiprocessing":
        return MultiprocessingBackend(**kwargs)
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {list(_BACKENDS.keys())} or 'gpu'"
        )
    return _BACKENDS[name](**kwargs)
