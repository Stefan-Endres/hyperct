"""
Batch computation backends for hyperct.

Provides a unified interface for batch field evaluation, constraint checking,
distance computation, and determinant calculation. Backends:

- ``NumpyBackend``: Vectorized numpy (default, always available)
- ``MultiprocessingBackend``: CPU parallelism via ``multiprocessing.Pool``
- ``CuPyBackend``: GPU via CuPy (requires ``cupy``)
- ``JaxBackend``: GPU/TPU via JAX with ``jit`` + ``vmap`` (requires ``jax``)

Usage::

    from hyperct._backend import get_backend

    backend = get_backend("numpy")      # explicit
    backend = get_backend("gpu")        # auto-detect GPU, fallback to numpy
    backend = get_backend("cupy")       # CuPy specifically
    backend = get_backend("jax")        # JAX specifically
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
        self._field_wrapper = None
        self._cons_wrapper = None

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
        # Fall back to numpy for distance matrix — parallelizing this
        # over multiprocessing has too much overhead for the gain.
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def batch_determinants(self, matrices: np.ndarray) -> np.ndarray:
        return np.linalg.det(matrices)

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


# ---------------------------------------------------------------------------
# CuPy backend (optional GPU)
# ---------------------------------------------------------------------------
class CuPyBackend:
    """GPU backend using CuPy. Requires ``cupy`` to be installed."""

    name = "cupy"

    def __init__(self):
        import cupy  # noqa: F811
        self.cp = cupy

    def batch_field_eval(
        self,
        coords: np.ndarray,
        field_fn: Callable,
        field_args: tuple,
    ) -> np.ndarray:
        # Field functions are user-defined Python callables — they may not be
        # CuPy-compatible.  We try to pass GPU arrays first; if that fails,
        # fall back to per-row CPU evaluation.
        cp = self.cp
        try:
            coords_gpu = cp.asarray(coords)
            # Try vectorized call with full (N, dim) array
            result = field_fn(coords_gpu, *field_args)
            result = cp.asnumpy(cp.asarray(result)).ravel()
            result[np.isnan(result)] = np.inf
            return result
        except Exception:
            # Fallback: evaluate row-by-row on CPU
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
        cp = self.cp
        n = coords.shape[0]
        feasible = np.ones(n, dtype=bool)
        for g, args in zip(g_cons, g_cons_args):
            try:
                coords_gpu = cp.asarray(coords)
                g_vals = g(coords_gpu, *args)
                g_vals = cp.asnumpy(cp.asarray(g_vals))
                # g(x) < 0 means infeasible
                if g_vals.ndim == 1:
                    feasible &= ~(g_vals < 0.0)
                else:
                    feasible &= ~np.any(g_vals < 0.0, axis=1)
            except Exception:
                # Fallback: per-point CPU evaluation
                for i in range(n):
                    if not feasible[i]:
                        continue
                    if np.any(g(coords[i], *args) < 0.0):
                        feasible[i] = False
        return feasible

    def batch_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        cp = self.cp
        coords_gpu = cp.asarray(coords)
        diff = coords_gpu[:, cp.newaxis, :] - coords_gpu[cp.newaxis, :, :]
        dist = cp.sqrt(cp.sum(diff ** 2, axis=-1))
        return cp.asnumpy(dist)

    def batch_determinants(self, matrices: np.ndarray) -> np.ndarray:
        cp = self.cp
        matrices_gpu = cp.asarray(matrices)
        dets = cp.linalg.det(matrices_gpu)
        return cp.asnumpy(dets)


# ---------------------------------------------------------------------------
# JAX backend (optional GPU/TPU)
# ---------------------------------------------------------------------------
class JaxBackend:
    """GPU/TPU backend using JAX. Requires ``jax`` to be installed."""

    name = "jax"

    def __init__(self):
        import jax
        import jax.numpy as jnp
        self.jax = jax
        self.jnp = jnp

    def batch_field_eval(
        self,
        coords: np.ndarray,
        field_fn: Callable,
        field_args: tuple,
    ) -> np.ndarray:
        jax = self.jax
        jnp = self.jnp
        try:
            # Try jax.vmap for automatic vectorization
            vmapped = jax.vmap(lambda x: field_fn(x, *field_args))
            coords_jax = jnp.asarray(coords)
            result = np.asarray(vmapped(coords_jax))
            result[np.isnan(result)] = np.inf
            return result
        except Exception:
            # Fallback: per-row CPU evaluation
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
        jnp = self.jnp
        n = coords.shape[0]
        feasible = np.ones(n, dtype=bool)
        for g, args in zip(g_cons, g_cons_args):
            try:
                vmapped_g = self.jax.vmap(lambda x: g(x, *args))
                coords_jax = jnp.asarray(coords)
                g_vals = np.asarray(vmapped_g(coords_jax))
                if g_vals.ndim == 1:
                    feasible &= ~(g_vals < 0.0)
                else:
                    feasible &= ~np.any(g_vals < 0.0, axis=1)
            except Exception:
                for i in range(n):
                    if not feasible[i]:
                        continue
                    if np.any(g(coords[i], *args) < 0.0):
                        feasible[i] = False
        return feasible

    def batch_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        jnp = self.jnp
        coords_jax = jnp.asarray(coords)
        diff = coords_jax[:, jnp.newaxis, :] - coords_jax[jnp.newaxis, :, :]
        dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
        return np.asarray(dist)

    def batch_determinants(self, matrices: np.ndarray) -> np.ndarray:
        jnp = self.jnp
        matrices_jax = jnp.asarray(matrices)
        dets = jnp.linalg.det(matrices_jax)
        return np.asarray(dets)


# ---------------------------------------------------------------------------
# Backend registry and auto-detection
# ---------------------------------------------------------------------------
_BACKENDS: dict[str, type] = {
    "numpy": NumpyBackend,
    "multiprocessing": MultiprocessingBackend,
    "cupy": CuPyBackend,
    "jax": JaxBackend,
}


def _detect_gpu_backend() -> BatchBackend:
    """Auto-detect the best available GPU backend, falling back to numpy."""
    try:
        return CuPyBackend()
    except (ImportError, Exception):
        pass
    try:
        return JaxBackend()
    except (ImportError, Exception):
        pass
    return NumpyBackend()


def get_backend(name: str | None = None, **kwargs: Any) -> BatchBackend:
    """Get a computation backend by name.

    Parameters
    ----------
    name : str or None
        Backend name: ``"numpy"``, ``"multiprocessing"``, ``"cupy"``,
        ``"jax"``, ``"gpu"`` (auto-detect), or ``None`` (numpy default).
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
