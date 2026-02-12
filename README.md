# hyperct

Low memory simplicial complex structures via hypercube triangulations and sub-triangulations.

## Features

- Triangulates high-dimensional domains (10D+)
- Symmetry constraints to reduce complexity by O(n!)
- Scalar and vector field associations
- Inequality constraints for non-convex domains
- Multiple refinement strategies (local, global, generation-based)
- Optional GPU acceleration via PyTorch

## Installation

```bash
pip install hyperct
```

### Optional dependencies

```bash
# GPU support (PyTorch backend, auto-detects CUDA)
pip install hyperct[gpu]

# Plotting support (matplotlib)
pip install hyperct[plotting]

# Development and testing
pip install hyperct[dev]

# Multiple extras
pip install hyperct[gpu,plotting]
```

## Quick start

```python
from hyperct import Complex

# Triangulate a 3D unit hypercube
C = Complex(3)
C.triangulate()
C.split_generation()
C.split_generation()
```

### GPU-accelerated field evaluation

```python
from hyperct import Complex

def my_field(x):
    return sum(xi**2 for xi in x)

C = Complex(3, func=my_field, backend='gpu')
C.triangulate()
C.split_generation()
```

## Requirements

- Python >= 3.7
- NumPy >= 1.16.0
- Optional: PyTorch >= 2.0 (GPU backend)
- Optional: matplotlib >= 3.0 (plotting)

## License

MIT
