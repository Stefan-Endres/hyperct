# hyperct

Low memory simplicial complex structures via hypercube triangulations and sub-triangulations.

## When to use hyperct

- **Dynamic meshes with changing topologies**: The library is based on efficiently tracking vertex-vertex connections only which avoids the computational expense of storing simplexes while still allowing many important operations that are applied on simplicial complex structures. Generation-based refinement lets you locally refine and split simplices without rebuilding the entire complex.
- **Computing homologies of very high-dimensional spaces**: the hypercube triangulation scales to hundreds of dimensions while symmetric space reduction cuts complexity by up to O(n!).
- **Working with connectivity of high-dimensional data**: the simplicial complex exposes vertex adjacency, cell membership, and dual mesh structure directly.

## When _not_ to use hyperct

If you need fast conventional computational geometry in low dimensions, dedicated libraries will outperform hyperct:

- **Delaunay / Voronoi tessellations**: [SciPy spatial](https://docs.scipy.org/doc/scipy/reference/spatial.html), [CGAL](https://www.cgal.org/) (via [scikit-geometry](https://github.com/scikit-geometry/scikit-geometry)), [Qhull](http://www.qhull.org/)
- **Unstructured mesh generation**: [Gmsh](https://gmsh.info/), [pygmsh](https://github.com/meshio/pygmsh), [TetGen](https://wias-berlin.de/software/tetgen/) (via [tetgen](https://github.com/pyvista/tetgen)), [MeshPy](https://github.com/inducer/meshpy)
- **Surface meshing & remeshing**: [PyMesh](https://pymesh.readthedocs.io/), [trimesh](https://trimsh.org/), [libigl](https://libigl.github.io/) (via [igl](https://github.com/libigl/libigl-python-bindings))
- **FEM / simulation meshes**: [FEniCS/DOLFINx](https://fenicsproject.org/), [Firedrake](https://www.firedrakeproject.org/)
- **Topological data analysis**: [GUDHI](https://gudhi.inria.fr/), [Ripser](https://github.com/Ripser/ripser), [giotto-tda](https://giotto-ai.github.io/gtda-docs/)

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

- **`hyperct[gpu]`** — PyTorch backend for GPU-accelerated field evaluation and batch vertex processing. Auto-detects CUDA; falls back to CPU when unavailable.
- **`hyperct[plotting]`** — Matplotlib-based visualization of triangulations, scalar fields, and dual meshes (1D/2D/3D).
- **`hyperct[dev]`** — pytest, coverage, and benchmark tooling.

```bash
pip install hyperct[gpu]
pip install hyperct[plotting]
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
