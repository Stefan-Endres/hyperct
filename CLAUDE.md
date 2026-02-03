# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`hyperct` is a Python library for generating low-memory simplicial complex structures via hypercube triangulations and sub-triangulations. It enables efficient triangulation of high-dimensional domains (up to 10D+) with support for:
- Symmetry constraints to reduce complexity by O(n!)
- Scalar and vector field associations
- Inequality constraints for non-convex domains
- Multiple refinement strategies

## Commands

### Installation
```bash
python setup.py install
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest hyperct/tests/test__hyperct.py

# Run specific test function
pytest hyperct/tests/test__hyperct.py::TestClass::test_1_1_2D_cube_init

# Run with coverage
pytest --cov=hyperct
```

### Development
The project uses standard Python setuptools. Install in development mode:
```bash
pip install -e .
```

## Architecture

### Core Components

The library is structured around several key abstractions:

**1. Complex (`hyperct/_complex.py`)**
- Main entry point and orchestrator (3600+ lines)
- `Complex` class manages the entire simplicial complex structure
- Two essential attributes:
  - `Complex.V`: Vertex cache storing all vertices and their connections
  - `Complex.H`: Storage structure of vertex groups (Cell, Simplex objects)
- Key methods:
  - `triangulate()`: Initial domain triangulation
  - `split_generation()`: Subdivide cells in current generation
  - `refine()`: Add n vertices via local refinement
  - `refine_local_space()`: Core refinement algorithm for vector partitions

**2. Vertex System (`hyperct/_vertex.py`)**
- Base class hierarchy for vertices:
  - `VertexBase` (ABC): Core vertex interface
  - `VertexCube`: Pure simplicial complex vertices (no fields)
  - `VertexScalarField`: Vertices with associated scalar field f: R^n → R
- Two cache types:
  - `VertexCacheIndex`: Basic vertex cache
  - `VertexCacheField`: Extended cache with field computation support
- Vertices maintain neighbor connections via `nn` attribute (set of connected vertices)

**3. Field System (`hyperct/_field.py`)**
- `FieldCache`: Base class for field computations with feasibility checking
- `ScalarFieldCache`: Associates scalar fields with geometry
- Handles constraint evaluation (g_cons functions)

**4. Vertex Groups (`hyperct/_vertex_group.py`)**
- `VertexGroup`: Base container for vertex collections
- `Cell`: Hypercube-symmetric cells with origin/supremum
- `Simplex`: Symmetry-constrained simplexes with generation cycles

**5. Simplex Objects (`hyperct/_simplex.py`)**
- `SimplexBase`: Abstract base for simplex structures
- `SimplexOrdered`: Ordered simplex implementation

### Key Architectural Patterns

**Cyclic Product Structure**
The triangulation uses group theory concepts: hypercubes are viewed as cartesian products of C2 cyclic groups (H = C2 × C2 × ... × C2). This enables efficient parallel connection of vertices using cosets.

**Generation-Based Refinement**
- Initial triangulation creates generation 0
- `split_generation()` subdivides all cells in current generation
- Each generation stored in `Complex.H[gen]`
- Centroids connect new subcubes during refinement

**Lazy Evaluation**
The `refine()` method uses generators and exception handling to lazily build triangulations up to a target vertex count, switching between initial triangulation and local refinement as needed.

**Symmetry Reduction**
When variables are symmetric, the complexity reduces significantly. Symmetry is specified as a dict/list mapping variables to their symmetry groups, allowing bounds to be shared.

## Test Data Structure

Test data is stored in `hyperct/tests/test_data/` as JSON files with naming convention:
```
test_{n+1}_{gen+1}_{n}D_{cube|symm}_gen_{gen}[_s_{symmetry}].json
```
Examples:
- `test_2_2_1D_cube_gen_1.json`: 1D cube, generation 1
- `test_4_3_3D_symm_gen_2_s_000.json`: 3D with symmetry [0,0,0], generation 2

Tests validate triangulation by:
1. Generating reference complex
2. Loading expected vertex/edge counts from JSON
3. Comparing actual vs expected structure

## Important Notes

### Constraint System
- Inequality constraints: `g(x) <= 0` (must be non-negative when inverted internally)
- Constraints defined as dicts with keys: `type`, `fun`, `args`
- Feasibility checking cuts away infeasible regions from convex domain

### Domain and Bounds
- Default domain: unit hypercube [0, 1]^dim
- Custom domains via `domain` parameter (list of (lower, upper) tuples)
- Non-convex domains created by constraint cutting

### Optional Dependencies
- **matplotlib**: Required for plotting functions (plots.py, _plotting.py)
- **clifford**: For discrete exterior calculus (currently disabled in imports)
- Python 2.7 and 3.5+ support (functools.lru_cache backport in _misc.py)

## Working with the Code

### Creating a Complex
```python
from hyperct import Complex

# Basic 2D complex
H = Complex(dim=2)
H.triangulate()  # Build initial triangulation

# With scalar field and constraints
H = Complex(dim=2,
            sfield=my_function,
            sfield_args=(),
            constraints={'type': 'ineq', 'fun': g_cons})
```

### Refinement Strategies
- `refine(n)`: Add n vertices via adaptive refinement
- `refine_all()`: Refine all existing structure
- `split_generation()`: Subdivide current generation completely

### Accessing Structure
- `H.V.cache`: Dict of all vertices indexed by coordinate tuple
  - `H.V[vertex_tuple]`: Access specific vertex
- Iterate vertices: `for v in H.V: ...`
- Vertex neighbors: `v.nn` (set of connected vertices)

### Visualization
See `hyperct/plots.py` for examples of triangulation visualization in 2D/3D using matplotlib.
