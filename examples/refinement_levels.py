"""
Comparing refinement levels and their effect on mesh resolution.

Shows how successive calls to refine_all() or split_generation()
subdivide the simplicial complex, increasing vertex density.
Also demonstrates the lazy refine(n) method that adds exactly n vertices.
"""
import numpy as np
from hyperct import Complex

# Simple quadratic bowl
def bowl(x):
    return np.sum(np.array(x)**2)


# --- Progressive refinement with refine_all ---
print("=== Progressive refinement (refine_all) ===")
H = Complex(2, domain=[(-5.0, 5.0), (-5.0, 5.0)], sfield=bowl)
H.triangulate()
print(f"Generation 0 (triangulate): {len(H.V.cache)} vertices")

for gen in range(1, 5):
    H.refine_all()
    print(f"Generation {gen}: {len(H.V.cache)} vertices")


# --- Lazy refinement with refine(n) ---
print("\n=== Lazy refinement (refine to target vertex count) ===")
H2 = Complex(2, domain=[(-5.0, 5.0), (-5.0, 5.0)], sfield=bowl)
H2.refine(n=20)
print(f"After refine(20): {len(H2.V.cache)} vertices")

H2.refine(n=50)
print(f"After refine(50): {len(H2.V.cache)} vertices")


# --- High-dimensional triangulation ---
print("\n=== High-dimensional triangulation ===")
for dim in [2, 3, 4, 5]:
    H_nd = Complex(dim)
    H_nd.triangulate()
    n_init = len(H_nd.V.cache)
    H_nd.refine_all()
    n_ref = len(H_nd.V.cache)
    print(f"  {dim}D: {n_init} vertices (init) -> {n_ref} vertices (1 refinement)")
