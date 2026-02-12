"""
Basic hypercube triangulation in 2D and 3D.

Demonstrates how to create a simplicial complex from a hypercube domain,
triangulate it, and refine the mesh through successive generations.
"""
import numpy as np
from hyperct import Complex

# --- 2D unit cube triangulation ---
H = Complex(2)
H.triangulate()
print(f"After triangulate: {len(H.V.cache)} vertices")

# Refine the mesh by splitting all cells
H.refine_all()
print(f"After 1 refinement: {len(H.V.cache)} vertices")

H.refine_all()
print(f"After 2 refinements: {len(H.V.cache)} vertices")

# Print vertex coordinates and their neighbor counts
for v in H.V.cache:
    vertex = H.V[v]
    print(f"  vertex {v} has {len(vertex.nn)} neighbors")

# --- 3D cube with custom domain ---
H3 = Complex(3, domain=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])
H3.triangulate()
print(f"\n3D complex after triangulate: {len(H3.V.cache)} vertices")

H3.refine_all()
print(f"3D complex after 1 refinement: {len(H3.V.cache)} vertices")
