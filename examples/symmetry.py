"""
Exploiting variable symmetry to reduce triangulation complexity.

When a function has symmetric variables (f(x0, x1) == f(x1, x0)), the
symmetry parameter groups them so that only one canonical ordering is
triangulated. This reduces vertex count by up to O(n!).
"""
import numpy as np
from hyperct import Complex

# Fully symmetric function: sum of squares (f is invariant under permutation)
def sum_of_squares(x):
    return sum(xi**2 for xi in x)

# --- Without symmetry ---
H_full = Complex(3, domain=[(-2.0, 2.0)] * 3, sfield=sum_of_squares)
H_full.triangulate()
H_full.refine_all()
print(f"3D without symmetry: {len(H_full.V.cache)} vertices")

# --- With full symmetry (all 3 variables interchangeable) ---
# symmetry=[0, 0, 0] means all variables belong to the same symmetry group
H_sym = Complex(3,
                domain=[(-2.0, 2.0)] * 3,
                sfield=sum_of_squares,
                symmetry=[0, 0, 0])
H_sym.triangulate()
H_sym.refine_all()
print(f"3D with full symmetry [0,0,0]: {len(H_sym.V.cache)} vertices")

# --- With partial symmetry (x0 independent, x1/x2 symmetric) ---
# symmetry=[0, 1, 1] means x1 and x2 are interchangeable but x0 is not
H_partial = Complex(3,
                    domain=[(-2.0, 2.0)] * 3,
                    sfield=sum_of_squares,
                    symmetry=[0, 1, 1])
H_partial.triangulate()
H_partial.refine_all()
print(f"3D with partial symmetry [0,1,1]: {len(H_partial.V.cache)} vertices")
