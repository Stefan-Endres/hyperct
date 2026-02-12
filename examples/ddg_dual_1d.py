"""
1D Dual Mesh Example
====================

Demonstrates compute_vd on a 1D complex. In 1D the dual vertex is always
the midpoint of each primary edge, regardless of the method chosen.

Both barycentric and circumcentric methods produce identical results in 1D.
"""
from hyperct import Complex
from hyperct.ddg import compute_vd
from hyperct.ddg.plot_dual import plot_dual_mesh_1D

# Build a 1D complex on [0, 1]
HC = Complex(1)
HC.triangulate()
HC.refine_all()

# Mark boundary vertices (endpoints of the interval)
dV = HC.boundary()
for v in dV:
    v.boundary = True

# Compute barycentric dual
compute_vd(HC, method="barycentric")
print(f"1D complex: {len(HC.V)} primal vertices, {len(HC.Vd)} dual vertices")
plot_dual_mesh_1D(HC, show=True)
