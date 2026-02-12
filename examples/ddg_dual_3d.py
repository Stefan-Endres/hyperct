"""
3D Dual Mesh Example: Barycentric vs Circumcentric
===================================================

Demonstrates compute_vd on a 3D complex with side-by-side comparison
of barycentric (centroid) and circumcentric dual vertex placement.

- Barycentric: dual vertices at centroids of tetrahedra
- Circumcentric: dual vertices at circumcenters of tetrahedra
"""
import matplotlib.pyplot as plt

from hyperct import Complex
from hyperct.ddg import compute_vd
from hyperct.ddg.plot_dual import plot_dual_mesh_3D


def build_and_compute(method="barycentric"):
    """Build a 3D complex and compute its dual."""
    HC = Complex(3)
    HC.triangulate()
    #HC.refine_all()

    # Mark boundary vertices
    dV = HC.boundary()
    for v in dV:
        v.boundary = True

    compute_vd(HC, method=method)
    return HC


# --- Barycentric dual ---
HC_bary = build_and_compute("barycentric")
print(
    f"Barycentric: {len(HC_bary.V)} primal vertices, "
    f"{len(HC_bary.Vd)} dual vertices"
)

# --- Circumcentric dual ---
HC_circ = build_and_compute("circumcentric")
print(
    f"Circumcentric: {len(HC_circ.V)} primal vertices, "
    f"{len(HC_circ.Vd)} dual vertices"
)

# --- Side-by-side plot ---
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121, projection="3d")
plot_dual_mesh_3D(HC_bary, ax=ax1, show=False)
ax1.set_title("3D Barycentric Dual")

ax2 = fig.add_subplot(122, projection="3d")
plot_dual_mesh_3D(HC_circ, ax=ax2, show=False)
ax2.set_title("3D Circumcentric Dual")

plt.tight_layout()
plt.show()
