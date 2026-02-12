"""
4D Dual Mesh Example: Barycentric vs Circumcentric
===================================================

Demonstrates compute_vd on a 4D complex with comparison of barycentric
(centroid) and circumcentric dual vertex placement.

- Barycentric: dual vertices at centroids of 4-simplices
- Circumcentric: dual vertices at circumcenters of 4-simplices

Note: Visualization is not available for 4D, but we can print statistics
and verify the dual mesh structure.
"""
from hyperct import Complex
from hyperct.ddg import compute_vd, e_star, v_star, d_area


def build_and_compute(method="barycentric"):
    """Build a 4D complex and compute its dual."""
    HC = Complex(4)
    HC.triangulate()
    # Optionally refine for more structure
    # HC.refine_all()

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

# Compute some operator values
sample_v = list(HC_bary.V)[0]
if len(sample_v.nn) > 0:
    neighbor = list(sample_v.nn)[0]
    e_star_val = e_star(sample_v, neighbor, HC_bary, dim=4)
    print(f"  Sample e_star value: {e_star_val:.6f}")

    v_star_vals = v_star(sample_v, neighbor, HC_bary, dim=4)
    print(f"  Sample v_star returned {len(v_star_vals[0])} values")

    area = d_area(sample_v)
    print(f"  Sample d_area: {area:.6f}")

# --- Circumcentric dual ---
HC_circ = build_and_compute("circumcentric")
print(
    f"\nCircumcentric: {len(HC_circ.V)} primal vertices, "
    f"{len(HC_circ.Vd)} dual vertices"
)

# Compute some operator values
sample_v = list(HC_circ.V)[0]
if len(sample_v.nn) > 0:
    neighbor = list(sample_v.nn)[0]
    e_star_val = e_star(sample_v, neighbor, HC_circ, dim=4)
    print(f"  Sample e_star value: {e_star_val:.6f}")

    v_star_vals = v_star(sample_v, neighbor, HC_circ, dim=4)
    print(f"  Sample v_star returned {len(v_star_vals[0])} values")

    area = d_area(sample_v)
    print(f"  Sample d_area: {area:.6f}")

print("\n4D dual mesh computation completed successfully!")
print("For higher-dimensional duals, use HC.V and HC.Vd to access")
print("primal and dual vertex structures programmatically.")
