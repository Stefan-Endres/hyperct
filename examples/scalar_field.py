"""
Triangulating a domain with an associated scalar field.

Associates a scalar function f: R^n -> R with the complex so that each
vertex stores the field value. The directed graph structure then points
edges from high to low field values, revealing local minimizers.
"""
import numpy as np
from hyperct import Complex

# Eggholder-like function on [-100, 100]^2
def eggholder(x):
    return (-(x[1] + 47.0)
            * np.sin(np.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0))))
            - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0)))))


H = Complex(2,
            domain=[(-100.0, 100.0), (-100.0, 100.0)],
            sfield=eggholder)

H.triangulate()
H.refine_all()
H.refine_all()

# Compute field values at all vertices
H.V.process_pools()
print(f"Vertices: {len(H.V.cache)}")

# Each vertex now stores its field value
for v in list(H.V.cache)[:5]:
    vertex = H.V[v]
    print(f"  x={v}, f(x)={vertex.f:.4f}")

# Find minimizer vertices (all neighbors have higher field values)
minimizers = []
for v in H.V.cache:
    vertex = H.V[v]
    if vertex.minimiser():
        minimizers.append((v, vertex.f))

print(f"\nLocal minimizers found: {len(minimizers)}")
for coords, fval in sorted(minimizers, key=lambda t: t[1]):
    print(f"  x={coords}, f(x)={fval:.4f}")

# --- Plot (requires matplotlib) ---
try:
    import matplotlib
    matplotlib.use("Agg")
    H.plot_complex(show=False, save_fig=False)
    print("\nPlot generated successfully (use show=True to display).")
except ImportError:
    print("\nInstall matplotlib to visualize the complex.")
