"""
Non-convex domain via inequality constraints.

Hyperct triangulates convex hypercubes by default. Inequality constraints
g(x) <= 0 carve away infeasible regions, creating non-convex or even
disconnected domains.
"""
import numpy as np
from hyperct import Complex

# Objective: Ursem01 function
def ursem01(x):
    return (-np.sin(2 * x[0] - 0.5 * np.pi)
            - 3.0 * np.cos(x[1])
            - 0.5 * x[0])

# Constraint 1: circular region (keep interior of circle)
#   (x0 - 5)^2 + (x1 - 5)^2 + 5*sqrt(x0*x1) - 29 <= 0
def g1(x):
    return (x[0] - 5)**2 + (x[1] - 5)**2 + 5 * np.sqrt(x[0] * x[1]) - 29

# Constraint 2: quartic boundary
#   (x0 - 6)^4 - x1 + 2 <= 0
def g2(x):
    return (x[0] - 6)**4 - x[1] + 2

constraints = [
    {"type": "ineq", "fun": g1},
    {"type": "ineq", "fun": g2},
]

H = Complex(2,
            domain=[(2.0, 10.0), (2.0, 10.0)],
            sfield=ursem01,
            constraints=constraints)

H.triangulate()
H.refine_all()
H.refine_all()

# Compute field values and constraint feasibility
H.V.process_pools()

# Count feasible vs total vertices
feasible = [v for v in H.V.cache if H.V[v].feasible]
print(f"Total vertices: {len(H.V.cache)}")
print(f"Feasible vertices: {len(feasible)}")

# Show minimizers in the constrained domain
for v in H.V.cache:
    vertex = H.V[v]
    if hasattr(vertex, "minimiser") and vertex.minimiser():
        print(f"  minimizer at x={v}, f(x)={vertex.f:.4f}")
