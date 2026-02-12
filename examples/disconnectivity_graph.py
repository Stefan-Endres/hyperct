"""
Disconnectivity Graph Visualization
====================================

Demonstrates building a disconnectivity graph (barrier tree) from
a hyperct Complex with a scalar field containing multiple local minima.

The disconnectivity graph shows the hierarchical structure of energy
basins and the barriers separating them, similar to energy landscape
visualizations used in molecular chemistry and global optimization.
"""
import numpy as np
import matplotlib.pyplot as plt

from hyperct import Complex
from hyperct._vis_disc import DisconnectivityGraph, database_from_complex


# Define a scalar field with multiple basins
def landscape(x):
    """Multi-basin landscape: high-frequency sin wave plus shallow quadratic."""
    return 3 * np.sin(5 * x[0]) + 0.5 * (x[0]**2 + x[1]**2)


# Build and refine the complex
HC = Complex(2, domain=[(-3, 3), (-3, 3)], sfield=landscape)
HC.triangulate()
HC.refine_all()
HC.refine_all()
HC.refine_all()


# Compute field values at all vertices
HC.V.process_pools()
print(f"Vertices: {len(HC.V.cache)}")

# Find local minima
minimizers = [v for v in HC.V.cache if HC.V[v].minimiser()]
print(f"Local minima found: {len(minimizers)}\n")

HC.plot_complex()
# --- Method 1: Simple path using from_complex ---
print("Building disconnectivity graph (simple path)...")
dg = DisconnectivityGraph.from_complex(HC, nlevels=15, ts_energy="max")
dg.calculate()

# --- Method 2: Manual path for educational purposes ---
print("Building disconnectivity graph (manual path)...")
graph, energy_cache = database_from_complex(HC, ts_energy="max")
dg_manual = DisconnectivityGraph(graph, nlevels=15, energy_cache=energy_cache)
dg_manual.calculate()

# Both methods produce identical results; use the simpler from_complex approach
print(f"Graph has {graph.number_of_nodes()} nodes "
      f"({len([n for n in graph.nodes() if hasattr(n, 'coords')])} minima)\n")

# Plot the disconnectivity graph
dg.plot(title="Disconnectivity Graph: 3sin(5x) + 0.5(x² + y²)")
plt.tight_layout()
plt.savefig("disconnectivity_graph.png", dpi=150, bbox_inches="tight")
print("Saved disconnectivity_graph.png")
plt.show()
