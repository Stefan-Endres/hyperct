"""
2D cube flow animation.

Vertices of a refined unit cube move in the +x direction at 0.05 per frame
and are deleted when they leave the domain [(0, 1), (0, 1)].  Demonstrates
topology changes (vertex removal, edge disconnection) during animation.
"""
import numpy as np
from hyperct import Complex
from hyperct._plotting import animate_complex
import matplotlib.pyplot as plt

# Build a 2-D unit cube and refine twice
H = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
H.triangulate()
H.refine_all()
H.refine_all()
print(f"Initial: {len(H.V.cache)} vertices")

velocity = 0.05  # x-displacement per frame

def flow(hc, frame):
    """Move all vertices in +x by velocity; delete those leaving the domain."""
    to_remove = []
    for v in list(hc.V):
        new_x = (v.x[0] + velocity, v.x[1])
        if new_x[0] > 1.0:
            to_remove.append(v)
        else:
            hc.V.move(v, new_x)

    for v in to_remove:
        hc.V.remove(v)

    remaining = len(hc.V.cache)
    if remaining == 0:
        return
    print(f"  frame {frame:3d}: {remaining} vertices remaining")

fig, ax, anim = animate_complex(
    H, flow,
    frames=25, interval=200,
    figsize=(6, 6),
)

plt.tight_layout()
plt.show()
