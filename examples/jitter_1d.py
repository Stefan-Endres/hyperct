"""
1D dynamic jitter animation.

Triangulates a 1-D domain and animates random vertex jitter using
animate_complex.  The update callback uses ``hc.V.move()`` to relocate
vertices each frame.
"""
import numpy as np
from hyperct import Complex
from hyperct._plotting import animate_complex
import matplotlib.pyplot as plt

# Build a 1-D complex and refine it a few times
H = Complex(1, domain=[(-2.0, 2.0)])
H.triangulate()
H.refine_all()
H.refine_all()
H.refine_all()
print(f"1D complex: {len(H.V.cache)} vertices")

# Store original vertex positions for the wobble reference
originals = {v.x: v.x for v in H.V}

rng = np.random.default_rng(42)

def jitter(hc, frame):
    """Perturb vertex positions with a smooth sinusoidal wobble."""
    amplitude = 0.05
    for v in list(hc.V):
        x0 = originals[v.x] if v.x in originals else v.x
        noise = rng.normal(0, 0.01)
        new_x = (x0[0] + amplitude * np.sin(0.1 * frame + x0[0] * 3) + noise,)
        originals[new_x] = originals.pop(v.x, v.x)
        hc.V.move(v, new_x)

fig, ax, anim = animate_complex(
    H, jitter,
    frames=200, interval=50,
    figsize=(10, 2),
)

plt.tight_layout()
plt.show()
