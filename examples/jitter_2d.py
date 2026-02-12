"""
2D dynamic jitter animation.

Triangulates a 2-D domain and animates random vertex jitter using
animate_complex.  The update callback uses ``hc.V.move()`` to relocate
vertices each frame.
"""
import numpy as np
from hyperct import Complex
from hyperct._plotting import animate_complex
import matplotlib.pyplot as plt

# Build a 2-D complex and refine
H = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
H.triangulate()
H.refine_all()
H.refine_all()
print(f"2D complex: {len(H.V.cache)} vertices")

# Store original vertex positions for the wobble reference
originals = {v.x: v.x for v in H.V}

rng = np.random.default_rng(42)

def jitter(hc, frame):
    """Apply a travelling-wave distortion to the 2D mesh."""
    amplitude = 0.02
    phase = 0.1 * frame
    for v in list(hc.V):
        x0 = originals[v.x] if v.x in originals else v.x
        noise = rng.normal(0, 0.005, size=2)
        dx = amplitude * np.sin(phase + x0[1] * 6 * np.pi) + noise[0]
        dy = amplitude * np.cos(phase + x0[0] * 6 * np.pi) + noise[1]
        new_x = (x0[0] + dx, x0[1] + dy)
        originals[new_x] = originals.pop(v.x, v.x)
        hc.V.move(v, new_x)

fig, ax, anim = animate_complex(
    H, jitter,
    frames=200, interval=50,
    figsize=(6, 6),
)

plt.tight_layout()
plt.show()
