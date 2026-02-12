"""
3D dynamic jitter animation.

Triangulates a 3-D domain and animates random vertex jitter using
animate_complex.  The update callback uses ``hc.V.move()`` to relocate
vertices each frame.
"""
import numpy as np
from hyperct import Complex
from hyperct._plotting import animate_complex
import matplotlib.pyplot as plt

# Build a 3-D complex and refine
H = Complex(3, domain=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
H.triangulate()
H.refine_all()
print(f"3D complex: {len(H.V.cache)} vertices")

# Store original vertex positions for the wobble reference
originals = {v.x: v.x for v in H.V}

rng = np.random.default_rng(42)

def jitter(hc, frame):
    """Apply a radial breathing distortion to the 3D mesh."""
    amplitude = 0.03
    coords = np.array([v.x for v in hc.V])
    center = coords.mean(axis=0)
    for v in list(hc.V):
        x0 = np.array(originals[v.x] if v.x in originals else v.x)
        radial = x0 - center
        noise = rng.normal(0, 0.005, size=3)
        scale = 1.0 + amplitude * np.sin(
            0.1 * frame + np.linalg.norm(radial) * 4)
        new_pos = center + radial * scale + noise
        new_x = tuple(new_pos)
        originals[new_x] = originals.pop(v.x, v.x)
        hc.V.move(v, new_x)

fig, ax, anim = animate_complex(
    H, jitter,
    frames=200, interval=50,
    figsize=(8, 8),
)

plt.tight_layout()
plt.show()
