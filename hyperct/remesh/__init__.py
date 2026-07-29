"""Interface-preserving adaptive remeshing for simplicial complexes.

This module provides local mesh operations — edge split, edge collapse,
edge flip (2D) — that maintain mesh quality on Lagrangian meshes while
preserving sharp phase interfaces as hard constraints.  It is a drop-in
alternative to global Delaunay retopologization for multiphase flow
simulations where cross-phase edges must never be introduced.

Main entry point: :func:`adaptive_remesh`.

References
----------
Persson, P.-O. & Strang, G. (2004). "A Simple Mesh Generator in MATLAB".
    SIAM Review 46(2), 329-345.
Freitag, L.A. & Ollivier-Gooch, C. (1997). "Tetrahedral mesh improvement
    using swapping and smoothing". Int. J. Numer. Methods Eng. 40(21).
"""

from hyperct.remesh._quality import (
    triangle_min_angle,
    triangle_aspect_ratio,
    triangle_area,
    mesh_quality_histogram,
)
from hyperct.remesh._interface import (
    is_interface_edge,
    can_flip,
    can_collapse,
    vertex_phase,
)
from hyperct.remesh._operations_2d import (
    triangles_around_edge,
    edge_split_2d,
    edge_collapse_2d,
    edge_flip_2d,
)
from hyperct.remesh._driver import adaptive_remesh

__all__ = [
    "triangle_min_angle",
    "triangle_aspect_ratio",
    "triangle_area",
    "mesh_quality_histogram",
    "is_interface_edge",
    "can_flip",
    "can_collapse",
    "vertex_phase",
    "triangles_around_edge",
    "edge_split_2d",
    "edge_collapse_2d",
    "edge_flip_2d",
    "adaptive_remesh",
]
