"""
Discrete Differential Geometry (DDG) dual computations for hyperct.

Provides barycentric and circumcentric dual mesh computation on
hyperct simplicial complexes.

Usage::

    from hyperct import Complex
    from hyperct.ddg import compute_vd, e_star, v_star, d_area

    HC = Complex(2)
    HC.triangulate()
    HC.refine_all()

    # Set boundary vertices
    dV = HC.boundary()
    for v in dV:
        v.boundary = True

    # Compute dual mesh (barycentric or circumcentric)
    compute_vd(HC, method="barycentric")

    # Use discrete operators
    for v1 in HC.V:
        area = d_area(v1)
"""
from ._compute_dual import compute_vd
from ._curvature import (
    HNdC_ijk,
    integrated_curvature,
    mean_curvature,
    normal_area,
)
from ._operators import d_area, e_star, v_star
from ._strategies import barycenter, circumcenter

__all__ = [
    "compute_vd",
    "e_star",
    "v_star",
    "d_area",
    "barycenter",
    "circumcenter",
    "HNdC_ijk",
    "normal_area",
    "mean_curvature",
    "integrated_curvature",
]
