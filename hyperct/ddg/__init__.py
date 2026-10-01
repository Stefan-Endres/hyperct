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
from ._boundary import boundary_from_simplices
from ._compute_dual import compute_vd
from ._curvature import (
    HNdC_ijk,
    integrated_curvature,
    mean_curvature,
    normal_area,
)
from ._dual_cell import (
    dual_cell_area_2d,
    dual_cell_faces_3d,
    dual_cell_polygon_2d,
    dual_cell_vertices_1d,
)
from ._dual_volume import simplex_dual_volumes, vertex_dual_volume
from ._operators import batch_e_star, d_area, e_star, v_star
from ._retriangulation import (
    apex_vertices,
    connect_and_cache_simplices,
    get_edge_apex_map,
    invalidate_simplex_cache,
    rebuild_simplex_cache_2d,
    rebuild_simplex_cache_3d,
)
from ._strategies import barycenter, circumcenter

__all__ = [
    "apex_vertices",
    "batch_e_star",
    "boundary_from_simplices",
    "compute_vd",
    "connect_and_cache_simplices",
    "dual_cell_area_2d",
    "dual_cell_faces_3d",
    "dual_cell_polygon_2d",
    "dual_cell_vertices_1d",
    "e_star",
    "v_star",
    "d_area",
    "barycenter",
    "circumcenter",
    "get_edge_apex_map",
    "HNdC_ijk",
    "integrated_curvature",
    "invalidate_simplex_cache",
    "mean_curvature",
    "normal_area",
    "rebuild_simplex_cache_2d",
    "rebuild_simplex_cache_3d",
    "simplex_dual_volumes",
    "vertex_dual_volume",
]
