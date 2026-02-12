"""
Polyscope-based 3D visualization for DDG dual meshes.

Requires the optional ``polyscope`` package. Install with::

    pip install polyscope

Usage::

    from hyperct.ddg._plot_polyscope import plot_dual_polyscope_3D
    plot_dual_polyscope_3D(HC)
"""
from __future__ import annotations

import collections

import numpy as np


# Inline colour constants (originally from ddgclib._misc.coldict)
_COLORS = {
    "do": (0.85, 0.55, 0.10),   # dark orange
    "lo": (1.00, 0.75, 0.30),   # light orange
    "db": (0.12, 0.25, 0.50),   # dark blue
    "lb": (0.40, 0.60, 0.85),   # light blue
    "tg": (0.17, 0.63, 0.17),   # tab:green
}


def plot_dual_polyscope_3D(
    HC,
    vd=None,
    fn: str = "",
    length_scale: float = 1.0,
    point_radii: float = 0.005,
    show: bool = True,
):
    """Plot dual mesh around a vertex using polyscope.

    :param HC: Complex with ``HC.Vd`` populated by ``compute_vd``.
    :param vd: Primal vertex to visualise dual around. If None, uses
        the first vertex in HC.V.
    :param fn: Filename for screenshot (empty string = no screenshot).
    :param length_scale: Polyscope length scale.
    :param point_radii: Point cloud radius.
    :param show: If True, call ``ps.show()``.
    :return: (ps, surface) polyscope objects.
    """
    try:
        import polyscope as ps
    except ImportError:
        raise ImportError(
            "polyscope is required for this function. "
            "Install with: pip install polyscope"
        )

    do = _COLORS["do"]
    db = _COLORS["db"]

    # Reset indices for plotting
    for i, v in enumerate(HC.V):
        v.index = i

    if vd is None:
        v1 = list(HC.V)[0]
    else:
        v1 = vd

    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")

    # Build dual mesh simplices around v1
    dual_points_set = set()
    ssets = []  # Triangle simplices for the dual surface
    for v2 in v1.nn:
        # Midpoint of primary edge
        vc_12 = 0.5 * (v2.x_a - v1.x_a) + v1.x_a
        vc_12 = HC.Vd[tuple(vc_12)]

        # Shared dual vertices
        dset = v2.vd.intersection(v1.vd)
        vd_i = list(dset)[0]

        if _has_boundary(v1) and _has_boundary(v2):
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = len(list(dset)) - 2
        else:
            iter_len = len(list(dset))

        dsetnn = vd_i.nn.intersection(dset)
        vd_j = list(dsetnn)[0]

        for _ in range(iter_len):
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)
            dsetnn_k.remove(vd_i)
            vd_i = vd_j
            try:
                vd_j = list(dsetnn_k)[0]
            except IndexError:
                pass

        # Collect dual points
        for vd_pt in dset:
            dual_points_set.add(vd_pt.x)

    # Register dual points
    dual_points = np.array(list(dual_points_set))
    ps_cloud = ps.register_point_cloud("Dual points", dual_points)
    ps_cloud.set_color(do)
    ps_cloud.set_radius(point_radii)

    # Build triangle mesh from simplices
    vdict = collections.OrderedDict()
    ind = 0
    faces = []
    for s in ssets:
        f = []
        for vd_pt in s:
            if vd_pt.x not in vdict:
                vdict[vd_pt.x] = ind
                ind += 1
            f.append(vdict[vd_pt.x])
        faces.append(f)

    verts = np.array(list(vdict.keys()))
    faces = np.array(faces)

    dsurface = ps.register_surface_mesh(
        "Dual face", verts, faces,
        color=do, edge_width=0.0, smooth_shade=False,
    )
    dsurface.set_transparency(0.5)

    # Plot primary mesh
    HC.dim = 2  # Temporarily set to 2D for surface extraction
    HC.vertex_face_mesh()
    HC.dim = 3
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)

    # Primary vertices
    ps_cloud_p = ps.register_point_cloud("Primary points", points)
    ps_cloud_p.set_color(db)
    ps_cloud_p.set_radius(point_radii)

    # Primary surface
    surface = ps.register_surface_mesh(
        "Primary surface", points, triangles,
        color=db, edge_width=1.0,
        edge_color=(0.0, 0.0, 0.0), smooth_shade=False,
    )
    surface.set_transparency(0.3)

    # Scene settings
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.1)
    ps.set_shadow_darkness(0.2)
    ps.set_shadow_blur_iters(2)
    ps.set_transparency_mode("pretty")
    ps.set_length_scale(length_scale)
    ps.set_screenshot_extension(".png")
    if fn:
        ps.screenshot(fn)

    if show:
        ps.show()

    return ps, dsurface


def _has_boundary(v) -> bool:
    """Check if vertex has the boundary attribute set to True."""
    try:
        return v.boundary
    except AttributeError:
        return False
