"""
.. deprecated::
    This module is deprecated. Use ``hyperct.ddg.compute_vd`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "hyperct.ddg.circumcentric.circumcentric_duals is deprecated. "
    "Use hyperct.ddg.compute_vd(HC, method='circumcentric') instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np

from hyperct import Complex, VertexCacheField


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


################################
# New code for volumes:
################################
def _signed_volume_parallelepiped(u, v, w):
    u, v, w = map(np.array, (u, v, w))
    v_para = np.dot(u, v) * u
    v_ortho = v - v_para
    w_prime = w - 2 * v_para
    return (np.cross(u, v_ortho)).dot(w_prime) / 6


def _volume_parallelepiped(u, v, w):
    vol = np.abs(_signed_volume_parallelepiped(u, v, w))
    print(f"Volume of Parallelepiped={vol}")
    return vol


################################


# Circumcenter computation (replaces barycentric mean)
def circumcenter(verts):
    """
    Compute the circumcenter of a simplex (triangle in 2D, tetrahedron in 3D).
    Returns the barycenter (mean) as fallback for degenerate cases.
    """
    verts = np.asarray(verts, dtype=float)
    n, dim = verts.shape
    if n != dim + 1:
        raise ValueError(f"Expected {dim + 1} points for dim={dim} simplex")

    if dim == 2:
        # 2D triangle circumcenter (determinant formula)
        A, B, C = verts
        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        if abs(D) < 1e-12:
            return np.mean(verts, axis=0)  # degenerate fallback
        Ux = ((A[0] ** 2 + A[1] ** 2) * (B[1] - C[1]) +
              (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1]) +
              (C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])) / D
        Uy = ((A[0] ** 2 + A[1] ** 2) * (C[0] - B[0]) +
              (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0]) +
              (C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])) / D
        return np.array([Ux, Uy])

    elif dim == 3:
        # 3D tetrahedron circumcenter
        A, B, C, D_pt = verts  # rename to avoid conflict with D
        AB = B - A
        AC = C - A
        AD = D_pt - A
        M = np.stack((AB, AC, AD))
        rhs = 0.5 * np.array([np.dot(AB, AB), np.dot(AC, AC), np.dot(AD, AD)])
        try:
            sol = np.linalg.solve(M, rhs)
            return A + sol
        except np.linalg.LinAlgError:
            return np.mean(verts, axis=0)  # degenerate fallback

    else:
        raise NotImplementedError("Only dim=2 and dim=3 supported")


# Example Usage:
if 0:
    u = np.array((1, 0, 0))
    v = np.array((0, 1, 0))
    w = np.array((0, 0, 1))
    _volume_parallelepiped(u, v, w)


class _PlanePoints:
    """
    A special helper class for plot_dual to define attributes when tri is known, but can't be found with QHull due to
    too few points
    """

    def __init__(self, simplices, points):
        self.simplices = simplices
        self.points = points


def plot_dual(vd, HC, vector_field=None, scalar_field=None, fn='', up="x_up"
              , stl=False, length_scale=1.0, point_radii=0.005):
    # Reset the indices for plotting:
    for i, v in enumerate(HC.V):
        v.index = i
    v1 = vd
    # Initialize polyscope
    ps.init()
    ps.set_up_dir('z_up')
    do = coldict['do']
    lo = coldict['lo']
    db = coldict['db']
    lb = coldict['lb']
    tg = coldict['tg']  # Tab:green colour
    # %% Plot Circumcentric dual mesh
    # Loop over primary edges
    dual_points_set = set()
    ssets = []  # Sets of simplices
    v1 = vd
    for i, v2 in enumerate(v1.nn):
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v2.x_a - v1.x_a) + v1.x_a  # Midpoint (common crossing point)
        vc_12 = HC.Vd[tuple(vc_12)]

        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)  # Always 5 for boundaries
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v1.boundary and v2.boundary:
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = len(list(dset)) - 2
        else:
            iter_len = len(list(dset))

        # Main loop
        dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
        vd_j = list(dsetnn)[0]
        # NOTE: In the boundary edges the last triangle does not have
        #      a final vd_j
        # print(f'dset = {dset}')
        for _ in range(iter_len):  # For boundaries should be length 2?
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
            # print(f'dsetnn_k = {dsetnn_k}')
            dsetnn_k.remove(vd_i)  # Should now be size 1
            vd_i = vd_j
            try:
                # Alternatively it should produce an IndexError only when
                # _ = 2 (end of range(3) and we are on a boundary edge
                # so that (v1.boundary and v2.boundary) is true
                vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
            except IndexError:
                pass  # Should only happen for boundary edges

        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)
        pi = []
        for vd in dset:
            # pi.append(vd.x + 1e-9 * np.random.rand())
            pi.append(vd.x)
            dual_points_set.add(vd.x)
        pi = np.array(pi)
        pi_2d = pi[:, :2] + 1e-9 * np.random.rand()

        # Plot dual points:
        dual_points = []
        for vd in dual_points_set:
            dual_points.append(vd)

        dual_points = np.array(dual_points)
        ps_cloud = ps.register_point_cloud("Dual points", dual_points)
        ps_cloud.set_color(do)
        ps_cloud.set_radius(point_radii)

    # Build the simplices for plotting
    faces = []
    vdict = collections.OrderedDict()  # Ordered cache of vertices to plot
    ind = 0
    # Now iterate through all the constructed simplices and find indexes
    for s in ssets:
        f = []
        for vd in s:
            if not (vd.x in vdict):
                vdict[vd.x] = ind
                ind += 1

            f.append(vdict[vd.x])
        faces.append(f)

    verts = np.array(list(vdict.keys()))
    faces = np.array(faces)

    print(f'verts = {verts}')
    dsurface = ps.register_surface_mesh(f"Dual face", verts, faces,
                                        color=do,
                                        edge_width=0.0,
                                        edge_color=(0.0, 0.0, 0.0),
                                        smooth_shade=False)

    dsurface.set_transparency(0.5)
    # Plot primary mesh
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    HC.dim = 3  # Reset the dimension to 3
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)

    # %% Register the primary vertices as a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("Primary points", my_points)
    ps_cloud.set_color(tuple(db))
    ps_cloud.set_radius(point_radii)
    # ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    faces = triangles
    if stl:
        #  msh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                pass
                # msh.vectors[i][j] = verts[f[j], :]

        # msh.save(f'{fn}.stl')

    ### Plot the primary mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    if 1:
        surface = ps.register_surface_mesh("Primary surface", verts, faces,
                                           color=db,
                                           edge_width=1.0,
                                           edge_color=(0.0, 0.0, 0.0),
                                           smooth_shade=False)

        surface.set_transparency(0.3)
        # Add a scalar function and a vector function defined on the mesh
        # vertex_scalar is a length V numpy array of values
        # face_vectors is an Fx3 array of vectors per face

        # Scene options (New, not working for scaling
        # NOTE: VERY BROKEN AS IT SCALES THE DIFFERENT MESHES RELATIVELY: NEVER USE THIS:
        # ps.set_autocenter_structures(True)
        # ps.set_autoscale_structures(True)

        # View the point cloud and mesh we just registered in the 3D UI
        # ps.show()
        # Plot particles
        # Ground plane options
        ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
        ps.set_ground_plane_height_factor(0.1)  # adjust the plane height
        ps.set_shadow_darkness(0.2)  # lighter shadows
        ps.set_shadow_blur_iters(2)  # lighter shadows
        ps.set_transparency_mode('pretty')
        ps.set_length_scale(length_scale)
        # ps.set_length_scale(length_scale)
        #   ps.set_length_scale(length_scale)
        # ps.look_at((0., -10., 0.), (0., 0., 0.))
        # ps.look_at((1., -8., -8.), (0., 0., 0.))
        # ps.set_ground_plane_height_factor(x, is_relative=True)
        ps.set_screenshot_extension(".png")
        # Take a screenshot
        # It will be written to your current directory as screenshot_000000.jpg, etc
        ps.screenshot(fn)

    return ps, du


# ... (all other plot_dual_old* functions remain unchanged - they are legacy)


def _set_boundary(v, val=True):
    """
    small helper fuction to set the boundary value property for the supplied vertex.
    :param v:
    :return:
    """
    v.boundary = val


# Dual complex computation functions:
def _merge_local_duals_vector(x_a_l, Vd_cache, cdist=1e-10):
    """
    For a proposed new vertex position, first check the local dual cache
    of vertices for a similar position, if one is found, use that exact
    position instead to avoid generating duplicate dual vetices.

    This is needed due to overflow errors giving slightly different results
    and therefore producing multiple keys for the same dual.

    :param x_a_l: List of vectors of new vertex position
    :param Vd_cache: iterable object of local dual vertices
    :param cdist: scalar, tolerance of identifying dual vertices
    :return: x_a_l: The modified list of vertex, newly merged and unmerged
    """
    for vd_i in Vd_cache:
        for i, x_a in enumerate(x_a_l):
            dist = np.linalg.norm(vd_i.x_a - x_a)
            if dist < cdist:
                x_a_l[i] = vd_i.x_a

    return x_a_l


def compute_vd(HC, cdist=1e-10):
    """
    Computes the dual vertices of a primal vertex cache HC.V on
    each dim - 1 simplex using CIRCUMCENTERS (instead of barycenters).

    Currently only dim = 2, 3 is supported

    cdist: float, tolerance for where a unique dual vertex can exist

    """
    # TODO: Merging the dual vertices is probably inefficient, it might
    # actually be more efficient to do one global merge of the vertices
    # after the routine has finished instead.
    # Construct dual cache
    HC.Vd = VertexCacheField()

    # Construct dual neighbour sets
    for v in HC.V:
        v.vd = set()

    # hcv = copy.copy(HC.V)
    if HC.dim == 2:
        for v1 in HC.V:

            for v2 in v1.nn:
                # Compute the local dual neighbourhood to current v2:
                # NOTE: This should be updated in every v3 for loop because dual vertices are
                #      being added to this nn cache every loop:
                v1_d_nn = list(v1.vd)
                # If boundary vertex, we stop and generate a new vertex on the boundary edge.
                try:
                    if v1.boundary and v2.boundary:
                        cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        cd = _merge_local_duals_vector([cd], v1_d_nn, cdist=cdist)[0]
                        vd = HC.Vd[tuple(cd)]
                        v1.vd.add(vd)
                        v2.vd.add(vd)
                        # Connect to dual simplex
                        v1nn_u_v2nn = v1.nn.intersection(v2.nn)  # Should always be length 1
                        v3 = list(v1nn_u_v2nn)[0]
                        verts = np.zeros([3, HC.dim])
                        verts[0] = v1.x_a
                        verts[1] = v2.x_a
                        verts[2] = v3.x_a
                        cd1 = circumcenter(verts)  # CIRCUMCENTER
                        vd1 = HC.Vd[tuple(cd1)]
                        # Connect the two dual vertices forming the boundary dual edge:
                        vd.connect(vd1)
                        continue
                except AttributeError:
                    pass
                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)  # Should always be length 2
                # In 2D there are only two
                v3_1 = list(v1nn_u_v2nn)[0]
                v3_2 = list(v1nn_u_v2nn)[1]
                if (v3_1 is v1) or (v3_2 is v1):
                    continue
                verts = np.zeros([3, HC.dim])
                verts[0] = v1.x_a
                verts[1] = v2.x_a
                verts[2] = v3_1.x_a
                # Compute the circumcentre:
                cd1 = circumcenter(verts)  # CIRCUMCENTER

                # Compute the circumcentre of the second triangle
                verts[2] = v3_2.x_a
                cd2 = circumcenter(verts)  # CIRCUMCENTER

                # Ensure that floating point errors are not generating a unique vertex
                # NOTE: In the future cdist should be selected dynamically based on the local
                #       distance between v2 and its dual vertices / primary edge connections
                (cd1, cd2) = _merge_local_duals_vector([cd1, cd2], v1_d_nn, cdist=cdist)

                vd1 = HC.Vd[tuple(cd1)]
                vd2 = HC.Vd[tuple(cd2)]
                # Connect the two dual vertices:
                vd1.connect(vd2)

                # Connect to all primal vertices of v3_1 dual
                for v in [v1, v2, v3_1]:
                    v.vd.add(vd1)

                # Connect to all primal vertices of v3_2 dual
                for v in [v1, v2, v3_2]:
                    v.vd.add(vd2)

    elif HC.dim == 3:
        for v1 in HC.V:
            for v2 in v1.nn:
                # Note: every boundary primary edge only has two boundary tetrahedra connected
                # and therefore only two circumcentric dual points. We do not need to connect with
                # other duals therefore simply connect to the primary edges.

                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                for v3 in v1nn_u_v2nn:
                    # Compute the local dual neighbourhood to current v2:
                    # NOTE: This should be updated in every v3 for loop because dual vertices are
                    #      being added to this nn cache every loop:
                    v1_d_nn = list(v1.vd)
                    # print('-')
                    if (v3 is v1):
                        continue

                    v1nn_u_v2nn_u_v3nn = v1nn_u_v2nn.intersection(
                        v3.nn)  # Should be length 2, unless the triangle is on the boundary
                    # print(f'v1.x = {v1.x}')
                    # print(f'v2.x = {v2.x}')
                    # print(f'v3.x = {v3.x}')
                    v4_1 = list(v1nn_u_v2nn_u_v3nn)[0]
                    # v4_2 = list(v1nn_u_v2nn_u_v3nn)[1]

                    # debug above, should never occur?:
                    if 1:
                        if (v4_1 is v1) or (v4_1 is v2):
                            print(f'WARNING (v4_1 is v1) or (v4_1 is v2)')

                    # Compute the two duals of tetrahedra connected by face f_123 of triangle [v1, v2, v3]
                    verts = np.zeros([HC.dim + 1, HC.dim])
                    verts[0] = v1.x_a
                    verts[1] = v2.x_a
                    verts[2] = v3.x_a
                    verts[3] = v4_1.x_a
                    #  Compute the circumcentre of the first connected simplex sharing primary face f_123:
                    cd1 = circumcenter(verts)  # CIRCUMCENTER

                    # If v123 is on the boundary then we instead want to generate the circumcentre
                    # dual vd123 and then connect it to edge dual vd12 and cd1
                    if (v1.boundary and v2.boundary) and v3.boundary:
                        # debug print:
                        if len(list(v1nn_u_v2nn_u_v3nn)) > 1:
                            print(
                                f'WARNING: len(list(v1nn_u_v2nn_u_v3nn)) = {len(list(v1nn_u_v2nn_u_v3nn))} which is > expected 1')

                        # verts_b = np.zeros([3, HC.dim])
                        verts_b = verts[:3]
                        cd2 = circumcenter(verts_b)  # CIRCUMCENTER of boundary triangle

                        # Connect the dual of e_12 primal edge vertices
                        cd12 = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        cd12 = _merge_local_duals_vector([cd12], v1_d_nn, cdist=cdist)[0]
                        vd12 = HC.Vd[tuple(cd12)]
                        v1.vd.add(vd12)
                        v2.vd.add(vd12)

                    # Compute the circumcentre of the second connected simplex sharing primary face f_123:
                    else:
                        v4_2 = list(v1nn_u_v2nn_u_v3nn)[1]
                        verts[3] = v4_2.x_a
                        cd2 = circumcenter(verts)  # CIRCUMCENTER

                        # debug above, should never occur?:
                        if 1:
                            if (v4_2 is v1) or (v4_2 is v2):
                                print(f'WARNING (v4_1 is v1) or (v4_1 is v2)')

                    (cd1, cd2) = _merge_local_duals_vector([cd1, cd2], v1_d_nn, cdist=cdist)

                    #  Define the new dual vertices
                    vd1 = HC.Vd[tuple(cd1)]
                    vd2 = HC.Vd[tuple(cd2)]
                    # Connect the two dual vertices:
                    vd1.connect(vd2)

                    # Connect to all primal vertices of v3_1 dual
                    for v in [v1, v2, v3, v4_1]:
                        v.vd.add(vd1)

                    # Connect to all primal vertices of v3_2 dual
                    if (v1.boundary and v2.boundary) and v3.boundary:
                        for v in [v1, v2, v3]:  # v4_2 doesn't exist on boundary face
                            v.vd.add(vd2)
                    else:
                        for v in [v1, v2, v3, v4_2]:
                            v.vd.add(vd2)

    return HC  # self

# (All other functions such as triang_dual, plot_dual_mesh_2D, plot_dual_mesh_3D,
# area_of_polygon, volume_of_geometric_object, v_star, e_star, d_area, dP, du, dudt,
# boundary helpers, etc. remain unchanged. The geometry computations adapt automatically
# because they now use the circumcenter positions stored in HC.Vd.)

# NOTE: The legacy compute_vd_old* functions are not modified here as they are deprecated.
# If needed, apply the same circumcenter replacement to them.