"""
.. deprecated::
    This module is deprecated. Use ``hyperct.ddg.compute_vd`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "hyperct.ddg.barycentric._duals is deprecated. "
    "Use hyperct.ddg.compute_vd(HC, method='barycentric') instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from scipy.spatial import Delaunay

from hyperct import Complex, VertexCacheField


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# Special helplers for volume computations:
def _signed_volume_parallelepiped(u, v, w):
    u, v, w = map(np.array, (u, v, w))
    v_para = np.dot(u, v)*u
    v_ortho = v - v_para
    w_prime = w - 2*v_para
    return (np.cross(u, v_ortho)).dot(w_prime)/6

def _volume_parallelepiped(u, v, w):
    vol = np.abs(_signed_volume_parallelepiped(u, v, w))
    print(f"Volume of Parallelepiped={vol}")
    return vol

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
    each dim - 1 simplex.

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
                        cd1 = np.mean(verts, axis=0)
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
                # cd = circumcenter(verts)
                # Compute the barycentre of the first connected triangle sharing primary edge/face e_1e:
                cd1 = np.mean(verts, axis=0)

                # Compute the barycentre of the first connected triangle sharing primary edge/face e_1e:
                verts[2] = v3_2.x_a
                cd2 = np.mean(verts, axis=0)

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
                # and therefore only two barycentric dual points. We do not need to connect with
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

                    # if (v4_1 is v1) or (v4_1 is v2) or (v4_2 is v1) or (v4_2 is v2):
                    #    continue

                    # debug above, should never occur?:
                    if 1:
                        if (v4_1 is v1) or (v4_1 is v2):
                            print(f'WARNING (v4_1 is v1) or (v4_1 is v2)')

                    # Compute the two duals of tetrahedra connected by face f_123 of triangle [v1, v2, v3]
                    verts = np.zeros([HC.dim + 1, HC.dim])
                    verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                    verts[1] = v2.x_a
                    verts[2] = v3.x_a
                    verts[3] = v4_1.x_a
                    #  Compute the barycentre of the first connected simplex sharing primary face f_123:
                    cd1 = np.mean(verts, axis=0)

                    # If v123 is on the boundary then we instead want to generate the barycenter
                    # dual vd123 and then connect it to edge dual vd12 and cd1
                    if (v1.boundary and v2.boundary) and v3.boundary:
                        # debug print:
                        if len(list(v1nn_u_v2nn_u_v3nn)) > 1:
                            print(
                                f'WARNING: len(list(v1nn_u_v2nn_u_v3nn)) = {len(list(v1nn_u_v2nn_u_v3nn))} which is > expected 1')

                        # verts_b = np.zeros([3, HC.dim])
                        verts_b = verts[:3]
                        cd2 = np.mean(verts_b, axis=0)

                        # Connect the dual of e_12 primal edge vertices
                        cd12 = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        cd12 = _merge_local_duals_vector([cd12], v1_d_nn, cdist=cdist)[0]
                        vd12 = HC.Vd[tuple(cd12)]
                        v1.vd.add(vd12)
                        v2.vd.add(vd12)

                    # Compute the barycentre of the second connected simplex sharing primary face f_123:
                    else:
                        v4_2 = list(v1nn_u_v2nn_u_v3nn)[1]
                        verts[3] = v4_2.x_a
                        cd2 = np.mean(verts, axis=0)

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


# Find the Delaunay dual
def triang_dual(points, plot_delaunay=False):
    """
    Compute the Delaunay triangulation plus the dual points. Put into hyperct complex object.
    #TODO: We need to compute boundaries before compute_vd
    """
    dim = points.shape[1]
    tri = Delaunay(points)
    if plot_delaunay:  # Plot Delaunay complex
        import matplotlib.pyplot as plt
        plt.triplot(points[: ,0], points[: ,1], tri.simplices)
        plt.plot(points[: ,0], points[: ,1], 'o')
        plt.show()

    # Put Delaunay back into hyperct Complex object:
    HC = Complex(dim)
    for s in tri.simplices:
        for v1i in s:
            for v2i in s:
                if v1i is v2i:
                    continue
                else:
                    v1 = tuple(points[v1i])
                    v2 = tuple(points[v2i])
                    HC.V[v1].connect(HC.V[v2])

    #compute_vd(HC, cdist =1e-10)
    return HC, tri

# Plot duals


# Geometry and dual computations
def area_of_polygon(points):
    """Calculates the area of a polygon in 3D space.

    Args:
      points: A numpy array of shape (n, 3), where each row represents a point in
        3D space.

    Returns:
      The area of the polygon.
    """

    # Calculate the cross product of each pair of adjacent edges.
    edges = points[1:] - points[:-1]
    cross_products = np.cross(edges[:-1], edges[1:])

    # Calculate the area of the triangle formed by each pair of adjacent edges and
    # the origin.
    triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)

    # Sum the areas of all the triangles to get the total area of the polygon.
    return np.sum(triangle_areas)

def volume_of_geometric_object(points, extra_point):
    """Calculates the volume of a geometric object defined by adding an extra
    point away from the plane and connecting all points in the plane to it.

    Args:
    points: A numpy array of shape (n, 3), where each row represents a point in
      3D space.
    extra_point: A numpy array of shape (3,), representing the extra point away
      from the plane.

    Returns:
    The volume of the geometric object.
    """

    # Calculate the normal vector to the plane that contains the base polygon.
    normal_vector = np.cross(points[1] - points[0], points[2] - points[0])

    # Calculate the projection of the extra point onto the plane.
    projected_extra_point = extra_point - np.dot(extra_point - points[0], normal_vector) / np.linalg.norm(normal_vector)**2 * normal_vector

    # Calculate the distance between the extra point and its projection onto the plane.
    distance = np.linalg.norm(extra_point - projected_extra_point)

    # Calculate the area of the base polygon.
    base_area = area_of_polygon(points)

    # Calculate the volume of the geometric object.
    volume = 1/3 * base_area * distance

    return volume


def v_star(v_i, v_j, HC, n=None, dim=2):
    """
    Compute the dual flux planes and volume of the primary edge e_ij.
    It's needed to specify the dimension dim.

    n is a directional vector

    return : e_ij_star
    """
    if dim == 2:
        # e_ij_star = 0  # Initialize total dual area to zero
        # Find the shared dual vertices between vp1 and vp2
        vdnn = v_i.vd.intersection(v_j.vd)  # Should always be 2 for dim=2

        vd1 = list(vdnn)[0]
        vd2 = list(vdnn)[1]
        e_ij_star = np.linalg.norm(vd1.x_a - vd2.x_a)

    elif dim == 3:
        # Set irectional vector if None:
        if n is None:
            n = np.array([0, 0, 0])
        e_ij_star = 0
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a  # TODO: Should be done in the compute_vd function
        vc_12 = HC.Vd[tuple(vc_12)]
        # Find local dual points intersecting vertices terminating edge:
        dset = v_j.vd.intersection(v_i.vd)  # Always 5 for boundaries
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v_i.boundary and v_j.boundary:
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = 3
        else:
            iter_len = len(list(dset))

        # Main loop
        dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
        vd_j = list(dsetnn)[0]
        # NOTE: In the boundary edges the last triangle does not have
        #      a final vd_j
        ssets = []  # Sets of simplices
        A_ij = []  # list of triangle vector areas
        V_ij = []  # list of signed tetrahedral vector volumes
        for _ in range(iter_len):  # For boundaries should be length 2?
            # Compute the discrete vector area of the local triangle
            wedge_dij_ik = np.cross(vc_12.x_a - vd_i.x_a, vd_j.x_a - vd_i.x_a)
            if np.dot(normalized(wedge_dij_ik), n) < 0:  #TODO: Should just reverse sign
                wedge_dij_ik = np.cross(vd_j.x_a - vd_i.x_a, vc_12.x_a - vd_i.x_a)

            A_ij.append(wedge_dij_ik/2.0)

            # Compute the local signed volume
            verts = np.zeros([3, 3])
            verts[0] = vc_12.x_a
            verts[1] = vd_i.x_a
            verts[2] = vd_j.x_a
            #v_dij_i = volume_of_geometric_object(points, extra_point)
            v_dij_i = volume_of_geometric_object(verts, v_i.x_a)
            V_ij.append(v_dij_i)

            # Add to the set of simplces (undeeded here?)
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
            dsetnn_k.remove(vd_i)  # Should now be size 1
            vd_i = vd_j
            try:
                # Alternatively it should produce an IndexError only when
                # _ = 2 (end of range(3) and we are on a boundary edge
                # so that (v_i.boundary and v_j.boundary) is true
                vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
            except IndexError:
                pass  # Should only happen for boundary edges
        A_ij = np.array(A_ij)
        #A_ij = np.sum(A_ij, axis=0)
        V_ij = np.array(V_ij)
    else:
        print("WARNING: Not implemented yet from dim > 3")

    #return e_ij_star
    return A_ij, V_ij


def e_star(v_i, v_j, HC, n=None, dim=2):
    """
    Compute the dual of the primary edge e_ij. It's needed to specify the dimension dim.

    n is a directional vector

    return : e_ij_star
    """
    if dim == 2:
        # e_ij_star = 0  # Initialize total dual area to zero
        # Find the shared dual vertices between vp1 and vp2
        vdnn = v_i.vd.intersection(v_j.vd)  # Should always be 2 for dim=2

        vd1 = list(vdnn)[0]
        vd2 = list(vdnn)[1]
        e_ij_star = np.linalg.norm(vd1.x_a - vd2.x_a)

    elif dim == 3:
        # Set irectional vector if None:
        if n is None:
            n = np.array([0, 0, 0])
        e_ij_star = 0
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a  # TODO: Should be done in the compute_vd function
        vc_12 = HC.Vd[tuple(vc_12)]
        # Find local dual points intersecting vertices terminating edge:
        dset = v_j.vd.intersection(v_i.vd)  # Always 5 for boundaries
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v_i.boundary and v_j.boundary:
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = 3
        else:
            iter_len = len(list(dset))

        # Main loop
        dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
        vd_j = list(dsetnn)[0]
        # NOTE: In the boundary edges the last triangle does not have
        #      a final vd_j
        ssets = []  # Sets of simplices
        A_ij = []  # list of triangle vector areas
        for _ in range(iter_len):  # For boundaries should be length 2?
            # Compute the discrete vector area of the local triangle
            wedge_ij_ik = np.cross(vc_12.x_a - vd_i.x_a, vd_j.x_a - vd_i.x_a)
            if np.dot(normalized(wedge_ij_ik), n) < 0:  #TODO: Should just reverse sign
                wedge_ij_ik = np.cross(vd_j.x_a - vd_i.x_a, vc_12.x_a - vd_i.x_a)

            A_ij.append(wedge_ij_ik/2.0)
            # Add to the set of simplces (undeeded here?)
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
            dsetnn_k.remove(vd_i)  # Should now be size 1
            vd_i = vd_j

            try:
                # Alternatively it should produce an IndexError only when
                # _ = 2 (end of range(3) and we are on a boundary edge
                # so that (v_i.boundary and v_j.boundary) is true
                vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
            except IndexError:
                pass  # Should only happen for boundary edges

    else:
        print("WARNING: Not implemented yet from dim > 3")

    return e_ij_star

# Area computations
def d_area(vp1):
    """
    Compute the dual area of a vertex object vp1, which is the sum of the areas
    of the local dual triangles formed between vp1, its neighbouring vertices,
    and their shared dual vertices.

    Parameters:
    -----------
    vp1 : object
        A vertex object containing the following attributes:
        - vp1.nn: a list of neighboring vertex objects
        - vp1.vd: a set of dual vertex objects
        - vp1.x_a: a numpy array representing the position of vp1

    Returns:
    --------
    darea : float
        The total dual area of the vertex object vp1
    """

    darea = 0  # Initialize total dual area to zero
    for vp2 in vp1.nn:  # Iterate over neighboring vertex objects
        # Find the shared dual vertices between vp1 and vp2
        vdnn = vp1.vd.intersection(vp2.vd)
        # Compute the midpoint between vp1 and vp2
        mp = (vp1.x_a + vp2.x_a) / 2
        # Compute the height of the dual triangle between vp1, vp2, and a dual vertex
        h = np.linalg.norm(mp - vp1.x_a)
        for vdi in vdnn:  # Iterate over shared dual vertices
            # Compute the base of the dual triangle between vp1, vp2, and vdi
            b = np.linalg.norm(vdi.x_a - mp)
            # Add the area of the dual triangle to the total dual area
            darea += 0.5 * b * h

    return darea

# Volume computations (Note: in DDG the Hodge dual of scalar points in 3D)


## Boundary computations
import numpy as np

def _reflect_vertex_over_edge(triangle, target_index=0):
    """
    Reflect a given vertex of a triangle  (passed as array) over
    the opposing edge of the target vertex, maintaining the same plane.

    :param triangle: np.ndarray, input triangle
    :param target_index: int, target index
    :return: np.ndarray: Updated triangle with the reflected vertex.
    """
    p_o = triangle[target_index]
    p_1 = triangle[(target_index + 1) % 3]
    p_2 = triangle[(target_index + 2) % 3]
    p_midpoint = (p_1 + p_2) / 2
    # Move along the direction of (p_midpoint-p0) for twice the distance
    p_ref = p_o + 2 * (p_midpoint - p_o)
    triangle[target_index] = p_ref
    return triangle


def _find_intersection(plane1, plane2, plane3):
    """
    Find the intersection of 3 planes, the planes,
    the arguments are supplied as coeeficients of the planes
    e.g. for the first plane:

    a_1 x_1 + b_1 x_2 + c_1 x_3 + d_1 = 0

    :param plane1: np.ndarray, vector of 4 elements
    :param plane2: np.ndarray, vector of 4 elements
    :param plane3: np.ndarray, vector of 4 elements
    :return: interesection_point, np.ndarray, vector 3 elements
    """
    # Extract coefficients from each plane equation
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    a3, b3, c3, d3 = plane3

    # Coefficients matrix (A)
    A = np.array([[a1, b1, c1],
                  [a2, b2, c2],
                  [a3, b3, c3]])

    # Check if the matrix is singular
    if np.linalg.det(A) == 0:
        raise ValueError("The planes are parallel or nearly parallel. No unique solution.")

    # Right-hand side vector (b)
    b = np.array([-d1, -d2, -d3])

    # Solve the system of linear equations
    intersection_point = np.linalg.solve(A, b)

    return intersection_point


def _find_plane_equation(v_1, v_2, v_3):
    """
    Find the plane equation of a given plane spanned by vectors e_21 = v_2 - v_1
    and e_31 = v_3 - v_1
    """
    # Compute vectors lying in the plane
    vector1 = np.array(v_2) - np.array(v_1)
    vector2 = np.array(v_3) - np.array(v_1)

    # Compute the normal vector using the cross product
    normal_vector = np.cross(vector1, vector2)

    # Extract coefficients for the plane equation ax + by + cz + d = 0
    a, b, c = normal_vector
    d = -np.dot(normal_vector, np.array(v_1))

    # Return the coefficients [a, b, c, d]
    return [a, b, c, d]

# DDG gradient operations on primary edges (for continuum)

def dP(vp1, dim=3, z=1):
    """
    Compute the integrated pressure differential for vertex vp1

    dim=3
    z=1

    Note the routine is the same for 2D and 3D
    """
    dP_i = 0  # Total integrated Laplacian for each vertex
    for vp2 in vp1.nn:  # Iterate over neighboring vertex objects
        # Compute the dual length of e_ij
        e_dual = e_star(vp1, vp2, dim=dim)

        # Compute the area flux for the pressure differential:
        Area = e_dual * 1  # m2, Chosen height was 1 for our 2D test case
        # Compute the dual
        dP_ij = Area * (vp2.P - vp1.P)
        print('-')
        print(f'vp2.x, vp1.x = {vp2.x, vp1.x}')
        print(f'vp2.P - vp1.P = {vp2.P - vp1.P}')
        print(f'Area= {Area}')
        dP_i += dP_ij
    print(f'dP_i = {dP_i}')
    return dP_i


def du(vp1, dim=3):
    """
    Compute the Laplacian of the velocity field for vertex v

    """
    du_i = 0  # Total integrated Laplacian for each vertex
    for vp2 in vp1.nn:  # Iterate over neighboring vertex objects
        l_ij = np.linalg.norm(vp2.x_a - vp1.x_a)
        e_dual = e_star(vp1, vp2, dim=dim)

        w_ij = l_ij / e_dual  # Weight
        if (w_ij is np.inf) or (e_dual == 0):
            continue
        # Compute the dual
        du_ij = np.abs(w_ij) * (vp2.u - vp1.u)
        du_i += du_ij
    print(f'du_i = {du_i}')
    return du_i


def dudt(v, dim=3, mu=8.90 * 1e-4):
    # Equal to the acceleration at a vertex (RHS of equation)
    dudt = -dP(v, dim=dim) + mu * du(v, dim=dim)
    dudt = dudt/v.m  # normalize by mass
    return dudt