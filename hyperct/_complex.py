"""
Base classes for low memory simplicial complex structures.

TODO: -Allow for sub-triangulations to track arbitrary points. Detect which
      simplex it is in and then connect the new points to it
      -Turn the triangulation into a generator that yields a specified number
      of finite points. Ideas:
       https://docs.python.org/2/library/itertools.html#itertools.product
      -Track only origin-suprenum vectors instead of using vertex group struct-
       tures for mesh refinement


TODO: -Approximate vector field if no vfield is field. Construct by finding the
      - average vector field at a vertex???  Note that we can compute vector
      field approximations by solving a LP that fits the scalar approximations
      (dotted with unit vectors)

TODO: -The ugliness of H.V[(0,)] for 1-dimensional complexes

TODO: -Replace split_generation with refine (for limited points)

TODO: -Get rid of Complex.H (vertex group) structures

FUTURE: Triangulate arbitrary domains other than n-cubes
(ex. using delaunay and low disc. sampling subject to constraints, or by adding
     n-cubes and other geometries)

     Starting point:
     An algorithm for automatic Delaunay triangulation of arbitrary planar domains
     https://www.sciencedirect.com/science/article/pii/096599789600004X

    Also note that you can solve a linear system of constraints to find all the
    initial vertices (NP-hard operation)
"""
# Std. Library
import copy
import logging
import os
import itertools
from abc import ABC, abstractmethod
# Required modules:
import numpy

# Optional modules for plotting:
try:
    import matplotlib
    from matplotlib import pyplot
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.tri import Triangulation
    from mpl_toolkits.mplot3d import axes3d, Axes3D, proj3d
    from hyperct._misc import Arrow3D
except ImportError:
    logging.warning("Plotting functions are unavailable. To use install "
                    "matplotlib, install using ex. `pip install matplotlib` ")
    matplotlib_available = False
else:
    matplotlib_available = True

# Optional modules for plotting:
try:
    import clifford as cf
except ImportError:
    logging.warning("Discrete exterior calculus functionality will be "
                    "unavailable, To use install the clifford package with "
                    "`pip install clifford`")
    dec = False
else:
    dec = True

try:
    from functools import lru_cache  # For Python 3 only
except ImportError:  # Python 2:
    import time
    import functools
    import collections
    from hyperct._misc import lru_cache

# Module specific imports
from hyperct._vertex import (VertexCacheIndex, VertexCacheField)
from hyperct._vertex_group import (Subgroup, Cell, Simplex)


# Main complex class:
class Complex:
    def __init__(self, dim, domain=None, sfield=None, sfield_args=(),
                 vfield=None, vfield_args=None,
                 symmetry=False, g_cons=None, g_cons_args=()):
        """
        A base class for a simplicial complex described as a cache of vertices
        together with their connections.

        Important methods:
            Domain triangulation:
                    Complex.triangulate, Complex.split_generation
            Triangulating arbitrary points (must be traingulable,
                may exist outside domain):
                    Complex.triangulate(sample_set)  #TODO
            Converting another simplicial complex structure data type to the
                structure used in Complex (ex. OBJ wavefront)
                    Complex.convert(datatype, data)  #TODO
            Convert the structure in the Complex to other data type:
                    #TODO

        Important objects:
            HC.V: The cache of vertices and their connection
            HC.H: Storage structure of all vertex groups

        :param dim: int, Spatial dimensionality of the complex R^dim
        :param domain: list of tuples, optional
                The bounds [x_l, x_u]^dim of the hyperrectangle space
                ex. The default domain is the hyperrectangle [0, 1]^dim
                Note: The domain must be convex, non-convex spaces can be cut
                      away from this domain using the non-linear
                      g_cons functions to define any arbitrary domain
                      (these domains may also be disconnected from each other)
        :param sfield: A scalar function defined in the associated domain
                           f: R^dim --> R
        :param sfield_args: tuple, Additional arguments to be passed to sfield
        :param vfield: A scalar function defined in the associated domain
                           f: R^dim --> R^m
                       (for example a gradient function of the scalar field)
        :param vfield_args: tuple, Additional arguments to be passed to sfield
        :param symmetry: If all the variables in the field are symmetric this
                option will reduce complexity of the triangulation by O(n!)
        :param g_cons: Constraint functions on the domain g: R^dim --> R^m
        :param g_cons_args: tuple, Additional arguments to be passed to g_cons
        """
        self.dim = dim

        # Domains
        self.domain = domain
        if domain is None:
            self.bounds = [(0, 1), ] * dim
        else:
            self.bounds = domain  # TODO: Assert that len(domain) is dim
        self.symmetry = symmetry  # TODO: Define the functions to be used
        #      here in init to avoid if checks

        # Field functions
        self.sfield = sfield
        self.vfield = vfield
        self.sfield_args = sfield_args
        self.vfield_args = vfield_args

        # Constraint functions
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

        # Homology properties
        self.gen = 0
        self.perm_cycle = 0

        # Every cell is stored in a list of its generation,
        # ex. the initial cell is stored in self.H[0]
        # 1st get new cells are stored in self.H[1] etc.
        # When a cell is sub-generated it is removed from this list

        self.H = []  # Storage structure of vertex groups
        # Cache of all vertices
        if (sfield is not None) or (g_cons is not None):
            self.V = VertexCacheField(field=sfield, field_args=sfield_args,
                                      g_cons=g_cons, g_cons_args=g_cons_args)
        else:
            self.V = VertexCacheIndex()

        self.V_non_symm = []  # Lost of non-symmetric vertices

        if vfield is not None:
            logging.warning("Vector field applications have not been "
                            "implemented yet.")

        if dec:
            self.dec = True
            self.calgebras = {}
        else:
            self.dec = False

    def __call__(self):
        return self.H

    # %% Triangulation methods
    def cyclic_product(self, bounds, origin, supremum, printout=False):
        vo = list(origin)
        vot = tuple(origin)
        vut = tuple(supremum)  # Hyperrectangle supremum
        self.V[vot]
        yield vot
        self.V[vut].connect(self.V[vot])
        yield vut
        # Cyclic group approach with second x_l --- x_u operation.

        # These containers store the "lower" and "upper" vertices
        # corresponding to the origin or supremum of every C2 group.
        # It has the structure of `dim` times embedded lists each containing
        # these vertices as the entire complex grows. Bounds[0] has to be done
        # outside the loops before we have symmetric containers.
        #NOTE: This means that bounds[0][1] must always exist
        C0x = [[self.V[vot]]]
        a_vo = copy.copy(vo)
        a_vo[0] = vut[0]  # Update aN Origin
        self.V[vot].connect(self.V[tuple(a_vo)])
        C1x = [[self.V[tuple(a_vo)]]]
        ab_C = []  # Container for a + b operations

        # Loop over remaining bounds
        for i, x in enumerate(bounds[1:]):
            # Update lower and upper containers
            C0x.append([])
            C1x.append([])
            # try to access a second bound (if not, C1 is symmetric)
            try:
                # Early try so that we don't have to copy the cache before
                # moving on to next C1/C2: Try to add the operation of a new
                # C2 product by accessing the upper bound
                x[1]
                # Copy lists for iteration
                cC0x = [x[:] for x in C0x[:i + 1]]
                cC1x = [x[:] for x in C1x[:i + 1]]
                for j, (VL, VU) in enumerate(zip(cC0x, cC1x)):
                    for k, (vl, vu) in enumerate(zip(VL, VU)):
                        # Build aN vertices for each lower-upper pair in N:
                        a_vl = list(vl.x)
                        a_vu = list(vu.x)
                        a_vl[i + 1] = vut[i + 1]
                        a_vu[i + 1] = vut[i + 1]
                        a_vl = self.V[tuple(a_vl)]
                        #TODO: We can check if the vertex is already in the
                        #      so that we do not yield a non-new vertex,
                        #      however, this might cause a significant slowdown.
                        yield a_vl.x
                        a_vu = self.V[tuple(a_vu)]
                        yield a_vu.x

                        # Connect vertices in N to corresponding vertices
                        # in aN:
                        vl.connect(a_vl)
                        vu.connect(a_vu)

                        # Connect new vertex pair in aN:
                        a_vl.connect(a_vu)

                        # Connect lower pair to upper (triangulation
                        # operation of a + b (two arbitrary operations):
                        vl.connect(a_vu)
                        ab_C.append((vl, a_vu))

                        # Update the containers
                        C0x[i + 1].append(vl)
                        C0x[i + 1].append(vu)
                        C1x[i + 1].append(a_vl)
                        C1x[i + 1].append(a_vu)

                        # Update old containers
                        C0x[j].append(a_vl)
                        C1x[j].append(a_vu)

                # Try to connect aN lower source of previous a + b
                # operation with a aN vertex
                ab_Cc = copy.copy(ab_C)
                for vp in ab_Cc:
                    b_v = list(vp[0].x)  # vl + b
                    ab_v = list(vp[1].x)  # a_vl + b
                    b_v[i + 1] = vut[i + 1]
                    ab_v[i + 1] = vut[i + 1]
                    b_v = self.V[tuple(b_v)]
                    ab_v = self.V[tuple(ab_v)]
                    # Note o---o is already connected
                    vp[0].connect(ab_v)  # o-s
                    b_v.connect(ab_v)  # s-s

                    # Add new list of cross pairs
                    ab_C.append((vp[0], ab_v))
                    ab_C.append((b_v, ab_v))

            except IndexError:
                # Add new group N + aN group supremum, connect to all
                # Get previous
                vs = C1x[i][-1]
                a_vs = list(C1x[i][-1].x)
                a_vs[i + 1] = vut[i + 1]
                a_vs = self.V[tuple(a_vs)]

                # Connect a_vs to vs (the nearest neighbour in N --- aN)
                a_vs.connect(vs)

                # Update the containers (only 2 new entries)
                C0x[i + 1].append(vs)
                C1x[i + 1].append(a_vs)

                # Loop over lower containers. Connect lower pair to a_vs
                # triangulation operation of a + b (two arbitrary operations):
                cC0x = [x[:] for x in C0x[:i + 1]]
                for j, VL in enumerate(cC0x):
                    for k, vu in enumerate(VL):
                        if vu is not a_vs:
                            vu.connect(a_vs)
                            #NOTE: Only needed when there will be no more
                            #      symmetric points later on

            # Printing
            if printout:
                print("=" * 19)
                print("Current symmetry group:")
                print("=" * 19)
                # for v in self.C0():
                #   v.print_out()
                for v in self.V.cache:
                    self.V[v].print_out()

                print("=" * 19)

        # Clean class trash
        try:
            del C0x
            del cC0x
            del C1x
            del cC1x
            del ab_C
            del ab_Cc
        except UnboundLocalError:
            pass


    def triangulate_c(self, n=None, symmetry=None, printout=False):
        """
        Triangulate the initial domain, if n is not None then a limited number
        of points will be generated

        :param n:
        :param symmetry:

            Ex. Dictionary/hashtable
            f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            symmetry = symmetry[0]: 0,  # Variable 1
                       symmetry[1]: 0,  # symmetric to variable 1
                       symmetry[2]: 0,  # symmetric to variable 1
                       symmetry[3]: 3,  # Variable 4
                       symmetry[4]: 3,  # symmetric to variable 4
                       symmetry[5]: 3,  # symmetric to variable 4
                        }

        :param printout:
        :return:

        NOTES:
        ------
        Rather than using the combinatorial algorithm to connect vertices we
        make the following observation:

        The bound pairs are similar a C2 cyclic group and the structure is
        formed using the cartesian product:

        H = C2 x C2 x C2 ... x C2 (dim times)

        So construct any normal subgroup N and consider H/N first, we connect
        all vertices within N (ex. N is C2 (the first dimension), then we move
        to a left coset aN (an operation moving around the defined H/N group by
        for example moving from the lower bound in C2 (dimension 2) to the
        higher bound in C2. During this operation connection all the vertices.
        Now repeat the N connections. Note that these elements can be connected
        in parrallel.
        """
        # Build origin and supremum vectors
        origin = [i[0] for i in self.bounds]
        self.origin = origin
        supremum = [i[1] for i in self.bounds]

        self.supremum = supremum

        #TODO: Add check that len(symmetry) is equal to len(self.bounds)
        if symmetry is None:
            cbounds = self.bounds
        else:
            cbounds = copy.copy(self.bounds)
            for i, j in enumerate(symmetry):
                if i is not j:
                    # pop second entry on second symmetry vars
                    cbounds[i] = [self.bounds[symmetry[i]][0]]
                    # Sole (first) entry is the sup value and there is no origin
                    cbounds[i] = [self.bounds[symmetry[i]][1]]
                    if self.bounds[symmetry[i]] is not self.bounds[symmetry[j]]:
                        logging.warning(f"Variable {i} was specified as "
                                        f"symmetetric to variable {j}, however,"
                                        f"the bounds {i} ="
                                        f" {self.bounds[symmetry[i]]} and {j} ="
                                        f" {self.bounds[symmetry[j]]} do not "
                                        f"match, the mismatch was ignored in "
                                        f"the initial triangulation.")
                        cbounds[i] = self.bounds[symmetry[j]]


        if n is None:
            # Build generator
            self.cp = self.cyclic_product(cbounds, origin, supremum, printout)
            for i in self.cp:
                print(f"Yield = {i}")


        else:
            #Check if generator already exists
            try:
                self.cp
            except (AttributeError, KeyError):
                self.cp = self.cyclic_product(cbounds, origin, supremum,
                                              symmetry, printout)



                i#print(f'Big outside gen i = {i}')

        # Replace with limited iterator and next()



        #for vg in self.vgen:

            #self.V[tuple(origin)].connect(self.V[vg])
            #self.V[tuple(supremum)].connect(self.V[vg])

        # Save the triangulated space for future refinement
        self.triangulated_vectors = [(self.origin, self.supremum)]

        if printout:
            print("=" * 19)
            print("Initial hyper cube:")
            print("=" * 19)
            # for v in self.C0():
            #   v.print_out()
            for v in self.V.cache:
                self.V[v].print_out()

            print("=" * 19)


    def triangulate(self, domain=None, n=None, symm=None):
        """

        :param n: Limited number of points to generate

        :

        Triangulate a domain in [x_l, x_u]^dim \in R^dim specified by bounds and
        constraints.

        If domain is None the default domain is the hyperrectangle [0, 1]^dim

        FUTURE: Currently only hyperrectangle domains are possible. In the
                future we'd like to define more complex domains.
        """
        # Generate n-cube here:
        self.H.append([])
        self.n_cube(symmetry=self.symmetry, printout=1)

        # TODO: Assign functions to a the complex instead
        if self.symmetry:
            self.generation_cycle = 1
            # self.centroid = self.C0()[-1].x
            # self.C0.centroid = self.centroid
        else:
            self.add_centroid()

        # Build initial graph
        self.graph_map()

        if self.domain is not None:
            # Delete the vertices generated during n_cube
            # del(self.V)
            #TODO: We need a method with using the field for non-cube bounds
            self.V = VertexCacheField(field=self.sfield,
                                      field_args=self.sfield_args,
                                      g_cons=self.g_cons,
                                      g_cons_args=self.g_cons_args)
            # TODO: Find a way not to delete the entire vertex cache in situations
            # where this method is used to triangulate the domain together with
            # other in place connections. ex simply move n_cube to if statement
            # and use a temporary cache

            # Construct the initial spatial vector
            origin = []  # origin of complex domain vector
            supremum = []  # supremum of complex domain vector
            for i, (lb, ub) in enumerate(self.domain):
                origin.append(lb)
                supremum.append(ub)
                # x_a[i] = x_a[i] * (ub - lb) + lb
            # del(self.C0)
            self.origin = tuple(origin)
            self.supremum = tuple(supremum)
            # self.C0 =
            self.construct_hypercube(self.origin, self.supremum, 0, 0)

            # TODO: Find new C0 by looping through C_0 and checking if v in Cnew
            #      Then delete unused C0 and set Cnew to C_0

            # x_a = numpy.array(x, dtype=float)
            # if self.domain is not None:
            #    for i, (lb, ub) in enumerate(self.domain):
            #        x_a[i] = x_a[i] * (ub - lb) + lb
        else:
            self.H[0].append(self.C0)

        # TODO: Create a self.n_cube_finite to generate an initial complex for
        # a finite number of points.

        if (self.sfield is not None) or (self.g_cons is not None):
            self.hgr = self.C0.homology_group_rank()
            self.hgrd = 0  # Complex group rank differential
            self.hgr = self.C0.hg_n

    def n_cube(self, symmetry=False, printout=False):
        """
        Generate the simplicial triangulation of the n dimensional hypercube
        containing 2**n vertices
        """
        # TODO: Check for loaded data and load if available
        import numpy
        origin = list(numpy.zeros(self.dim, dtype=int))
        self.origin = origin
        supremum = list(numpy.ones(self.dim, dtype=int))
        self.supremum = supremum

        # tuple versions for indexing
        origintuple = tuple(origin)
        supremumtuple = tuple(supremum)

        x_parents = [origintuple]

        if symmetry:
            self.C0 = Simplex(0, 0, 0, self.dim)  # Initial cell object
            self.C0.add_vertex(self.V[origintuple])

            i_s = 0
            self.perm_symmetry(i_s, x_parents, origin)
            self.C0.add_vertex(self.V[supremumtuple])
        else:
            self.C0 = Cell(0, 0, origin, supremum)  # Initial cell object
            self.C0.add_vertex(self.V[origintuple])
            self.C0.add_vertex(self.V[supremumtuple])

            i_parents = []
            self.perm(i_parents, x_parents, origin)

        if printout:
            print("Initial hyper cube:")
            for v in self.C0():
                v.print_out()

    def n_rec(self):
        raise NotImplementedError("To implement this simply run n_cube then "
                                  "create a new self.C0 based on the n_cube"
                                  "results.")

    def perm(self, i_parents, x_parents, xi):
        # TODO: Cut out of for if outside linear constraint cutting planes
        xi_t = tuple(xi)

        # Construct required iterator
        iter_range = [x for x in range(self.dim) if x not in i_parents]

        for i in iter_range:
            i2_parents = copy.copy(i_parents)
            i2_parents.append(i)
            xi2 = copy.copy(xi)
            xi2[i] = 1
            # Make new vertex list a hashable tuple
            xi2_t = tuple(xi2)
            # Append to cell
            self.C0.add_vertex(self.V[xi2_t])
            # Connect neighbours and vice versa
            # Parent point
            self.V[xi2_t].connect(self.V[xi_t])

            # Connect all family of simplices in parent containers
            for x_ip in x_parents:
                self.V[xi2_t].connect(self.V[x_ip])

            x_parents2 = copy.copy(x_parents)
            x_parents2.append(xi_t)

            # Permutate
            self.perm(i2_parents, x_parents2, xi2)

    def perm_symmetry(self, i_s, x_parents, xi):
        # TODO: Cut out of for if outside linear constraint cutting planes
        xi_t = tuple(xi)
        xi2 = copy.copy(xi)
        xi2[i_s] = 1
        # Make new vertex list a hashable tuple
        xi2_t = tuple(xi2)
        # Append to cell
        self.C0.add_vertex(self.V[xi2_t])
        # Connect neighbours and vice versa
        # Parent point
        self.V[xi2_t].connect(self.V[xi_t])

        # Connect all family of simplices in parent containers
        for x_ip in x_parents:
            self.V[xi2_t].connect(self.V[x_ip])

        x_parents2 = copy.copy(x_parents)
        x_parents2.append(xi_t)

        i_s += 1
        if i_s == self.dim:
            return
        # Permutate
        self.perm_symmetry(i_s, x_parents2, xi2)

    def add_centroid(self):
        """Split the central edge between the origin and supremum of
        a cell and add the new vertex to the complex"""
        self.centroid = list(
            (numpy.array(self.origin) + numpy.array(self.supremum)) / 2.0)
        self.C0.add_vertex(self.V[tuple(self.centroid)])
        self.C0.centroid = self.centroid

        # Disconnect origin and supremum
        self.V[tuple(self.origin)].disconnect(self.V[tuple(self.supremum)])

        # Connect centroid to all other vertices
        for v in self.C0():
            self.V[tuple(self.centroid)].connect(self.V[tuple(v.x)])

        self.centroid_added = True
        return

    # %% Construct incidence array:
    def incidence(self):
        """
        TODO: Find directed (if sfield is not none) array over whole complex
        :return:
        """
        if self.centroid_added:
            self.structure = numpy.zeros([2 ** self.dim + 1, 2 ** self.dim + 1],
                                         dtype=int)
        else:
            self.structure = numpy.zeros([2 ** self.dim, 2 ** self.dim],
                                         dtype=int)

        for v in self.HC.C0():
            for v2 in v.nn:
                self.structure[v.index, v2.index] = 1

        return

    # %% A more sparse incidence generator:
    def graph_map(self):
        """ Make a list of size 2**n + 1 where an entry is a vertex
        incidence, each list element contains a list of indexes
        corresponding to that entries neighbours"""

        self.graph = [[v2.index for v2 in v.nn] for v in self.C0()]

    # Graph structure method:
    # 0. Capture the indices of the initial cell.
    # 1. Generate new origin and supremum scalars based on current generation
    # 2. Generate a new set of vertices corresponding to a new
    #    "origin" and "supremum"
    # 3. Connected based on the indices of the previous graph structure
    # 4. Disconnect the edges in the original cell

    # %% Refinement
    # % Refinement based on vector partitions

    def refine(self):
        """
        Refine the entire domain of the current complex
        :return:
        """
        print(f'self.triangulated_vectors = {self.triangulated_vectors}')
        tvs = copy.copy(self.triangulated_vectors)
        for vp in tvs:
            print(f'tvs = {tvs}')
            self.refine_local_space(*vp)
            self.triangulated_vectors.remove(vp)#

    def refine_local_space2(self, origin, supremum):
        """
        Refines the inside the hyperrectangle captured by the vector

        #TODO: Ensure correct dimensions in input vectors

        :param origin: vector origin tuple/list
        :param supremum: vector supremum tuple/list
        :return:
        """
        vot = tuple(origin)
        vst = tuple(supremum)
        print('='*20)
        print(f'origin = {origin}')
        print(f'supremum = {supremum}')
        print(f'vot = {vot}')
        print(f'vst = {vst}')
        print('=' * 20)
        # Initiate vertices in case they don't exist
        vo = self.V[vot]
        vs = self.V[vst]

        # Disconnect the origin and supremum
        vo.disconnect(vs)

        #IN NEW METHOD WE NEED TO RUN THIS TO DISCONNECT EVERYTHING\

        # Find the lower/upper bounds of the refinement hyperrectangle
        bl = list(vot)
        bu = list(vst)
        print(f'bl = {bl}')
        print(f'bu = {bu}')
        for i, (voi, vsi) in enumerate(zip(vot, vst)):
            print(f'i = {i}')
            print(f'voi = {voi}')
            print(f'vsi = {vsi}')
            if bl[i] > vsi:
                bl[i] = vsi
            if bu[i] < voi:
                bu[i] = voi


        #TODO: These for loops can easily be replaced by numpy operations,
        #      tests should be run to determine which method is faster.
        #      NOTE: This is mostly done with sets/lists because we aren't sure
        #            how well the numpy arrays will scale to thousands of
        #             variables.
        vn_pool = set()
        vn_pool.update(vo.nn)
        vn_pool.update(vs.nn)
        print(f'vn_pool = {vn_pool}')
        cvn_pool = copy.copy(vn_pool)
        for vn in cvn_pool:
            for i, xi in enumerate(vn.x):
                #print(f' bl[i] <= xi and xi <= bu[i] '
                #      f'= {bl[i] <= xi and xi <= bu[i]}')
               # print(f'vn.x = {vn.x}')
               # print(f'xi = {xi}')
               # print(f'bl = {bl}')
               # print(f'bu = {bu}')
                if bl[i] <= xi <= bu[i]:
                    pass
                else:
                    try:
                        vn_pool.remove(vn)
                    except KeyError:
                        pass  #NOTE: Not all neigbouds are in initial pool
        # Build centroid
        # vca = (vo.x_a + vs.x_a) / 2.0
        vca = (vs.x_a - vo.x_a) / 2.0 + vo.x_a
        vc = self.V[tuple(vca)]
        print(f'vc.x = {vc.x}')

        # Connect the origin and supremum  to the centroid
        # vo.disconnect(vs)
        vc.connect(vo)
        vc.connect(vs)

        for vn in vn_pool:
            print('-'*5)
            print(f'vn.x = {vn.x}')
            print('-'*5)
            # Disconnect with origin vertex
            vn.disconnect(vo)
            #Disconnect with supremum vertex
            vn.disconnect(vs)

            # Create the new vertex to connect to vo and von
            vjt = (vn.x_a - vo.x_a) / 2.0 + vo.x_a
            print(f'vjt (vo---vn) = {vjt}')
            vj = self.V[tuple(vjt)]
            vj.connect(vo)
            vj.connect(vn)
            # Connect the vertices to the centroid (vo is already connected)
            vj.connect(vc)
            vn.connect(vc)

            # Create the new vertex to connect to vs and vn
            vkt = (vn.x_a - vs.x_a) / 2.0 + vs.x_a
            print(f'vkt (vs---vn) = {vkt}')
            vk = self.V[tuple(vkt)]
            vk.connect(vs)
            vk.connect(vn)

            # Connect the vertices to the centroid (vo is already connected)
            vk.connect(vc)
            vn.connect(vc)

            # Append the newly triangulated search spaces for future refinement
            self.triangulated_vectors.append((vc.x, vn.x))
            print(f'self.triangulated_vectors.append({(vc.x, vn.x)})')
            #print(f'self.triangulated_vectors = { self.triangulated_vectors}')

        #self.triangulated_vectors.append((vc.x, vo.x))
        print(f'self.triangulated_vectors.append({(vc.x, vo.x)})')
        #self.triangulated_vectors.append((vc.x, vs.x))
        print(f'self.triangulated_vectors.append({(vc.x, vs.x)})')

        # Pool all neighbours
        if 0:
            print(f'vo.nn = {vo.nn}')
            von_pool = []
            vsn_pool = []
            for von in vo.nn:
                print(f'von = {von}')
                print(f'von.x = {von.x}')
                print(f'von.x_a = {von.x_a}')
                for i, xi in enumerate(von.x):
                    print(f'i = {i}')
                    print(f'xi = {xi}')
                    print(f'von[i] = {von.x[i]}')
                    print(f'xi <= von[i] = {xi <= von.x[i]}')
                    if xi <= von.x[i]:
                        break  # The else statement will run breaking main loop
                else:
                    break  # This breaks back to loop "for von in vo.nn:"

                # If no breaks we can add to pool
                von_pool.append(von)

            for vsn in vs.nn:
                print(f'vsn.x = {vsn.x}')
                for i, xi in enumerate(vsn.x):
                    print(f'i = {i}')
                    print(f'xi = {xi}')
                    print(f'vs[i] = {vsn.x[i]}')
                    print(f'xi >= von[i] = {xi >= vsn.x[i]}')
                    if xi >= vsn.x[i]:
                        break  # The else statement will run breaking main loop
                else:
                    break  # This breaks back to loop "for von in vo.nn:"

                # If no breaks we can add to pool
                vsn_pool.append(von)

            print(f'von_pool = {von_pool}')
            print(f'vsn_pool = {vsn_pool}')

            # Build centroid
            vca = (vo.x_a + vs.x_a)/2.0
            vc = self.V[tuple(vca)]
            print(f'vc.x = {vc.x}')

            # Connect the origin and supremum  to the centroid
            #vo.disconnect(vs)
            vc.connect(vo)
            vc.connect(vs)

            print(f'vc.nn = {vc.nn}')
            for von in von_pool:
                # Disconnect with origin vertex
                von.disconnect(vo)

                # Create the new vertex to connect to vo and von
                vnt = (vo.x_a + von.x_a) / 2.0
                vn = self.V[tuple(vnt)]
                vn.connect(vo)
                vn.connect(von)

                # Connect the vertices to the centroid (vo is already connected)
                von.connect(vc)
                vn.connect(vc)

            for vsn in vsn_pool:
                # Disconnect with origin vertex
                vsn.disconnect(vs)

                # Create the new vertex to connect to vo and von
                vnt = (vs.x_a + vsn.x_a) / 2.0
                vn = self.V[tuple(vnt)]
                vn.connect(vs)
                vn.connect(vsn)

                # Connect the vertices to the centroid (vo is already connected)
                vsn.connect(vc)
                vn.connect(vc)

    def refine_local_space(self, origin, supremum):
        """
        Refines the inside the hyperrectangle captured by the vector

        #TODO: Ensure correct dimensions in input vectors

        :param origin: vector origin tuple/list
        :param supremum: vector supremum tuple/list
        :return:
        """
        vot = tuple(origin)
        vst = tuple(supremum)
        print('='*20)
        print(f'origin = {origin}')
        print(f'supremum = {supremum}')
        print(f'vot = {vot}')
        print(f'vst = {vst}')
        print('=' * 20)
        # Initiate vertices in case they don't exist
        vo = self.V[vot]
        vs = self.V[vst]

        # Disconnect the origin and supremum
        vo.disconnect(vs)

        # Find the lower/upper bounds of the refinement hyperrectangle
        bl = list(vot)
        bu = list(vst)
        print(f'bl = {bl}')
        print(f'bu = {bu}')
        for i, (voi, vsi) in enumerate(zip(vot, vst)):
            print(f'i = {i}')
            print(f'voi = {voi}')
            print(f'vsi = {vsi}')
            if bl[i] > vsi:
                bl[i] = vsi
            if bu[i] < voi:
                bu[i] = voi


        #TODO: These for loops can easily be replaced by numpy operations,
        #      tests should be run to determine which method is faster.
        #      NOTE: This is mostly done with sets/lists because we aren't sure
        #            how well the numpy arrays will scale to thousands of
        #             variables.
        vn_pool = []
        vn_pool = set()
        vn_pool.update(vo.nn)
        vn_pool.update(vs.nn)
        print(f'vn_pool = {vn_pool}')
        cvn_pool = copy.copy(vn_pool)
        for vn in cvn_pool:
            for i, xi in enumerate(vn.x):
                #print(f' bl[i] <= xi and xi <= bu[i] '
                #      f'= {bl[i] <= xi and xi <= bu[i]}')
               # print(f'vn.x = {vn.x}')
               # print(f'xi = {xi}')
               # print(f'bl = {bl}')
               # print(f'bu = {bu}')
                if bl[i] <= xi <= bu[i]:
                    pass
                else:
                    try:
                        vn_pool.remove(vn)
                    except KeyError:
                        pass  #NOTE: Not all neigbouds are in initial pool
        # Build centroid
        # vca = (vo.x_a + vs.x_a) / 2.0
        vca = (vs.x_a - vo.x_a) / 2.0 + vo.x_a
        vc = self.V[tuple(vca)]
        print(f'vc.x = {vc.x}')

        # Connect the origin and supremum  to the centroid
        # vo.disconnect(vs)
        vc.connect(vo)
        vc.connect(vs)

        for vn in vn_pool:
            print('-'*5)
            print(f'vn.x = {vn.x}')
            print('-'*5)
            # Disconnect with origin vertex
            vn.disconnect(vo)
            #Disconnect with supremum vertex
            vn.disconnect(vs)

            # Create the new vertex to connect to vo and von
            vjt = (vn.x_a - vo.x_a) / 2.0 + vo.x_a
            print(f'vjt (vo---vn) = {vjt}')
            vj = self.V[tuple(vjt)]
            vj.connect(vo)
            vj.connect(vn)
            # Connect the vertices to the centroid (vo is already connected)
            vj.connect(vc)
            vn.connect(vc)

            # Create the new vertex to connect to vs and vn
            vkt = (vn.x_a - vs.x_a) / 2.0 + vs.x_a
            print(f'vkt (vs---vn) = {vkt}')
            vk = self.V[tuple(vkt)]
            vk.connect(vs)
            vk.connect(vn)

            # Connect the vertices to the centroid (vo is already connected)
            vk.connect(vc)
            vn.connect(vc)

            # Append the newly triangulated search spaces for future refinement
            self.triangulated_vectors.append((vc.x, vn.x))
            print(f'self.triangulated_vectors.append({(vc.x, vn.x)})')
            #print(f'self.triangulated_vectors = { self.triangulated_vectors}')

        #self.triangulated_vectors.append((vc.x, vo.x))
        print(f'self.triangulated_vectors.append({(vc.x, vo.x)})')
        #self.triangulated_vectors.append((vc.x, vs.x))
        print(f'self.triangulated_vectors.append({(vc.x, vs.x)})')

        # Pool all neighbours
        if 0:
            print(f'vo.nn = {vo.nn}')
            von_pool = []
            vsn_pool = []
            for von in vo.nn:
                print(f'von = {von}')
                print(f'von.x = {von.x}')
                print(f'von.x_a = {von.x_a}')
                for i, xi in enumerate(von.x):
                    print(f'i = {i}')
                    print(f'xi = {xi}')
                    print(f'von[i] = {von.x[i]}')
                    print(f'xi <= von[i] = {xi <= von.x[i]}')
                    if xi <= von.x[i]:
                        break  # The else statement will run breaking main loop
                else:
                    break  # This breaks back to loop "for von in vo.nn:"

                # If no breaks we can add to pool
                von_pool.append(von)

            for vsn in vs.nn:
                print(f'vsn.x = {vsn.x}')
                for i, xi in enumerate(vsn.x):
                    print(f'i = {i}')
                    print(f'xi = {xi}')
                    print(f'vs[i] = {vsn.x[i]}')
                    print(f'xi >= von[i] = {xi >= vsn.x[i]}')
                    if xi >= vsn.x[i]:
                        break  # The else statement will run breaking main loop
                else:
                    break  # This breaks back to loop "for von in vo.nn:"

                # If no breaks we can add to pool
                vsn_pool.append(von)

            print(f'von_pool = {von_pool}')
            print(f'vsn_pool = {vsn_pool}')

            # Build centroid
            vca = (vo.x_a + vs.x_a)/2.0
            vc = self.V[tuple(vca)]
            print(f'vc.x = {vc.x}')

            # Connect the origin and supremum  to the centroid
            #vo.disconnect(vs)
            vc.connect(vo)
            vc.connect(vs)

            print(f'vc.nn = {vc.nn}')
            for von in von_pool:
                # Disconnect with origin vertex
                von.disconnect(vo)

                # Create the new vertex to connect to vo and von
                vnt = (vo.x_a + von.x_a) / 2.0
                vn = self.V[tuple(vnt)]
                vn.connect(vo)
                vn.connect(von)

                # Connect the vertices to the centroid (vo is already connected)
                von.connect(vc)
                vn.connect(vc)

            for vsn in vsn_pool:
                # Disconnect with origin vertex
                vsn.disconnect(vs)

                # Create the new vertex to connect to vo and von
                vnt = (vs.x_a + vsn.x_a) / 2.0
                vn = self.V[tuple(vnt)]
                vn.connect(vs)
                vn.connect(vsn)

                # Connect the vertices to the centroid (vo is already connected)
                vsn.connect(vc)
                vn.connect(vc)

    # % Split symmetric generations
    def split_generation(self):
        """
        Run sub_generate_cell for every cell in the current complex self.gen
        """
        no_splits = False  # USED IN SHGO
        try:
            for c in self.H[self.gen]:
                if self.symmetry:
                    # self.sub_generate_cell_symmetry(c, self.gen + 1)
                    self.split_simplex_symmetry(c, self.gen + 1)
                else:
                    self.sub_generate_cell(c, self.gen + 1)
        except IndexError:
            no_splits = True  # USED IN SHGO

        self.gen += 1
        return no_splits  # USED IN SHGO

    def sub_generate_cell(self, C_i, gen):
        """Subgenerate a cell `C_i` of generation `gen` and
        homology group rank `hgr`."""
        origin_new = tuple(C_i.centroid)
        centroid_index = len(C_i()) - 1

        # If not gen append
        try:
            self.H[gen]
        except IndexError:
            self.H.append([])

        # Generate subcubes using every extreme vertex in C_i as a supremum
        # and the centroid of C_i as the origin
        H_new = []  # list storing all the new cubes split from C_i
        for i, v in enumerate(C_i()[:-1]):
            supremum = tuple(v.x)
            H_new.append(
                self.construct_hypercube(origin_new, supremum, gen, C_i.hg_n))

        for i, connections in enumerate(self.graph):
            # Present vertex V_new[i]; connect to all connections:
            if i == centroid_index:  # Break out of centroid
                break

            for j in connections:
                C_i()[i].disconnect(C_i()[j])

        # Destroy the old cell
        if C_i is not self.C0:  # Garbage collector does this anyway; not needed
            del C_i

        # TODO: Recalculate all the homology group ranks of each cell
        return H_new

    # @lru_cache(maxsize=None)
    def construct_hypercube(self, origin, supremum, gen, hgr,
                            printout=False):
        """
        Construct a hypercube from the origin graph

        :param origin:
        :param supremum:
        :param gen:
        :param hgr:
        :param printout:
        :return:
        """
        # Initiate new cell
        C_new = Cell(gen, hgr, origin, supremum)
        C_new.centroid = tuple(
            (numpy.array(origin) + numpy.array(supremum)) / 2.0)

        # Cached calculation
        # print(f'self.C0 = {self.C0()}')
        # print(f'self.C0 = {self.C0()[self.graph[0]]}')
        # [self.C0()[index] for index in self.graph[i]]

        self.v_o = numpy.array(origin)
        self.v_s = numpy.array(supremum)
        for i, v in enumerate(self.C0()[:-1]):  # Build new vertices
            # print(f'v.x = {v.x}')
            t1 = self.generate_sub_cell_t1(origin, v.x)
            # print(t1)
            t2 = self.generate_sub_cell_t2(supremum, v.x)
            # print(t2)
            vec = t1 + t2
            # print(f'vec = {vec}')
            vec = tuple(vec)
            # nn_v = [self.C0()[index] for index in self.graph[i]]
            # C_new.add_vertex(self.V.__getitem__(vec, nn=nn_v))
            C_new.add_vertex(self.V[vec])
            # print(f'self.V[vec].x = {self.V[vec].x}')
            # print(f'C_new() = {C_new()}')

        # Add new centroid
        C_new.add_vertex(self.V[C_new.centroid])

        # print(C_new())
        # print(self.C0())

        for i, v in enumerate(C_new()):  # Connect new vertices
            nn_v = [C_new()[index] for index in self.graph[i]]
            self.V[v.x].nn.update(nn_v)

        # nn_v = [C_new()[index] for index in self.graph[-1]]
        # C_new.add_vertex(self.V.__getitem__(C_new.centroid, nn_v))

        # C_new.add_vertex(self.V.__getitem__(vec, nn=nn_v))
        # Add new centroid
        # C_new.add_vertex(self.V[C_new.centroid])

        # V_new.append(C_new.centroid)

        if printout:
            print("A sub hyper cube with:")
            print("origin: {}".format(origin))
            print("supremum: {}".format(supremum))
            for v in C_new():
                v.print_out()

        # Append the new cell to the to complex
        self.H[gen].append(C_new)
        return C_new

    def split_simplex_symmetry(self, S, gen):
        """
        Split a hypersimplex S into two sub simplices by building a hyperplane
        which connects to a new vertex on an edge (the longest edge in
        dim = {2, 3}) and every other vertex in the simplex that is not
        connected to the edge being split.

        This function utilizes the knowledge that the problem is specified
        with symmetric constraints

        The longest edge is tracked by an ordering of the
        vertices in every simplices, the edge between first and second
        vertex is the longest edge to be split in the next iteration.
        """
        # If not gen append
        try:
            self.H[gen]
        except IndexError:
            self.H.append([])

        # Find new vertex.
        # V_new_x = tuple((numpy.array(C()[0].x) + numpy.array(C()[1].x)) / 2.0)
        s = S()
        firstx = s[0].x
        lastx = s[-1].x
        V_new = self.V[tuple((numpy.array(firstx) + numpy.array(lastx)) / 2.0)]

        # Disconnect old longest edge
        self.V[firstx].disconnect(self.V[lastx])

        # Connect new vertices to all other vertices
        for v in s[:]:
            v.connect(self.V[V_new.x])

        # New "lower" simplex
        S_new_l = Simplex(gen, S.hg_n, self.generation_cycle,
                          self.dim)
        S_new_l.add_vertex(s[0])
        S_new_l.add_vertex(V_new)  # Add new vertex
        for v in s[1:-1]:  # Add all other vertices
            S_new_l.add_vertex(v)

        # New "upper" simplex
        S_new_u = Simplex(gen, S.hg_n, S.generation_cycle, self.dim)

        # First vertex on new long edge
        S_new_u.add_vertex(s[S_new_u.generation_cycle + 1])

        for v in s[1:-1]:  # Remaining vertices
            S_new_u.add_vertex(v)

        for k, v in enumerate(s[1:-1]):  # iterate through inner vertices
            if k == S.generation_cycle:
                S_new_u.add_vertex(V_new)
            else:
                S_new_u.add_vertex(v)

        S_new_u.add_vertex(s[-1])  # Second vertex on new long edge

        self.H[gen].append(S_new_l)
        self.H[gen].append(S_new_u)

        return

    @lru_cache(maxsize=None)
    def generate_sub_cell_t1(self, origin, v_x):
        # TODO: Test if looping lists are faster
        return self.v_o - self.v_o * numpy.array(v_x)

    @lru_cache(maxsize=None)
    def generate_sub_cell_t2(self, supremum, v_x):
        return self.v_s * numpy.array(v_x)


    def split_generation_non_symm(self):
        """
        Disconnect the non-symmetric vertices and reconnect them to a smaller
        simplex after splitting the generation.
        :return:
        """
        print(self.V_non_symm)
        non_sym_current = self.V_non_symm.copy()

        # Find the candidate star set by finding all neighbours of the previous
        # connections of every vertex v in the non-symmetric set and add those
        # connections to a set containing all their new connections.
        # This allows us to connect all the new connections that should be
        # inside new the simplex containing each v (but also connections far
        # away from the new simplex).
        for v in non_sym_current:
            vnn = v.nn.copy()  # The set of the new candidate star domain
            vnn_i = vnn.copy()  # The initial simplex containing v
            # Add all the neighbours' connections to vnn set
            for v2 in vnn_i:
                # Set union all neighbours with potential star domain
                vnn = vnn.union(v2.nn)
                # Disconnect the former edge
                v.disconnect(v2)

            # Build simplex from initial simplex containing v
            S = self.v_array(vset=vnn_i)

            # Now filter out the connections that are not contained in the vnn_i
            # simplex. This vastly reduces the number of combinatorial
            # operations that will be required in connect_vertex_non_symm
            # Filter the star domain:
            vnn_f = set()
            for v2 in vnn:
                print(f'filter test for v2.x = {v2.x}')
                print(f'self.in_simplex(S, v2.x) = {self.in_simplex(S, v2.x)}')
                #TODO: Fix in_simplex when vertex is on edge of simplex it does
                #      not work
                if self.in_simplex(S, v2.x):
                    vnn_f.add(v)

            print(f'vnn_f = {vnn_f}')

            vnn_f = vnn  #TODO: REMOVE AFTER FIXING FILTER
            print(f'vnn_f = {vnn_f}')
            self.connect_vertex_non_symm(v.x, near=vnn_f)

        return

    def connect_vertex_non_symm(self, v_x, near=None):
        """
        Adds a vertex at coords v_x to the complex that is not symmetric to the
        initial triangulation and sub-triangulation.

        If near is specified (for example; a star domain or collections of
        cells known to contain v) then only those simplices containd in near
        will be searched, this greatly speeds up the process.

        If near is not specified this method will search the entire simplicial
        complex structure.

        :param v_x: tuple, coordinates of non-symmetric vertex
        :param near: set or list of vertices, these are points near v to check for
        :return:
        """
        #TODO: if near is None assign all vertices in cache to star
        #TODO: if dim == 1 routine
        star = near
        # Create the vertex origin

        #TODO: TEST if v_x is not already a vertex (then return if in cache)
        if tuple(v_x) in self.V.cache:
            if self.V[v_x] in self.V_non_symm:
                pass
            else:
                return



        self.V[v_x]
        found_nn = False
        S_rows = []
        for v in star:
            S_rows.append(v.x)

        S_rows = numpy.array(S_rows)
        A = numpy.array(S_rows) - numpy.array(v_x)
        #Iterate through all the possible simplices of S_rows
        for s_i in itertools.combinations(range(S_rows.shape[0]),
                                          r=self.dim + 1):
            # Check if connected, else s_i is not a simplex
            valid_simplex = True
            for i in itertools.combinations(s_i, r=2):
                # Every combination of vertices must be connected, we check of
                # the current iteration of all combinations of s_i are connected
                # we break the loop if it is not.
                if ((not self.V[tuple(S_rows[i[1]])] in
                        self.V[tuple(S_rows[i[0]])].nn)
                    and (not (self.V[tuple(S_rows[i[0]])] in
                        self.V[tuple(S_rows[i[1]])].nn))):
                    valid_simplex = False
                    break  #TODO: Review this

            S = S_rows[[s_i]]
            if valid_simplex:
                if self.deg_simplex(S, proj=None):
                    valid_simplex = False

            # If s_i is a valid simplex we can test if v_x is inside si
            if valid_simplex:
                # Find the A_j0 value from the precalculated values
                A_j0 = A[[s_i]]
                if self.in_simplex(S, v_x, A_j0):
                    found_nn = True
                    break  # breaks the main for loop, s_i is the target simplex

        # Connect the simplex to point
        if found_nn:
            for i in s_i:
                self.V[v_x].connect(self.V[tuple(S_rows[i])])
        """
        gen = itertools.product(range(2), repeat=3)
        >>> for g in gen:
                print(g)
        """

        # Attached the simplex to storage for all non-symmetric vertices
        self.V_non_symm.append(self.V[v_x])
        #TODO: Disconnections? Not needed?
        return found_nn  # this bool value indicates a successful connection if True

    def in_simplex(self, S, v_x, A_j0=None):
        """
        Check if a vector v_x is in simplex S
        :param S: array containing simplex entries of vertices as rows
        :param v_x: a candidate vertex
        :param A_j0: array, optional, allows for A_j0 to be pre-calculated
        :return: boolean, if True v_x is in S, if False v_x is not in S

        Notes:
        https://stackoverflow.com/questions/21819132/how-do-i-check-if-a-simplex-contains-the-origin
        """
        A_11 = numpy.delete(S, 0, 0) - S[0]

        sign_det_A_11 = numpy.sign(numpy.linalg.det(A_11))
        if sign_det_A_11 == 0:
            #NOTE: We keep the variable A_11, but we loop through A_jj
            #ind=
            #while sign_det_A_11 == 0:
            #    A_11 = numpy.delete(S, ind, 0) - S[ind]
            #    sign_det_A_11 = numpy.sign(numpy.linalg.det(A_11))

            sign_det_A_11 = -1  #TODO: Choose another det of j instead?
            #TODO: Unlikely to work in many cases

        if A_j0 is None:
            A_j0 = S - v_x

        for d in range(self.dim + 1):
            det_A_jj = (-1)**d * sign_det_A_11
            #TODO: Note taht scipy might be faster to add as an optional
            #      dependency
            sign_det_A_j0 = numpy.sign(numpy.linalg.det(numpy.delete(A_j0, d, 0)))
            if det_A_jj == sign_det_A_j0:
                continue
            else:
                return False

        return True

    def deg_simplex(self, S, proj=None):
        """
        Test a simplex S for degeneracy (linear dependence in R^dim)
        :param S: Numpy array of simplex with rows as vertex vectors
        :param proj: array, optional, if the projection S[1:] - S[0] is already
                     computed it can be added as an optional argument.
        :return:
        """
        # Strategy: we test all combination of faces, if any of the determinants
        # are zero then the vectors lie on the same face and is therefore
        # linearly dependent in the space of R^dim
        if proj is None:
            proj = S[1:] - S[0]

        #TODO: Is checking the projection of one vertex against faces of other
        #       vertices sufficient? Or do we need to check more vertices in
        #       dimensions higher than 2?
        #TODO: Literature seems to suggest using proj.T, but why is this needed?
        if numpy.linalg.det(proj) == 0.0: #TODO: Repalace with tolerance?
            return True  # Simplex is degenerate
        else:
            return False  # Simplex is not degenerate

    def st(self, v_x):
        """
        Returns the star domain st(v) of a vertex with coordinates v_x.
        :param v: The vertex v in st(v)
        :return: st, a set containing all the vertices in st(v)
        """
        return self.V[v_x].star()

    # %% Discrete differential geometry
    def clifford(self, dim, q=''):
        """
        Memoize a specified clifford algebra so that it is only needed to
        initialise once, default is euclidean Cl(dim)
        :param dim:
        :param q:
        :return:
        """
        if not self.dec:
            logging.warning("Discrete exterior calculus functionality will be "
                            "unavailable, To use install the clifford package"
                            " with `pip install clifford`")
        try:
            return self.calgebras[str(dim) + q]
        except (AttributeError, KeyError):
            layout, blades = cf.Cl(dim)
            i = 0
            one_forms = []
            for n in blades:
                i += 1
                one_forms.append(1 * blades[n])
                if i == dim:
                    break

            self.calgebras[str(dim) + q] = layout, blades, one_forms
            return self.calgebras[str(dim) + q]

    def sharp(self, v_x):
        """
        Convert a vector to a 1-form

        TODO: Should be able to convert k-forms

        :param v_x:  vector, dimension may differ for class self.dim
        :return:
        """
        dim = len(v_x)
        # Call memoized Clifford algebra
        layout, blades, one_forms = self.clifford(dim)
        #NOTE: Using numpy.dot(v_x, one_forms) converts one_forms
        form = 0
        for i, of in enumerate(one_forms):
            form += v_x[i] * of
        return form

    def flat(self, form, dim):
        """
        Convert a 1-form to a vector

        Note that dim can by computed by evaluating len(v_x) after iterating
        form, but this is probably expensive.

        :param form:
        :return: v_x, numpy array
        """
        # Find 1 form grade projection
        oneform = form(1)  # Not reducing struction
        v_x = []
        for f in form:
            v_x.append(f)
        v_x = numpy.array(v_x)
        v_x = v_x[1:dim+1]
        return v_x

    # %% Plots
    def plot_complex(self, show=True, directed=True, complex_plot=True,
                     contour_plot=True, surface_plot=True,
                     surface_field_plot=True, minimiser_points=True,
                     point_color='do', line_color='do',
                     complex_color_f='lo', complex_color_e='do', pointsize=7,
                     no_grids=False, save_fig=True, strpath=None,
                     plot_path='fig/', fig_name='complex.pdf', arrow_width=None,
                     fig_surface=None, ax_surface=None, fig_complex=None,
                     ax_complex=None
                     ):
        """
        Plots the current simplicial complex contained in the class. It requires
        at least one vector in the self.V to have been defined.


        :param show: boolean, optional, show the output plots
        :param directed: boolean, optional, adds directed arrows to edges
        :param contour_plot: boolean, optional, contour plots of the field functions
        :param surface_plot: boolean, optional, a 3 simplicial complex + sfield plot
        :param surface_field_plot: boolean, optional, 3 dimensional surface + contour plot
        :param minimiser_points: boolean, optional, adds minimiser points
        :param point_color: str or vec, optional, colour of complex points
        :param line_color: str or vec, optional, colour of complex edges
        :param complex_color_f: str or vec, optional, colour of surface complex faces
        :param complex_color_e: str or vec, optional, colour of surface complex edges
        :param pointsize: float, optional, size of vectices on plots
        :param no_grids: boolean, optional, removes gridlines and axes
        :param save_fig: boolean, optional, save the output figure to file
        :param strpath: str, optional, string path of the file name
        :param plot_path: str, optional, relative path to file outputs
        :param fig_name: str, optional, name of the complex file to save
        :param arrow_width: float, optional, fixed size for arrows
        :return: self.ax_complex, a matplotlib Axes class containing the complex and field contour
        :return: self.ax_surface, a matplotlib Axes class containing the complex surface and field surface
        TODO: self.fig_* missing
        Examples
        --------
        # Initiate a complex class
        >>> import pylab
        >>> H = Complex(2, domain=[(0, 10)], sfield=func)

        # As an example we'll use the built in triangulation to generate vertices
        >>> H.triangulate()
        >>> H.split_generation()

        # Plot the complex
        >>> H.plot_complex()

        # You can add any sort of custom drawings to the Axes classes of the
        plots
        >>> H.ax_complex.plot(0.25, 0.25, '.', color='k', markersize=10)
        >>> H.ax_surface.scatter(0.25, 0.25, 0.25, '.', color='k', s=10)

        # Show the final plot
        >>> pylab.show()

        # Clear current plot instances
        >>> H.plot_clean()

        Example 2: Subplots  #TODO: Test
        >>> import matplotlib.pyplot as plt
        >>> fig, axes = plt.subplots(ncols=2)
        >>> H = Complex(2, domain=[(0, 10)], sfield=func)
        >>> H.triangulate()
        >>> H.split_generation()

        # Plot the complex on the same subplot
        >>> H.plot_complex(fig_surface=fig, ax_surface=axes[0],
        ...                fig_complex=fig, ax_complex=axes[1])

        # Note you can also plot several complex objects on larger subplots
        #  using this method.

        """
        if not matplotlib_available:
            logging.warning("Plotting functions are unavailable. To "
                            "install matplotlib install using ex. `pip install "
                            "matplotlib` ")
            return

        # Check if fix or ax arguments are passed
        if fig_complex is not None:
            self.fig_complex = fig_complex
        if ax_complex is not None:
            self.ax_complex = ax_complex
        if fig_surface is not None:
            self.fig_surface = fig_surface
        if ax_surface is not None:
            self.ax_surface = ax_surface

        # Create pyplot.figure instance if none exists yet
        try:
            self.fig_complex
        except AttributeError:
            self.fig_complex = pyplot.figure()

        # Consistency
        if self.sfield is None:
            if contour_plot:
                contour_plot = False
                logging.warning("Warning, no associated scalar field found. "
                                "Not plotting contour_plot.")

            if surface_field_plot:
                surface_field_plot = False
                logging.warning("Warning, no associated scalar field found. "
                                "Not plotting surface field.")

        # Define colours:
        coldict = {'lo': numpy.array([242, 189, 138]) / 255,  # light orange
                   'do': numpy.array([235, 129, 27]) / 255  # Dark alert orange
                   }

        def define_cols(col):
            if (col is 'lo') or (col is 'do'):
                col = coldict[col]
            elif col is None:
                col = None
            return col

        point_color = define_cols(point_color)  # None will generate
        line_color = define_cols(line_color)
        complex_color_f = define_cols(complex_color_f)
        complex_color_e = define_cols(complex_color_e)

        if self.dim == 1:
            if arrow_width is not None:
                self.arrow_width = arrow_width
            else:  # heuristic
                dx = self.bounds[0][1] - self.bounds[0][0]
                self.arrow_width = (dx * 0.13
                                    / (numpy.sqrt(len(self.V.cache))))
                self.mutation_scale = 58.83484054145521 * self.arrow_width * 1.3

            try:
                self.ax_complex
            except:
                self.ax_complex = self.fig_complex.add_subplot(1, 1, 1)

            min_points = []
            for v in self.V.cache:
                self.ax_complex.plot(v, 0, '.',
                                     color=point_color,
                                     markersize=pointsize)
                xlines = []
                ylines = []
                for v2 in self.V[v].nn:
                    xlines.append(v2.x)
                    ylines.append(0)

                    if directed:
                        if self.V[v].f > v2.f:  # direct V2 --> V1
                            x1_vec = list(self.V[v].x)
                            x2_vec = list(v2.x)
                            x1_vec.append(0)
                            x2_vec.append(0)
                            ap = self.plot_directed_edge(self.V[v].f, v2.f,
                                                         x1_vec, x2_vec,
                                                         mut_scale=0.5 * self.mutation_scale,
                                                         proj_dim=2,
                                                         color=line_color)

                            self.ax_complex.add_patch(ap)

                if minimiser_points:
                    if self.V[v].minimiser():
                        v_min = list(v)
                        v_min.append(0)
                        min_points.append(v_min)

                self.ax_complex.plot(xlines, ylines, color=line_color)

            if minimiser_points:
                self.ax_complex = self.plot_min_points(self.ax_complex,
                                                       min_points,
                                                       proj_dim=2,
                                                       point_color=point_color,
                                                       pointsize=pointsize)

            # Clean up figure
            if self.bounds is None:
                pyplot.ylim([-1e-2, 1 + 1e-2])
                pyplot.xlim([-1e-2, 1 + 1e-2])
            else:
                fac = 1e-2  # TODO: TEST THIS
                pyplot.ylim([0 - fac, 0 + fac])
                pyplot.xlim(
                    [self.bounds[0][0] - fac * (self.bounds[0][1]
                                                - self.bounds[0][0]),
                     self.bounds[0][1] + fac * (self.bounds[0][1]
                                                - self.bounds[0][0])])
            if no_grids:
                self.ax_complex.set_xticks([])
                self.ax_complex.set_yticks([])
                self.ax_complex.axis('off')

            # Surface plots
            if surface_plot or surface_field_plot:
                try:
                    self.fig_surface
                except AttributeError:
                    self.fig_surface = pyplot.figure()
                try:
                    self.ax_surface
                except:
                    self.ax_surface = self.fig_surface.add_subplot(1, 1, 1)

                # Add a plot of the field function.
                if surface_field_plot:
                    self.fig_surface, self.ax_surface = self.plot_field_surface(
                        self.fig_surface,
                        self.ax_surface,
                        self.bounds,
                        self.sfield,
                        self.sfield_args,
                        proj_dim=2,
                        color=complex_color_f)  # TODO: Custom field colour

                if surface_plot:
                    self.fig_surface, self.ax_surface = self.plot_complex_surface(
                        self.fig_surface,
                        self.ax_surface,
                        directed=directed,
                        pointsize=pointsize,
                        color_e=complex_color_e,
                        color_f=complex_color_f,
                        min_points=min_points)

                if no_grids:
                    self.ax_surface.set_xticks([])
                    self.ax_surface.set_yticks([])
                    self.ax_surface.axis('off')

        elif self.dim == 2:
            if arrow_width is not None:
                self.arrow_width = arrow_width
            else:  # heuristic
                dx1 = self.bounds[0][1] - self.bounds[0][0]
                dx2 = self.bounds[1][1] - self.bounds[1][0]

                self.arrow_width = (min(dx1, dx2) * 0.13
                                    / (numpy.sqrt(len(self.V.cache))))
                self.mutation_scale = 58.83484054145521 * self.arrow_width * 1.5

            try:
                self.ax_complex
            except:
                self.ax_complex = self.fig_complex.add_subplot(1, 1, 1)

            if contour_plot:
                self.plot_contour(self.bounds, self.sfield,
                                  self.sfield_args)

            if complex_plot:
                min_points = []
                for v in self.V.cache:
                    self.ax_complex.plot(v[0], v[1], '.', color=point_color,
                                         markersize=pointsize)

                    xlines = []
                    ylines = []
                    for v2 in self.V[v].nn:
                        xlines.append(v2.x[0])
                        ylines.append(v2.x[1])
                        xlines.append(v[0])
                        ylines.append(v[1])

                        if directed:
                            if self.V[v].f > v2.f:  # direct V2 --> V1
                                ap = self.plot_directed_edge(self.V[v].f, v2.f,
                                                             self.V[v].x, v2.x,
                                                             mut_scale=self.mutation_scale,
                                                             proj_dim=2,
                                                             color=line_color)

                                self.ax_complex.add_patch(ap)

                    if minimiser_points:
                        if self.V[v].minimiser():
                            min_points.append(v)

                    self.ax_complex.plot(xlines, ylines, color=line_color)

                if minimiser_points:
                    self.ax_complex = self.plot_min_points(self.ax_complex,
                                                           min_points,
                                                           proj_dim=2,
                                                           point_color=point_color,
                                                           pointsize=pointsize)
                else:
                    min_points = []

            # Clean up figure
            if self.bounds is None:
                pyplot.ylim([-1e-2, 1 + 1e-2])
                pyplot.xlim([-1e-2, 1 + 1e-2])
            else:
                fac = 1e-2  # TODO: TEST THIS
                pyplot.ylim(
                    [self.bounds[1][0] - fac * (self.bounds[1][1]
                                                - self.bounds[1][0]),
                     self.bounds[1][1] + fac * (self.bounds[1][1]
                                                - self.bounds[1][0])])
                pyplot.xlim(
                    [self.bounds[0][0] - fac * (self.bounds[1][1]
                                                - self.bounds[1][0]),
                     self.bounds[0][1] + fac * (self.bounds[1][1]
                                                - self.bounds[1][0])])

            if no_grids:
                self.ax_complex.set_xticks([])
                self.ax_complex.set_yticks([])
                self.ax_complex.axis('off')

            # Surface plots
            if surface_plot or surface_field_plot:
                try:
                    self.fig_surface
                except AttributeError:
                    self.fig_surface = pyplot.figure()
                try:
                    self.ax_surface
                except:
                    self.ax_surface = self.fig_surface.gca(projection='3d')

                # Add a plot of the field function.
                if surface_field_plot:
                    self.fig_surface, self.ax_surface = self.plot_field_surface(
                        self.fig_surface,
                        self.ax_surface,
                        self.bounds,
                        self.sfield,
                        self.sfield_args,
                        proj_dim=3)

                if surface_plot:
                    self.fig_surface, self.ax_surface = self.plot_complex_surface(
                        self.fig_surface,
                        self.ax_surface,
                        directed=directed,
                        pointsize=pointsize,
                        color_e=complex_color_e,
                        color_f=complex_color_f,
                        min_points=min_points)

                if no_grids:
                    self.ax_surface.set_xticks([])
                    self.ax_surface.set_yticks([])
                    self.ax_surface.axis('off')


        elif self.dim == 3:
            try:
                self.ax_complex
            except:
                self.ax_complex = Axes3D(self.fig_complex)

            min_points = []
            for v in self.V.cache:
                self.ax_complex.scatter(v[0], v[1], v[2],
                                        color=point_color, s=pointsize)
                x = []
                y = []
                z = []
                x.append(self.V[v].x[0])
                y.append(self.V[v].x[1])
                z.append(self.V[v].x[2])
                for v2 in self.V[v].nn:
                    x.append(v2.x[0])
                    y.append(v2.x[1])
                    z.append(v2.x[2])
                    x.append(self.V[v].x[0])
                    y.append(self.V[v].x[1])
                    z.append(self.V[v].x[2])
                    if directed:
                        if self.V[v].f > v2.f:  # direct V2 --> V1
                            ap = self.plot_directed_edge(self.V[v].f, v2.f,
                                                         self.V[v].x, v2.x,
                                                         proj_dim=3,
                                                         color=line_color)
                            self.ax_complex.add_artist(ap)

                self.ax_complex.plot(x, y, z,
                                     color=line_color)

                if minimiser_points:
                    if self.V[v].minimiser():
                        min_points.append(v)

            if minimiser_points:
                self.ax_complex = self.plot_min_points(self.ax_complex,
                                                       min_points,
                                                       proj_dim=3,
                                                       point_color=point_color,
                                                       pointsize=pointsize)

            self.fig_surface = None  # Current default
            self.ax_surface = None  # Current default

        else:
            logging.warning("dimension higher than 3 or wrong complex format")
            self.fig_complex = None
            self.ax_complex = None
            self.fig_surface = None
            self.ax_surface = None

        # Save figure to file
        if save_fig:
            if strpath is None:
                script_dir = os.getcwd()  # os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, plot_path)
                sample_file_name = fig_name

                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                strpath = results_dir + sample_file_name

            self.plot_save_figure(strpath)


        if show and (not self.dim > 3):
            self.fig_complex.show()
        try:
            self.fig_surface
            self.ax_surface
            if show:
                self.fig_surface.show()
        except AttributeError:
            self.fig_surface = None  # Set to None for return reference
            self.ax_surface = None

        return self.fig_complex, self.ax_complex, self.fig_surface, self.ax_surface

    def plot_save_figure(self, strpath):

        self.fig_complex.savefig(strpath, transparent=True,
                                 bbox_inches='tight', pad_inches=0)

    def plot_clean(self, del_ax=True, del_fig=True):
        try:
            if del_ax:
                del (self.ax_complex)
            if del_fig:
                del (self.fig_complex)
        except AttributeError:
            pass

    def plot_contour(self, bounds, func, func_args=()):
        """
        Plots the field functions. Mostly for developmental purposes
        :param fig:
        :param bounds:
        :param func:
        :param func_args:
        :param surface:
        :param contour:
        :return:
        """
        xg, yg, Z = self.plot_field_grids(bounds, func, func_args)
        cs = pyplot.contour(xg, yg, Z, cmap='binary_r', color='k')
        pyplot.clabel(cs)

    def plot_complex_surface(self, fig, ax, directed=True, pointsize=5,
                             color_e=None, color_f=None, min_points=[]):
        """
        fig and ax need to be supplied outside the method
        :param fig: ex. ```fig = pyplot.figure()```
        :param ax: ex.  ```ax = fig.gca(projection='3d')```
        :param bounds:
        :param func:
        :param func_args:
        :return:
        """
        if self.dim == 1:
            # Plot edges
            z = []
            for v in self.V.cache:
                ax.plot(v, self.V[v].f, '.', color=color_e,
                        markersize=pointsize)
                z.append(self.V[v].f)
                for v2 in self.V[v].nn:
                    ax.plot([v, v2.x],
                            [self.V[v].f, v2.f],
                            color=color_e)

                    if directed:
                        if self.V[v].f > v2.f:  # direct V2 --> V1
                            x1_vec = [float(self.V[v].x[0]), self.V[v].f]
                            x2_vec = [float(v2.x[0]), v2.f]

                            a = self.plot_directed_edge(self.V[v].f, v2.f,
                                                        x1_vec, x2_vec,
                                                        proj_dim=2,
                                                        color=color_e)
                            ax.add_artist(a)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$f$')

            if len(min_points) > 0:
                iter_min = min_points.copy()
                for ind, v in enumerate(iter_min):
                    min_points[ind][1] = float(self.V[v[0]].f)

                ax = self.plot_min_points(ax,
                                          min_points,
                                          proj_dim=2,
                                          point_color=color_e,
                                          pointsize=pointsize
                                          )

        elif self.dim == 2:
            # Plot edges
            z = []
            for v in self.V.cache:
                z.append(self.V[v].f)
                for v2 in self.V[v].nn:
                    ax.plot([v[0], v2.x[0]],
                            [v[1], v2.x[1]],
                            [self.V[v].f, v2.f],
                            color=color_e)

                    if directed:
                        if self.V[v].f > v2.f:  # direct V2 --> V1
                            x1_vec = list(self.V[v].x)
                            x2_vec = list(v2.x)
                            x1_vec.append(self.V[v].f)
                            x2_vec.append(v2.f)
                            a = self.plot_directed_edge(self.V[v].f, v2.f,
                                                        x1_vec, x2_vec,
                                                        proj_dim=3,
                                                        color=color_e)

                            ax.add_artist(a)

            # TODO: For some reason adding the scatterplots for minimiser spheres
            #      makes the directed edges disappear behind the field surface
            if len(min_points) > 0:
                iter_min = min_points.copy()
                for ind, v in enumerate(iter_min):
                    min_points[ind] = list(min_points[ind])
                    min_points[ind].append(self.V[v].f)

                ax = self.plot_min_points(ax,
                                          min_points,
                                          proj_dim=3,
                                          point_color=color_e,
                                          pointsize=pointsize
                                          )

            # Triangulation to plot faces
            # Compute a triangulation #NOTE: can eat memory
            self.vertex_face_mesh()

            ax.plot_trisurf(numpy.array(self.vertices_fm)[:, 0],
                            numpy.array(self.vertices_fm)[:, 1],
                            z,
                            triangles=numpy.array(self.simplices_fm_i),
                            # TODO: Select colour scheme
                            color=color_f,
                            alpha=0.4,
                            linewidth=0.2,
                            antialiased=True)

            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_zlabel('$f$')

        return fig, ax

    def plot_field_surface(self, fig, ax, bounds, func, func_args=(),
                           proj_dim=2, color=None):
        """
        fig and ax need to be supplied outside the method
        :param fig: ex. ```fig = pyplot.figure()```
        :param ax: ex.  ```ax = fig.gca(projection='3d')```
        :param bounds:
        :param func:
        :param func_args:
        :return:
        """
        if proj_dim == 2:
            from matplotlib import cm
            xr = numpy.linspace(self.bounds[0][0], self.bounds[0][1], num=1000)
            fr = numpy.zeros_like(xr)
            for i in range(xr.shape[0]):
                fr[i] = func(xr[i], *func_args)

            ax.plot(xr, fr, alpha=0.6, color=color)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$f$')

        if proj_dim == 3:
            from matplotlib import cm
            xg, yg, Z = self.plot_field_grids(bounds, func, func_args)
            ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                            # cmap=cm.coolwarm,
                            # cmap=cm.magma,
                        #    cmap=cm.plasma,  #TODO: Restore
                            # cmap=cm.inferno,
                            # cmap=cm.pink,
                            # cmap=cm.viridis,
                            #facecolors="do it differently, ok?",
                            color = [0.94901961, 0.74117647, 0.54117647],
                            linewidth=0,
                            antialiased=True, alpha=0.8, shade=True)

            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_zlabel('$f$')
        return fig, ax

    def plot_field_grids(self, bounds, func, func_args):
        try:
            return self.plot_xg, self.plot_yg, self.plot_Z
        except AttributeError:
            X = numpy.linspace(bounds[0][0], bounds[0][1])
            Y = numpy.linspace(bounds[1][0], bounds[1][1])
            xg, yg = numpy.meshgrid(X, Y)
            Z = numpy.zeros((xg.shape[0],
                             yg.shape[0]))

            for i in range(xg.shape[0]):
                for j in range(yg.shape[0]):
                    Z[i, j] = func([xg[i, j], yg[i, j]], *func_args)

            self.plot_xg, self.plot_yg, self.plot_Z = xg, yg, Z
            return self.plot_xg, self.plot_yg, self.plot_Z

    def plot_directed_edge(self, f_v1, f_v2, x_v1, x_v2, mut_scale=20,
                           proj_dim=2,
                           color=None):
        """
        Draw a directed edge embeded in 2 or 3 dimensional space between two
        vertices v1 and v2.

        :param f_v1: field value at f(v1)
        :param f_v2: field value at f(v2)
        :param x_v1: coordinate vector 1
        :param x_v2: coordinate vector 2
        :param proj_dim: int, must be either 2 or 3
        :param color: edge color
        :return: a, artist arrow object (add with ex. Axes.add_artist(a)
        """
        if proj_dim == 2:
            if f_v1 > f_v2:  # direct V2 --> V1
                dV = numpy.array(x_v1) - numpy.array(x_v2)
                ap = matplotlib.patches.FancyArrowPatch(
                    numpy.array(x_v2) + 0.5 * dV,  # tail
                    numpy.array(x_v2) + 0.6 * dV,  # head
                    mutation_scale=mut_scale,
                    arrowstyle='-|>',
                    fc=color, ec=color,
                    color=color,
                )

        if proj_dim == 3:
            if f_v1 > f_v2:  # direct V2 --> V1
                dV = numpy.array(x_v1) - numpy.array(x_v2)
                # TODO: Might not be correct (unvalidated)
                ap = Arrow3D([x_v2[0], x_v2[0] + 0.5 * dV[0]],
                             [x_v2[1], x_v2[1] + 0.5 * dV[1]],
                             [x_v2[2], x_v2[2] + 0.5 * dV[2]],
                             mutation_scale=20,
                             lw=1, arrowstyle="-|>",
                             color=color)

        return ap

    def plot_min_points(self, axes, min_points, proj_dim=2, point_color=None,
                        pointsize=5):
        """
        Add a given list of highlighted minimiser points to axes

        :param ax: An initiated matplotlib Axes class
        :param min_points: list of minimsier points
        :param proj_dim: projection dimension, must be either 2 or 3
        :param point_color: optional
        :param point_size: optional
        :return:
        """
        if proj_dim == 2:
            for v in min_points:
                if point_color is 'r':
                    min_col = 'k'
                else:
                    min_col = 'r'

                axes.plot(v[0], v[1], '.', color=point_color,
                          markersize=2.5 * pointsize)

                axes.plot(v[0], v[1], '.', color='k',
                          markersize=1.5 * pointsize)

                axes.plot(v[0], v[1], '.', color=min_col,
                          markersize=1.4 * pointsize)

        if proj_dim == 3:
            for v in min_points:
                if point_color is 'r':
                    min_col = 'k'
                else:
                    min_col = 'r'

                axes.scatter(v[0], v[1], v[2], color=point_color,
                             s=2.5 * pointsize)

                axes.scatter(v[0], v[1], v[2], color='k',
                             s=1.5 * pointsize)

                axes.scatter(v[0], v[1], v[2], color=min_col,
                             s=1.4 * pointsize)

        return axes

    # %% Conversions
    def incidence_array(self):
        """
        Construct v-v incidence array
        :return: self.incidence_structure
        """
        if self.centroid_added:
            self.incidence_structure = numpy.zeros([2 ** self.dim + 1,
                                                    2 ** self.dim + 1],
                                         dtype=int)
        else:
            self.incidence_structure = numpy.zeros([2 ** self.dim,
                                                    2 ** self.dim],
                                         dtype=int)

        for v in self.V.cache:
            for v2 in self.V[v].nn:
                self.incidence_structure[self.V[v].index, v2.index] = 1

        return self.incidence_structure

    def v_array(self, cache=None, vset=None):
        """
        Build a numpy array from a cache of vertices or a set of vertex objects
        :param cache: A cache of vertices (tuples), must be iterable
        :param cache: A set of vertices (vertex objects), must be iterable
        :return: VA, numpy array consisting of vertices for every row

        example
        -------
        >>> H.VA = H.v_array(H.V.cache)
        """
        vl = []
        if cache is not None:
            for v in cache:
                vl.append(v)
        if vset is not None:
            for v in vset:
                vl.append(v.x)

        return numpy.array(vl)

    def vertex_face_mesh(self, field_conversions=True):
        """
        Convert the current simplicial complex from the default
        vertex-vertex mesh (low memory) to a

        NM

        :param field_conversions, boolean, optional
                If True then any associated field properties will be added to
                ordered lists ex, self.sfield_vf and self.vfield_vf

        :return: self.vertices_fm, A list of vertex vectors (corresponding to
                                   the ordered dict in self.V.cache)
                 self.simplices_fm, A list of (dim + 1)-lists containing vertex
                                    objects in a simplex.

                 self.simplices_fm_i, Same as self.simplices_fm except contains
                                      the indices corresponding to the list in
                                      self.vertices_fm

                 self.sfield_fm, Scalar field values corresponding to the
                                 vertices in self.vertices_fm
        """
        #TODO: UNTESTED FOR DIMENSIONS HIGHER THAN 2
        self.vertices_fm = []  # Vertices (A list of ob
        self.simplices_fm = []  # Faces
        self.simplices_fm_i = []

        # TODO: Add in field

        for v in self.V.cache:  # Note that cache is an OrderedDict
            self.vertices_fm.append(v)

            #TODO: Should this not be outside the initial loop?
            simplex = (self.dim + 1) * [None]  # Predetermined simplex sizes
            simplex[0] = self.V[v]
            build_simpl = simplex.copy()

            # indexed simplices
            simplex_i = (self.dim + 1) * [None]
            simplex_i[0] = self.V[v].index

            # TODO: We need to recursively call a function that checks each nn
            #  and checks that the nn is in all parent nns (otherwise there is
            #  a deviding line in the simplex)
            # NOTE: The recursion depth is (self.dim + 1)

            # Start looping through the vertices in the star domain
            for v2 in self.V[v].nn:
                # For every v2 we loop through its neighbours in v2.nn, for
                # every v2.nn that is also in self.V[v].nn we want to build
                # simplices connecting to the current v. Note that we want to
                # try and build simplices from v, v2 and any connected
                # neighbours.
                # Since all simplices are ordered vertices, we can check that
                # all simplices are unique.
                # The reason we avoid recursion in higher dimensions is because
                # we only want to find the connecting chains v1--v2--v3--v1
                # to add to simplex stacks. Once a simplex is full we try to
                # find another chain v1--v2--vi--v1 to add to the simplex until
                # it is full, here vi is any neighbour of v2
                simplex_i[1] = v2.index
                build_simpl_i = simplex_i.copy()
                ind = 1

                if self.dim > 1:
                    for v3 in v2.nn:

                        # if v3 has a connection to v1, not in current simplex
                        # and not v1 itself:
                        if ((v3 in self.V[v].nn) and (v3 not in build_simpl_i)
                                and (v3 is not self.V[v])):
                            try:  # Fill simplex with v's neighbours until it is full
                                ind += 1
                                # (if v2.index not in build_simpl_i) and v2.index in v2.nn
                                build_simpl_i[ind] = v3.index

                            except IndexError:  # When the simplex is full
                                # ind = 1 #TODO: Check
                                # Append full simplex and create a new one
                                s_b_s_i = sorted(
                                    build_simpl_i)  # Sorted simplex indices
                                if s_b_s_i not in self.simplices_fm_i:
                                    self.simplices_fm_i.append(s_b_s_i)
                                    # TODO: Build simplices_fm
                                    # self.simplices_fm.append(s_b_s_i)

                                build_simpl_i = simplex_i.copy()
                                # Start the new simplex with current neighbour as second
                                #  entry
                                if ((v3 in self.V[v].nn) and (
                                        v3 not in build_simpl_i)
                                        and (v3 is not self.V[v])):
                                    build_simpl_i[2] = v3.index
                                    ind = 2

                                if self.dim == 2:  # Special case, for dim > 2
                                    # it will not be full
                                    if s_b_s_i not in self.simplices_fm_i:
                                        self.simplices_fm_i.append(s_b_s_i)

                    # After loop check if we have a filled simplex
                    if len(build_simpl_i) == self.dim + 1:
                        if not (None in build_simpl_i):
                            s_b_s_i = sorted(
                                build_simpl_i)  # Sorted simplex indices
                            if s_b_s_i not in self.simplices_fm_i:
                                self.simplices_fm_i.append(s_b_s_i)

                    # NOTE: If we run out of v3 in v2.nn before a simplex is
                    # completed then there were not enough vertices to form a
                    # simplex with.

        # TODO: BUILD self.vertices_fm from  self.simplices_fm_i and
        # self.vertices_fm
        for s in self.simplices_fm_i:
            sl = []
            for i in s:
                sl.append(self.vertices_fm[i])

            # print(sl)

            # Append the newly built simple

        return

    #TODO: face_vertex_mesh

    # %% Data persistence
    def save_complex(self, fn):
        """
        TODO: Save the complex to file using pickle
        https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
        :param fn: str, filename
        :return:
        """

    def load_complex(self, fn):
        """
        TODO: Load the complex from file using pickle
        :param fn: str, filename
        :return:
        """
