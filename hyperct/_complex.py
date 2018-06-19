"""
Base classes for low memory simplicial complex structures

Restructuring overview.

We'll use factory method pattern
(https://en.wikipedia.org/wiki/Factory_method_pattern)
to inherit the properties of different complexes depending on whether we'll use
Arrays or pure tuples
Scalar, vector or no fields (Also for example make connect an abstract method to
                             compute field gradients etc)
Simplices, cells
etc.

FUTURE: Triangulate arbitrary domains other than n-cubes
(ex. using delaunay and low disc. sampling subject to constraints, or by adding
     n-cubes and other geometries)

     Starting point:
     An algorithm for automatic Delaunay triangulation of arbitrary planar domains
     https://www.sciencedirect.com/science/article/pii/096599789600004X
"""
import copy
import os
from abc import ABC, abstractmethod

import numpy
try:
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import axes3d, Axes3D
except ImportError:
    pass  # Warning plotting functions will be unavailable
from hyperct._vertex import (VertexCacheIndex, VertexCacheField)
try:
    from functools import lru_cache  # For Python 3 only
except ImportError:  # Python 2:
    import time
    import functools
    import collections

    # Note to avoid using external packages such as functools32 we use this code
    # only using the standard library
    def lru_cache(maxsize=255, timeout=None):
        """
        Thanks to ilialuk @ https://stackoverflow.com/users/2121105/ilialuk for
        this code snippet. Modifications by S. Endres
        """

        class LruCacheClass(object):
            def __init__(self, input_func, max_size, timeout):
                self._input_func = input_func
                self._max_size = max_size
                self._timeout = timeout

                # This will store the cache for this function,
                # format - {caller1 : [OrderedDict1, last_refresh_time1],
                #  caller2 : [OrderedDict2, last_refresh_time2]}.
                #   In case of an instance method - the caller is the instance,
                # in case called from a regular function - the caller is None.
                self._caches_dict = {}

            def cache_clear(self, caller=None):
                # Remove the cache for the caller, only if exists:
                if caller in self._caches_dict:
                    del self._caches_dict[caller]
                    self._caches_dict[caller] = [collections.OrderedDict(),
                                                 time.time()]

            def __get__(self, obj, objtype):
                """ Called for instance methods """
                return_func = functools.partial(self._cache_wrapper, obj)
                return_func.cache_clear = functools.partial(self.cache_clear,
                                                            obj)
                # Return the wrapped function and wraps it to maintain the
                # docstring and the name of the original function:
                return functools.wraps(self._input_func)(return_func)

            def __call__(self, *args, **kwargs):
                """ Called for regular functions """
                return self._cache_wrapper(None, *args, **kwargs)

            # Set the cache_clear function in the __call__ operator:
            __call__.cache_clear = cache_clear

            def _cache_wrapper(self, caller, *args, **kwargs):
                # Create a unique key including the types (in order to
                # differentiate between 1 and '1'):
                kwargs_key = "".join(map(
                    lambda x: str(x) + str(type(kwargs[x])) + str(kwargs[x]),
                    sorted(kwargs)))
                key = "".join(
                    map(lambda x: str(type(x)) + str(x), args)) + kwargs_key

                # Check if caller exists, if not create one:
                if caller not in self._caches_dict:
                    self._caches_dict[caller] = [collections.OrderedDict(),
                                                 time.time()]
                else:
                    # Validate in case the refresh time has passed:
                    if self._timeout is not None:
                        if (time.time() - self._caches_dict[caller][1]
                                > self._timeout):
                            self.cache_clear(caller)

                # Check if the key exists, if so - return it:
                cur_caller_cache_dict = self._caches_dict[caller][0]
                if key in cur_caller_cache_dict:
                    return cur_caller_cache_dict[key]

                # Validate we didn't exceed the max_size:
                if len(cur_caller_cache_dict) >= self._max_size:
                    # Delete the first item in the dict:
                    try:
                        cur_caller_cache_dict.popitem(False)
                    except KeyError:
                        pass
                # Call the function and store the data in the cache (call it
                # with the caller in case it's an instance function)
                if caller is not None:
                    args = (caller,) + args
                cur_caller_cache_dict[key] = self._input_func(*args, **kwargs)

                return cur_caller_cache_dict[key]

        # Return the decorator wrapping the class (also wraps the instance to
        # maintain the docstring and the name of the original function):
        return (lambda input_func: functools.wraps(input_func)(
            LruCacheClass(input_func, maxsize, timeout)))


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
            self.bounds = [(0, 1),]*dim
        else:
            self.bounds = domain
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
        #self.V = VertexCache(field, func_args, bounds, g_cons, g_args)
        if (sfield is not None) or (g_cons is not None):
           self.V = VertexCacheField(field=sfield, field_args=sfield_args,
                                     g_cons=g_cons, g_cons_args=g_cons_args)
        else:
           self.V = VertexCacheIndex()

        if vfield is not None:
            raise Warning("Vector field applications have not been implemented"
                          "yet")

    def __call__(self):
        return self.H

    # Triangulation methods
    def triangulate(self, domain=None):
        """
        Triangulate a domain in [x_l, x_u]^dim \in R^dim specified by bounds and
        constraints.

        If domain is None the default domain is the hyperrectangle [0, 1]^dim

        FUTURE: Currently only hyperrectangle domains are possible. In the
                future we'd like to define more complex domains.
        """
        # Generate n-cube here:
        self.H.append([])
        self.n_cube(symmetry=self.symmetry)
        #

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
            #del(self.V)
            self.V = VertexCacheField(field=self.sfield, field_args=self.sfield_args,
                                      g_cons=self.g_cons, g_cons_args=self.g_cons_args)
            #TODO: Find a way not to delete the entire vertex cache in situations
            # where this method is used to triangulate the domain together with
            # other in place connections. ex simply move n_cube to if statement
            # and use a temporary cache

            # Construct the initial spatial vector
            origin = []  # origin of complex domain vector
            supremum = []  # supremum of complex domain vector
            for i, (lb, ub) in enumerate(self.domain):
                origin.append(lb)
                supremum.append(ub)
                #x_a[i] = x_a[i] * (ub - lb) + lb
            #del(self.C0)
            self.origin = tuple(origin)
            self.supremum = tuple(supremum)
            #self.C0 =
            self.construct_hypercube(self.origin, self.supremum, 0, 0)

            #TODO: Find new C0 by looping through C_0 and checking if v in Cnew
            #      Then delete unused C0 and set Cnew to C_0

            #x_a = numpy.array(x, dtype=float)
            #if self.domain is not None:
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
        #TODO: Check for loaded data and load if available
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

    # Construct incidence array:
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

    # A more sparse incidence generator:
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
        #print(f'self.C0 = {self.C0()}')
        #print(f'self.C0 = {self.C0()[self.graph[0]]}')
        #[self.C0()[index] for index in self.graph[i]]

        self.v_o = numpy.array(origin)
        self.v_s = numpy.array(supremum)
        for i, v in enumerate(self.C0()[:-1]):  # Build new vertices
            #print(f'v.x = {v.x}')
            t1 = self.generate_sub_cell_t1(origin, v.x)
            #print(t1)
            t2 = self.generate_sub_cell_t2(supremum, v.x)
            #print(t2)
            vec = t1 + t2
            #print(f'vec = {vec}')

            vec = tuple(vec)
            #nn_v = [self.C0()[index] for index in self.graph[i]]
            #C_new.add_vertex(self.V.__getitem__(vec, nn=nn_v))
            C_new.add_vertex(self.V[vec])
            #print(f'self.V[vec].x = {self.V[vec].x}')
            #print(f'C_new() = {C_new()}')

        # Add new centroid
        C_new.add_vertex(self.V[C_new.centroid])

        #print(C_new())
        #print(self.C0())

        for i, v in enumerate(C_new()):  # Connect new vertices
            nn_v = [C_new()[index] for index in self.graph[i]]
            self.V[v.x].nn.update(nn_v)


        #nn_v = [C_new()[index] for index in self.graph[-1]]
        #C_new.add_vertex(self.V.__getitem__(C_new.centroid, nn_v))

        #C_new.add_vertex(self.V.__getitem__(vec, nn=nn_v))
        # Add new centroid
        #C_new.add_vertex(self.V[C_new.centroid])

        #V_new.append(C_new.centroid)

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

    # Plots
    def plot_complex(self):
        """
             Here C is the LIST of simplexes S in the
             2 or 3 dimensional complex

             To plot a single simplex S in a set C, use ex. [C[0]]
        """
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        if self.dim == 2:
            pyplot.figure()
            for C in self.H:
                for c in C:
                    for v in c():
                        if self.bounds is None:
                            x_a = numpy.array(v.x, dtype=float)
                        else:
                            x_a = numpy.array(v.x, dtype=float)
                            for i in range(len(self.bounds)):
                                x_a[i] = (x_a[i] * (self.bounds[i][1]
                                                    - self.bounds[i][0])
                                          + self.bounds[i][0])

                        # logging.info('v.x_a = {}'.format(x_a))

                        pyplot.plot([x_a[0]], [x_a[1]], 'o')

                        xlines = []
                        ylines = []
                        for vn in v.nn:
                            if self.bounds is None:
                                xn_a = numpy.array(vn.x, dtype=float)
                            else:
                                xn_a = numpy.array(vn.x, dtype=float)
                                for i in range(len(self.bounds)):
                                    xn_a[i] = (xn_a[i] * (self.bounds[i][1]
                                                          - self.bounds[i][0])
                                               + self.bounds[i][0])

                            # logging.info('vn.x = {}'.format(vn.x))

                            xlines.append(xn_a[0])
                            ylines.append(xn_a[1])
                            xlines.append(x_a[0])
                            ylines.append(x_a[1])

                        pyplot.plot(xlines, ylines)

            if self.bounds is None:
                pyplot.ylim([-1e-2, 1 + 1e-2])
                pyplot.xlim([-1e-2, 1 + 1e-2])
            else:
                pyplot.ylim(
                    [self.bounds[1][0] - 1e-2, self.bounds[1][1] + 1e-2])
                pyplot.xlim(
                    [self.bounds[0][0] - 1e-2, self.bounds[0][1] + 1e-2])

            pyplot.show()

        elif self.dim == 3:
            fig = pyplot.figure()
            ax = Axes3D(fig)
            #ax = fig.add_subplot(111, projection='3d')

            for C in self.H:
                for c in C:
                    for v in c():
                        x = []
                        y = []
                        z = []
                        # logging.info('v.x = {}'.format(v.x))
                        x.append(v.x[0])
                        y.append(v.x[1])
                        z.append(v.x[2])
                        for vn in v.nn:
                            x.append(vn.x[0])
                            y.append(vn.x[1])
                            z.append(vn.x[2])
                            x.append(v.x[0])
                            y.append(v.x[1])
                            z.append(v.x[2])
                            # logging.info('vn.x = {}'.format(vn.x))

                        ax.plot(x, y, z, label='simplex')

            pyplot.show()
        else:
            print("dimension higher than 3 or wrong complex format")
        return

    def plot_complex_new(self):
        """
             Here C is the LIST of simplexes S in the
             2 or 3 dimensional complex

             To plot a single simplex S in a set C, use ex. [C[0]]
        """

        # Create pyplot.figure instance if none exists yet
        try:
            self.fig_complex
        except AttributeError:
            self.fig_complex = pyplot.figure()

        # Define colours:
        lo = numpy.array([242, 189, 138]) / 255  # light orange
        do = numpy.array([235, 129, 27]) / 255  # Dark alert orange

        directed = True
        contour_plot = True
        minimiser_points = True
        # TODO: Add dict for visual parameters
        point_color = do  # None will generate
        line_color = do
        pointsize = 5
        no_grids = False
        save_fig = True
        strpath = None  # Full string path of the file name
        plot_path = 'fig/'   # Name of the relative directory to save
        fig_name = 'complex.pdf'  # Name of the complex file to save
        arrow_width = None

        if arrow_width is not None:
            self.arrow_width = arrow_width
        else:  # hearistic #TODO: See how well rectangle stretching works
            dx1 = self.bounds[0][1] - self.bounds[0][0]
            dx2 = self.bounds[1][1] - self.bounds[1][0]
            numpy.linalg.norm([dx1, dx2])
            #TODO: Streched recs will look strange
            self.arrow_width = (numpy.linalg.norm([dx1, dx2]) * 0.13
                                #* 0.1600781059358212
                                / (numpy.sqrt(len(self.V.cache))))
            print(self.arrow_width)

        lw = 1  # linewidth

        if self.dim == 2:
            try:
                self.ax_complex
            except:
                self.ax_complex = self.fig_complex.add_subplot(1, 1, 1)

            if contour_plot:
                self.plot_contour(self.fig_complex, self.bounds, self.sfield,
                                  self.sfield_args)

            min_points = []
            for v in self.V.cache:
                self.ax_complex.plot(v[0], v[1], '.', color=point_color,
                                     markersize=pointsize)

                #complex_ax = pyplot.axes()

                xlines = []
                ylines = []
                for v2 in self.V[v].nn:
                    xlines.append(v2.x[0])
                    ylines.append(v2.x[1])
                    xlines.append(v[0])
                    ylines.append(v[1])

                    if directed:
                        if self.V[v].f > v2.f:  # direct V2 --> V1
                            dV = numpy.array(self.V[v].x) - numpy.array(v2.x)
                            self.ax_complex.arrow(v2.x[0],
                                                  v2.x[1],
                                                  0.5 * dV[0], 0.5 * dV[1],
                                                  head_width=self.arrow_width,
                                                  head_length=self.arrow_width,
                                                  fc=line_color, ec=line_color,
                                                  color=line_color)

                if minimiser_points:
                    if self.V[v].minimiser():
                        min_points.append(v)

                self.ax_complex.plot(xlines, ylines, color=line_color)

            if minimiser_points:
                for v in min_points:
                    if point_color is 'r':
                        min_col = 'k'
                    else:
                        min_col = 'r'

                    self.ax_complex.plot(v[0], v[1], '.', color=point_color,
                                         markersize=2.5*pointsize)

                    self.ax_complex.plot(v[0], v[1], '.', color='k',
                                         markersize=1.5*pointsize)

                    self.ax_complex.plot(v[0], v[1], '.', color=min_col,
                                         markersize=1.4*pointsize)

            # Clean up figure

            if self.bounds is None:
                pyplot.ylim([-1e-2, 1 + 1e-2])
                pyplot.xlim([-1e-2, 1 + 1e-2])
            else:
                # - fac * (self.bounds[1][1] - self.bounds[1][0])
                fac = 1e-2  #TODO: TEST THIS
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

        elif self.dim == 3:
            fig = pyplot.figure()
            ax = Axes3D(fig)
            #ax = fig.add_subplot(111, projection='3d')

            for C in self.H:
                for c in C:
                    for v in c():
                        x = []
                        y = []
                        z = []
                        # logging.info('v.x = {}'.format(v.x))
                        x.append(v.x[0])
                        y.append(v.x[1])
                        z.append(v.x[2])
                        for vn in v.nn:
                            x.append(vn.x[0])
                            y.append(vn.x[1])
                            z.append(vn.x[2])
                            x.append(v.x[0])
                            y.append(v.x[1])
                            z.append(v.x[2])
                            # logging.info('vn.x = {}'.format(vn.x))

                        ax.plot(x, y, z, label='simplex')

            pyplot.show()
        else:
            print("dimension higher than 3 or wrong complex format")

        # Save figure to file
        if save_fig:
            if strpath is None:
                script_dir = os.getcwd() #os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, plot_path)
                sample_file_name = fig_name

                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                strpath = results_dir + sample_file_name

            self.plot_save_figure(strpath)

        self.fig_complex.show()
        return self.ax_complex

    def plot_save_figure(self, strpath):

        self.fig_complex.savefig(strpath, transparent=True,
                           bbox_inches='tight', pad_inches=0)

    def plot_clean(self, del_ax=True, del_fig=True):
        try:
            if del_ax:
                del(self.ax_complex)
            if del_fig:
                del(self.fig_complex)
        except AttributeError:
            pass

    def plot_contour(self, fig, bounds, func, func_args,
                     surface=True, contour=True):
        #from mpl_toolkits.mplot3d import axes3d
        #import matplotlib.pyplot as plt
        from matplotlib import cm

        # X = points[:, 0]
        X = numpy.linspace(bounds[0][0], bounds[0][1])
        # Y = points[:, 1]
        Y = numpy.linspace(bounds[1][0], bounds[1][1])
        xg, yg = numpy.meshgrid(X, Y)
        Z = numpy.zeros((xg.shape[0],
                         yg.shape[0]))

        for i in range(xg.shape[0]):
            for j in range(yg.shape[0]):
                Z[i, j] = func([xg[i, j], yg[i, j]])

        if 0:#surface:
            # fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                            cmap=cm.coolwarm, linewidth=0,
                            antialiased=True, alpha=1.0, shade=True)

            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_zlabel('$f$')

        if contour:
            #plt.figure()
            cs = pyplot.contour(xg, yg, Z, cmap='binary_r', color='k')
            pyplot.clabel(cs)

    # Data persistence
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

class VertexGroup(object):
    def __init__(self, p_gen, p_hgr):
        self.p_gen = p_gen  # parent generation
        self.p_hgr = p_hgr  # parent homology group rank
        self.hg_n = None
        self.hg_d = None

        # Maybe add parent homology group rank total history
        # This is the sum off all previously split cells
        # cumulatively throughout its entire history
        self.C = []

    def __call__(self):
        return self.C

    def add_vertex(self, V):
        if V not in self.C:
            self.C.append(V)

    def homology_group_rank(self):
        """
        Returns the homology group order of the current cell
        """
        if self.hg_n is None:
            self.hg_n = sum(1 for v in self.C if v.minimiser())

        return self.hg_n

    def homology_group_differential(self):
        """
        Returns the difference between the current homology group of the
        cell and it's parent group
        """
        if self.hg_d is None:
            self.hgd = self.hg_n - self.p_hgr

        return self.hgd

    def polytopial_sperner_lemma(self):
        """
        Returns the number of stationary points theoretically contained in the
        cell based information currently known about the cell
        """
        pass

    def print_out(self):
        """
        Print the current cell to console
        """
        for v in self():
            v.print_out()


class Cell(VertexGroup):
    """
    Contains a cell that is symmetric to the initial hypercube triangulation
    """

    def __init__(self, p_gen, p_hgr, origin, supremum):
        super(Cell, self).__init__(p_gen, p_hgr)

        self.origin = origin
        self.supremum = supremum
        self.centroid = None  # (Not always used)
        # TODO: self.bounds


class Simplex(VertexGroup):
    """
    Contains a simplex that is symmetric to the initial symmetry constrained
    hypersimplex triangulation
    """

    def __init__(self, p_gen, p_hgr, generation_cycle, dim):
        super(Simplex, self).__init__(p_gen, p_hgr)

        self.generation_cycle = (generation_cycle + 1) % (dim - 1)


