"""
Base classes for low memory simplicial complex structures.

TODO: -Allow for sub-triangulations to track arbitrary points. Detect which
      simplex it is in and then connect the new points to it
      -Turn the triangulation into a generator that yields a specified number
      of finite points.


FUTURE: Triangulate arbitrary domains other than n-cubes
(ex. using delaunay and low disc. sampling subject to constraints, or by adding
     n-cubes and other geometries)

     Starting point:
     An algorithm for automatic Delaunay triangulation of arbitrary planar domains
     https://www.sciencedirect.com/science/article/pii/096599789600004X
"""
# Std. Library
import copy
import logging
import os
from abc import ABC, abstractmethod
# Required modules:
import numpy

# Optional modules:
try:
    import matplotlib
    from matplotlib import pyplot
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.tri import Triangulation
    from mpl_toolkits.mplot3d import axes3d, Axes3D, proj3d
    from hyperct._plotting import Arrow3D
except ImportError:
    logging.warning("Plotting functions are unavailable. To use install "
                    "matplotlib, install using ex. `pip install matplotlib` ")
    matplotlib_available = False
else:
    matplotlib_available = True

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

# Module specific imports
from hyperct._vertex import (VertexCacheIndex, VertexCacheField)


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

        if vfield is not None:
            logging.warning("Vector field applications have not been "
                            "implemented yet.")

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
            # del(self.V)
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

    # Plots
    def plot_complex(self, show=True, directed=True, contour_plot=True,
                     surface_plot=True, surface_field_plot=1,
                     minimiser_points=True, point_color='do', line_color='do',
                     complex_color_f='lo', complex_color_e='do', pointsize=7,
                     no_grids=False, save_fig=True, strpath=None,
                     plot_path='fig/', fig_name='complex.pdf', arrow_width=None
                     ):
        """
        Plots the current simplicial complex contained in the class. It requires
        at least one vector in the self.V to have been defined.


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
        :return: self.fig_surface, a matplotlib Axes class containing the complex surface and field surface

        Examples
        --------
        # Initiate a complex class
        >>> import pylab
        >>> H = Complex(2, sfield=func,domain=[(0, 10)])

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
        """
        if not matplotlib_available:
            logging.warning("Plotting functions are unavailable. To "
                            "install matplotlib install using ex. `pip install "
                            "matplotlib` ")
            return

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
            if surface_plot:
                surface_plot = False
                logging.warning("Warning, no associated scalar field found. "
                                "Not plotting complex surface field.")
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
            if surface_plot:
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
            if surface_plot:
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


        if show:
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
            xr = numpy.linspace(self.bounds[0][0], self.bounds[0][1], num=100)
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
                            cmap=cm.plasma,
                            # cmap=cm.inferno,
                            # cmap=cm.pink,
                            # cmap=cm.viridis,
                            linewidth=0,
                            antialiased=True, alpha=0.6, shade=True)

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

    # Conversions
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
        self.vertices_fm = []  # Vertices (A list of ob
        self.simplices_fm = []  # Faces
        self.simplices_fm_i = []

        # TODO: Add in field

        for v in self.V.cache:  # Note that cache is an OrderedDict
            self.vertices_fm.append(v)
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
