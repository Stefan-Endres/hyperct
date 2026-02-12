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

TODO: -Check domains for degeneracy

TODO: -Method to purge infeasible points

TODO: -Add hash tables to store vector pair values in the ab_C container in
       refine_local_space method

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
import collections
import logging
import os
import itertools
import json
import decimal

from abc import ABC, abstractmethod
# Required modules:
import numpy

# Standalone plotting functions
from . import _plotting
# Standalone DEC functions
from . import dec as _dec

from functools import cache
import multiprocessing as mp

# Module specific imports
from ._vertex import (VertexCacheIndex, VertexCacheField)
from ._field import (FieldCache, ScalarFieldCache)
from ._vertex_group import (Subgroup, Cell, Simplex)


def fdum(x):
    """Pickle-able dummy function for parallel boundary computation."""
    return 0


# Main complex class:
class Complex:
    def __init__(self, dim, domain=None, sfield=None, sfield_args=(),
                 vfield=None, vfield_args=None,
                 symmetry=None, constraints=None,
                 workers=None):
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

        constraints : dict or sequence of dict, optional
            Constraints definition.
            Function(s) ``R**n`` in the form::

                g(x) <= 0 applied as g : R^n -> R^m
                h(x) == 0 applied as h : R^n -> R^p

            Each constraint is defined in a dictionary with fields:

                type : str
                    Constraint type: 'eq' for equality, 'ineq' for inequality.
                fun : callable
                    The function defining the constraint.
                jac : callable, optional
                    The Jacobian of `fun` (only for SLSQP).
                args : sequence, optional
                    Extra arguments to be passed to the function and Jacobian.

            Equality constraint means that the constraint function result is to
            be zero whereas inequality means that it is to be non-negative.constraints : dict or sequence of dict, optional
            Constraints definition.
            Function(s) ``R**n`` in the form::

                g(x) <= 0 applied as g : R^n -> R^m
                h(x) == 0 applied as h : R^n -> R^p

            Each constraint is defined in a dictionary with fields:

                type : str
                    Constraint type: 'eq' for equality, 'ineq' for inequality.
                fun : callable
                    The function defining the constraint.
                jac : callable, optional
                    The Jacobian of `fun` (unused).
                args : sequence, optional
                    Extra arguments to be passed to the function and Jacobian.

            Equality constraint means that the constraint function result is to
            be zero whereas inequality means that it is to be non-negative.
        """
        self.dim = dim

        # Domains
        self.domain = domain
        if domain is None:
            self.bounds = [(0.0, 1.0), ] * dim
        else:
            self.bounds = domain  # TODO: Assert that len(domain) is dim
        self.symmetry = symmetry  # TODO: Define the functions to be used
        #      here in init to avoid if checks

        # Field functions
        self.sfield = sfield
        self.vfield = vfield
        self.sfield_args = sfield_args
        self.vfield_args = vfield_args

        # Process constraints
        # Constraints
        # Process constraint dict sequence:
        if constraints is not None:
            self.min_cons = constraints
            self.g_cons = []
            self.g_args = []
            if not isinstance(constraints, tuple | list):
                constraints = (constraints,)

            for cons in constraints:
                if cons['type'] == 'ineq':
                    self.g_cons.append(cons['fun'])
                    try:
                        self.g_args.append(cons['args'])
                    except KeyError:
                        self.g_args.append(())
            self.g_cons = tuple(self.g_cons)
            self.g_args = tuple(self.g_args)

        else:
            self.g_cons = None
            self.g_args = None

        # Homology properties
        self.gen = 0
        self.perm_cycle = 0

        # Every cell is stored in a list of its generation,
        # ex. the initial cell is stored in self.H[0]
        # 1st get new cells are stored in self.H[1] etc.
        # When a cell is sub-generated it is removed from this list

        self.H = []  # Storage structure of vertex groups #TODO: Dated?

        # Cache of all vertices
        if (sfield is not None) or (self.g_cons is not None):
            # Initiate a vertex cache and an associated field cache, note that
            # the field case is always initiated inside the vertex cache if an
            # associated field scalar field is defined:
            if sfield is not None:
                self.V = VertexCacheField(field=sfield, field_args=sfield_args,
                                          g_cons=self.g_cons,
                                          g_cons_args=self.g_args,
                                          workers=workers)
            elif self.g_cons is not None:
                self.V = VertexCacheField(field=sfield, field_args=sfield_args,
                                          g_cons=self.g_cons,
                                          g_cons_args=self.g_args,
                                          workers=workers)
        else:
            self.V = VertexCacheIndex()

        self.V_non_symm = []  # List of non-symmetric vertices

        if vfield is not None:
            logging.warning("Vector field applications have not been "
                            "implemented yet.")

    def __call__(self):
        return self.H

    # %% Triangulation methods
    def cyclic_product(self, bounds, origin, supremum, centroid=True,
                       printout=False):
        #TODO: Connect every vertex with the supremum before yielding
        #TODO: Change self.origin and self. supremum to local origin
        vot = tuple(origin)
        vut = tuple(supremum)  # Hyperrectangle supremum
        self.V[vot]
        vo = self.V[vot]
        yield vo.x
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
        a_vo = copy.copy(list(origin))
        a_vo[0] = vut[0]  # Update aN Origin
        a_vo = self.V[tuple(a_vo)]
        #self.V[vot].connect(self.V[tuple(a_vo)])
        self.V[vot].connect(a_vo)
        yield a_vo.x
        C1x = [[a_vo]]
        #C1x = [[self.V[tuple(a_vo)]]]
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

                        # Connect vertices in N to corresponding vertices
                        # in aN:
                        vl.connect(a_vl)

                        yield a_vl.x

                        a_vu = self.V[tuple(a_vu)]
                        # Connect vertices in N to corresponding vertices
                        # in aN:
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

                        # Yield new points
                        yield a_vu.x

                # Try to connect aN lower source of previous a + b
                # operation with a aN vertex
                ab_Cc = copy.copy(ab_C)

                #TODO: SHOULD THIS BE MOVED OUTSIDE THE try ?
                for vp in ab_Cc:
                    b_v = list(vp[0].x)
                    ab_v = list(vp[1].x)
                    b_v[i + 1] = vut[i + 1]
                    ab_v[i + 1] = vut[i + 1]
                    b_v = self.V[tuple(b_v)]  # b + vl
                    ab_v = self.V[tuple(ab_v)]  # b + a_vl
                    # Note o---o is already connected
                    vp[0].connect(ab_v)  # o-s
                    b_v.connect(ab_v)  # s-s

                    # Add new list of cross pairs
                    ab_C.append((vp[0], ab_v))
                    ab_C.append((b_v, ab_v))

            except IndexError:
                #TODO: The above routine works for symm inits, so maybe test if
                # vl exists and then work from there?
                # Copy lists for iteration
                # cC0x = [x[:] for x in C0x[:i + 1]]
                cC0x = C0x[i]
                cC1x = C1x[i]
                VL, VU = cC0x, cC1x
                for k, (vl, vu) in enumerate(zip(VL, VU)):
                    print(f'vl = {vl.x}')
                    print(f'vu = {vu.x}')
                    # Build aN vertices for each lower-upper pair in N:
                    a_vu = list(vu.x)
                    a_vu[i + 1] = vut[i + 1]

                    # Connect vertices in N to corresponding vertices
                    # in aN:
                    a_vu = self.V[tuple(a_vu)]
                    # Connect vertices in N to corresponding vertices
                    # in aN:
                    vu.connect(a_vu)
                    # Connect new vertex pair in aN:
                    # a_vl.connect(a_vu)
                    # Connect lower pair to upper (triangulation
                    # operation of a + b (two arbitrary operations):
                    vl.connect(a_vu)
                    ab_C.append((vl, a_vu))

                    #TODO: LOOK AT THIS AGAIN:
                    # Update the containers
                    #C0x[i + 1].append(vl)
                    C0x[i + 1].append(vu)
                    # C1x[i + 1].append(a_vl)
                    C1x[i + 1].append(a_vu)

                    # Update old containers
                    # C0x[j].append(a_vl)
                    # C1x[j].append(a_vu)

                    # Yield new points
                    a_vu.connect(self.V[vut])
                    yield a_vu.x
                    #TODO: Need to replicate last cube once?
                    # Try to connect aN lower source of previous a + b
                    # operation with a aN vertex
                    ab_Cc = copy.copy(ab_C)
                    for vp in ab_Cc:
                        for v in vp:
                            print(f'vp = {v.x}')

                        print(f'-')
                        if vp[1].x[i] == vut[i]:
                       #     b_v = list(vp[0].x)
                            ab_v = list(vp[1].x)
                        #    b_v[i + 1] = vut[i + 1]
                            ab_v[i + 1] = vut[i + 1]
                            print(f'ab_v = {ab_v}')
                        #    b_v = self.V[tuple(b_v)]  # b + vl
                            ab_v = self.V[tuple(ab_v)]  # b + a_vl
                            # Note o---o is already connected
                            vp[0].connect(ab_v)  # o-s
                        #    b_v.connect(ab_v)  # s-s

                            # Add new list of cross pairs
                            ab_C.append((vp[0], ab_v))
                       #     ab_C.append((b_v, ab_v))

            # Printing
            if printout:
                print("=" * 19)
                print("Current symmetry group:")
                print("=" * 19)
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

        # Extra yield to ensure that the triangulation is completed
        if centroid:
            vo = self.V[vot]
            vs = self.V[vut]
            # Disconnect the origin and supremum
            vo.disconnect(vs)
            # Build centroid
            vc = self.split_edge(vot, vut)
            # TODO: If not initial triangulation, we'll need to use a different
            # container
            for v in vo.nn:
                v.connect(vc)
            yield vc.x
            return vc.x
        else:
            yield vut
            return vut


    def triangulate(self, n=None, symmetry=None, centroid=True, printout=False):
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
        # Inherit class arguments
        if symmetry is None:
            symmetry = self.symmetry
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
            self.cp = self.cyclic_product(cbounds, origin, supremum, centroid,
                                          printout)
            for i in self.cp:
                i

            try:
                self.triangulated_vectors.append((tuple(self.origin),
                                                  tuple(self.supremum)))
            except (AttributeError, KeyError):
                self.triangulated_vectors = [(tuple(self.origin),
                                              tuple(self.supremum))]

        else:
            #Check if generator already exists
            try:
                self.cp
            except (AttributeError, KeyError):
                self.cp = self.cyclic_product(cbounds, origin, supremum,
                                              centroid, printout)

            try:
                while len(self.V.cache) < n:
                    next(self.cp)
            except StopIteration:
                #TODO: We should maybe append and add the possibility of
                #      of starting new triangulated domains on different
                #      complexes
                try:
                    self.triangulated_vectors.append((tuple(self.origin),
                                                      tuple(self.supremum)))
                except (AttributeError, KeyError):
                    self.triangulated_vectors = [(tuple(self.origin),
                                                  tuple(self.supremum))]


        if printout:
            print("=" * 19)
            print("Initial hyper cube:")
            print("=" * 19)
            # for v in self.C0():
            #   v.print_out()
            for v in self.V.cache:
                self.V[v].print_out()

            print("=" * 19)

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
    def refine(self, n=1):
        #TODO: Replace with while loop checking cache size instead?
        if n is None:
            try:
                self.triangulated_vectors
                self.refine_all()
                return
            except AttributeError as ae:
                if str(ae) == "'Complex' object has no attribute " \
                              "'triangulated_vectors'":
                    self.triangulate(symmetry=self.symmetry)
                    return
                else:
                    raise

        nt = len(self.V.cache) + n  # Target number of total vertices
        # In the outer while loop we iterate until we have added an extra `n`
        # vertices to the complex:
        #TODO: We can reduce the number of exception handling and exceptions to
        #  StopIteration if we try to generate the initial objects first and
        #  then refine only have the first few while loop runs
        while len(self.V.cache) < nt:  # while loop 1
            try:  # try 1
                # Try to access triangulated_vectors, this should only be
                # defined if an initial triangulation has already been performed
                self.triangulated_vectors
                # Try a usual iteration of the current generator, if it
                # does not exist or is exhausted then produce a new generator
                try:  # try 2
                    #next(self.rls)
                    #print(f'next(self.rls) = {next(self.rls)}')
                    next(self.rls)
                except (AttributeError, StopIteration, KeyError):
                    vp = self.triangulated_vectors[0]
                    #print(f'vp = {vp}')
                    self.rls = self.refine_local_space(*vp, bounds=self.bounds)
                    # next(self.rls)
                    #print(f'next(self.rls) = {next(self.rls)}')
                    next(self.rls)

            except (AttributeError, KeyError):
                # If an initial triangulation has not been completed, then
                # we start/continue the initial triangulation targeting `nt`
                # vertices, if nt is greater than the initial number of vertices
                # then the refine routine will move back to try 1.
                self.triangulate(nt, self.symmetry)
        return

    def refine_all(self, centroids=True):
        """
        Refine the entire domain of the current complex
        :return:
        """

        if 0:
            try:
                vn_pool_sets = self.vn_pool_sets
            except (AttributeError, KeyError):
                vn_pool_sets = []
                for vp in tvs:
                    vn_pool_sets.append(self.vpool(*vp))

        try:
            self.triangulated_vectors
            tvs = copy.copy(self.triangulated_vectors)
            for i, vp in enumerate(tvs):
                #self.rls =self.refine_local_space_c(*vp, vpool=vn_pool_sets[i])
                self.rls = self.refine_local_space(*vp, bounds=self.bounds)
                for i in self.rls:
                    i
        except AttributeError as ae:
            if str(ae) == "'Complex' object has no attribute " \
                          "'triangulated_vectors'":
                self.triangulate(symmetry=self.symmetry, centroid=centroids)
            else:
                raise

            #self.triangulate(symmetry=symmetry, centroid=centroids)

        # This adds a centroid to every new sub-domain generated and defined
        # by self.triangulated_vectors, in addition the vertices ! to complete
        # the triangulation
        if 0:
            if centroids:
                tvs = copy.copy(self.triangulated_vectors)
                self.vn_pool_sets = []
                for vp in tvs:
                    self.vn_pool_sets.append(self.vpool(*vp))
                vcg = self.vc_gen(self.vn_pool_sets)
                for vc in vcg:
                    print(f'vc = {vc}')
                    vc
        return

    def refine_local_space(self, origin, supremum, bounds, centroid=1,
                              printout=0):
        # Copy for later removal
        origin_c = copy.copy(origin)
        supremum_c = copy.copy(supremum)

        # Change the vector orientation so that it is only increasing
        s_ov = list(origin)
        s_origin = list(origin)
        s_sv = list(supremum)
        s_supremum = list(supremum)
        for i, vi in enumerate(s_origin):
            if s_ov[i] > s_sv[i]:
                s_origin[i] = s_sv[i]
                s_supremum[i] = s_ov[i]

        vot = tuple(s_origin)
        vut = tuple(s_supremum)  # Hyperrectangle supremum

        vo = self.V[vot]  # initiate if doesn't exist yet
        vs = self.V[vut]
        # Start by finding the old centroid of the new space:
        vco = self.split_edge(vo.x, vs.x)  # Split in case not centroid arg

        # Find set of extreme vertices in current local space
        sup_set = copy.copy(vco.nn)
        # Cyclic group approach with second x_l --- x_u operation.

        # These containers store the "lower" and "upper" vertices
        # corresponding to the origin or supremum of every C2 group.
        # It has the structure of `dim` times embedded lists each containing
        # these vertices as the entire complex grows. Bounds[0] has to be done
        # outside the loops before we have symmetric containers.
        #NOTE: This means that bounds[0][1] must always exist

        a_vl = copy.copy(list(vot))
        a_vl[0] = vut[0]  # Update aN Origin
        if tuple(a_vl) not in self.V.cache:
            vo = self.V[vot]  # initiate if doesn't exist yet
            vs = self.V[vut]
            # Start by finding the old centroid of the new space:
            vco = self.split_edge(vo.x, vs.x)  # Split in case not centroid arg

            # Find set of extreme vertices in current local space
            sup_set = copy.copy(vco.nn)
            a_vl = copy.copy(list(vot))
            a_vl[0] = vut[0]  # Update aN Origin
            a_vl = self.V[tuple(a_vl)]
        else:
            a_vl = self.V[tuple(a_vl)]

        c_v = self.split_edge(vo.x, a_vl.x)
        c_v.connect(vco)
        yield c_v.x
        Cox = [[vo]]
        Ccx = [[c_v]]
        Cux = [[a_vl]]
        ab_C = []  # Container for a + b operations
        s_ab_C = []  # Container for symmetric a + b operations

        # Loop over remaining bounds
        for i, x in enumerate(bounds[1:]):
            # Update lower and upper containers
            Cox.append([])
            Ccx.append([])
            Cux.append([])
            # try to access a second bound (if not, C1 is symmetric)
            try:
                t_a_vl = list(vot)
                t_a_vl[i + 1] = vut[i + 1]

                # New: lists are used anyway, so copy all
                # %%
                # Copy lists for iteration
                cCox = [x[:] for x in Cox[:i + 1]]
                cCcx = [x[:] for x in Ccx[:i + 1]]
                cCux = [x[:] for x in Cux[:i + 1]]
                # Try to connect aN lower source of previous a + b
                # operation with a aN vertex
                ab_Cc = copy.copy(ab_C)  # NOTE: We append ab_C in the
                # (VL, VC, VU) for-loop, but we use the copy of the list in the
                # ab_Cc for-loop.
                s_ab_Cc = copy.copy(s_ab_C)

                #%%
                # Early try so that we don't have to copy the cache before
                # moving on to next C1/C2: Try to add the operation of a new
                # C2 product by accessing the upper bound
                if tuple(t_a_vl) not in self.V.cache:
                    raise IndexError  # Raise error to continue symmetric refine
                t_a_vu = list(vut)
                t_a_vu[i + 1] = vut[i + 1]
                if tuple(t_a_vu) not in self.V.cache:
                    raise IndexError  # Raise error to continue symmetric refine

                # TODO: SHOULD THIS BE MOVED OUTSIDE THE try ?
                for vectors in s_ab_Cc:
                    # s_ab_C.append([c_vc, vl, vu, a_vu])
                    bc_vc = list(vectors[0].x)
                    b_vl = list(vectors[1].x)
                    b_vu = list(vectors[2].x)
                    ba_vu = list(vectors[3].x)

                    bc_vc[i + 1] = vut[i + 1]
                    b_vl[i + 1] = vut[i + 1]
                    b_vu[i + 1] = vut[i + 1]
                    ba_vu[i + 1] = vut[i + 1]

                    bc_vc = self.V[tuple(bc_vc)]
                    bc_vc.connect(vco)  # NOTE: Unneeded?
                    yield bc_vc

                    # Split to centre, call this centre group "d = 0.5*a"
                    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                    d_bc_vc.connect(bc_vc)
                    d_bc_vc.connect(vectors[1])  # Connect all to centroid
                    d_bc_vc.connect(vectors[2])  # Connect all to centroid
                    d_bc_vc.connect(vectors[3])  # Connect all to centroid
                    yield d_bc_vc.x
                    b_vl = self.V[tuple(b_vl)]
                    bc_vc.connect(b_vl)  # Connect aN cross pairs
                    d_bc_vc.connect(b_vl)  # Connect all to centroid


                    yield b_vl
                    b_vu = self.V[tuple(b_vu)]
                    bc_vc.connect(b_vu)  # Connect aN cross pairs
                    d_bc_vc.connect(b_vu)  # Connect all to centroid

                    #TODO: Should be done in the cyclical loop somehow?
                    b_vl_c = self.split_edge(b_vu.x, b_vl.x)
                    bc_vc.connect(b_vl_c)
                    ###

                    yield b_vu
                    ba_vu = self.V[tuple(ba_vu)]
                    bc_vc.connect(ba_vu)  # Connect aN cross pairs
                    d_bc_vc.connect(ba_vu)  # Connect all to centroid

                    # Split the a + b edge of the initial triangulation:
                    os_v = self.split_edge(vectors[1].x, ba_vu.x)  # o-s
                    ss_v = self.split_edge(b_vl.x, ba_vu.x)  # s-s

                    #TODO: Should be done in the cyclical loop somehow?
                    b_vu_c = self.split_edge(b_vu.x, ba_vu.x)
                    bc_vc.connect(b_vu_c)
                    yield os_v.x  # often equal to vco, but not always
                    yield ss_v.x  # often equal to bc_vu, but not always
                    yield ba_vu
                    # Split remaining to centre, call this centre group "d = 0.5*a"
                    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    yield d_bc_vc.x
                    d_b_vl = self.split_edge(vectors[1].x, b_vl.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_b_vl)  # Connect dN cross pairs
                    yield d_b_vl.x
                    d_b_vu = self.split_edge(vectors[2].x, b_vu.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_b_vu)  # Connect dN cross pairs
                    yield d_b_vu.x
                    d_ba_vu = self.split_edge(vectors[3].x, ba_vu.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_ba_vu)  # Connect dN cross pairs
                    yield d_ba_vu

                    # comb = [c_vc, vl, vu, a_vl, a_vu,
                    #       bc_vc, b_vl, b_vu, ba_vl, ba_vu]
                    comb = [vl, vu, a_vu,
                            b_vl, b_vu, ba_vu]
                    comb_iter = itertools.combinations(comb, 2)
                    for vecs in comb_iter:
                        self.split_edge(vecs[0].x, vecs[1].x)
                    # Add new list of cross pairs
                    #ab_C.append((bc_vc, b_vl, b_vu, ba_vl, ba_vu))
                    #ab_C.append((d_bc_vc, d_b_vl, d_b_vu, d_ba_vl, d_ba_vu))
                    ab_C.append((d_bc_vc, vectors[1], b_vl, a_vu, ba_vu))
                    ab_C.append((d_bc_vc, vl, b_vl, a_vu, ba_vu))  # = prev
                    #ab_C.append((d_bc_vc, vu, b_vu, a_vl, ba_vl))

                # TODO: SHOULD THIS BE MOVED OUTSIDE THE try ?
                for vectors in ab_Cc:
                    #TODO: Make for loops instead of using variables
                    bc_vc = list(vectors[0].x)
                    b_vl = list(vectors[1].x)
                    b_vu = list(vectors[2].x)
                    ba_vl = list(vectors[3].x)
                    ba_vu = list(vectors[4].x)
                    bc_vc[i + 1] = vut[i + 1]
                    b_vl[i + 1] = vut[i + 1]
                    b_vu[i + 1] = vut[i + 1]
                    ba_vl[i + 1] = vut[i + 1]
                    ba_vu[i + 1] = vut[i + 1]
                    bc_vc = self.V[tuple(bc_vc)]
                    bc_vc.connect(vco)  # NOTE: Unneeded?
                    yield bc_vc

                    # Split to centre, call this centre group "d = 0.5*a"
                    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                    d_bc_vc.connect(bc_vc)
                    d_bc_vc.connect(vectors[1])  # Connect all to centroid
                    d_bc_vc.connect(vectors[2])  # Connect all to centroid
                    d_bc_vc.connect(vectors[3])  # Connect all to centroid
                    d_bc_vc.connect(vectors[4])  # Connect all to centroid
                    yield d_bc_vc.x
                    b_vl = self.V[tuple(b_vl)]
                    bc_vc.connect(b_vl)  # Connect aN cross pairs
                    d_bc_vc.connect(b_vl)  # Connect all to centroid
                    yield b_vl
                    b_vu = self.V[tuple(b_vu)]
                    bc_vc.connect(b_vu)  # Connect aN cross pairs
                    d_bc_vc.connect(b_vu)  # Connect all to centroid
                    yield b_vu
                    ba_vl = self.V[tuple(ba_vl)]
                    bc_vc.connect(ba_vl)  # Connect aN cross pairs
                    d_bc_vc.connect(ba_vl)  # Connect all to centroid
                    self.split_edge(b_vu.x, ba_vl.x)
                    yield ba_vl
                    ba_vu = self.V[tuple(ba_vu)]
                    bc_vc.connect(ba_vu)  # Connect aN cross pairs
                    d_bc_vc.connect(ba_vu)  # Connect all to centroid
                    # Split the a + b edge of the initial triangulation:
                    os_v = self.split_edge(vectors[1].x, ba_vu.x)  # o-s
                    ss_v = self.split_edge(b_vl.x, ba_vu.x)  # s-s
                    yield os_v.x  # often equal to vco, but not always
                    yield ss_v.x  # often equal to bc_vu, but not always
                    yield ba_vu
                    # Split remaining to centre, call this centre group "d = 0.5*a"
                    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    yield d_bc_vc.x
                    d_b_vl = self.split_edge(vectors[1].x, b_vl.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_b_vl)  # Connect dN cross pairs
                    yield d_b_vl.x
                    d_b_vu = self.split_edge(vectors[2].x, b_vu.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_b_vu)  # Connect dN cross pairs
                    yield d_b_vu.x
                    d_ba_vl = self.split_edge(vectors[3].x, ba_vl.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_ba_vl)  # Connect dN cross pairs
                    yield d_ba_vl
                    d_ba_vu = self.split_edge(vectors[4].x, ba_vu.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_ba_vu)  # Connect dN cross pairs
                    yield d_ba_vu

                    #ab_C.append((c_vc, vl, vu, a_vl, a_vu))  # CHECK

                    c_vc, vl, vu, a_vl, a_vu = vectors

                    comb = [vl, vu, a_vl, a_vu,
                            b_vl, b_vu, ba_vl, ba_vu]
                    comb_iter = itertools.combinations(comb, 2)
                    for vecs in comb_iter:
                        self.split_edge(vecs[0].x, vecs[1].x)

                    # Add new list of cross pairs
                    ab_C.append((bc_vc, b_vl, b_vu, ba_vl, ba_vu))
                    ab_C.append((d_bc_vc, d_b_vl, d_b_vu, d_ba_vl, d_ba_vu))
                    ab_C.append((d_bc_vc, vectors[1], b_vl, a_vu, ba_vu))
                    ab_C.append((d_bc_vc, vu, b_vu, a_vl, ba_vl))

                for j, (VL, VC, VU) in enumerate(zip(cCox, cCcx, cCux)):
                    for k, (vl, vc, vu) in enumerate(zip(VL, VC, VU)):
                        # Build aN vertices for each lower-upper C3 group in N:
                        a_vl = list(vl.x)
                        a_vu = list(vu.x)
                        a_vl[i + 1] = vut[i + 1]
                        a_vu[i + 1] = vut[i + 1]
                        a_vl = self.V[tuple(a_vl)]
                        a_vu = self.V[tuple(a_vu)]
                        # Note, build (a + vc) later for consistent yields
                        # Split the a + b edge of the initial triangulation:
                        c_vc = self.split_edge(vl.x, a_vu.x)
                        self.split_edge(vl.x, vu.x)  # Equal to vc
                        # Build cN vertices for each lower-upper C3 group in N:
                        c_vc.connect(vco)
                        c_vc.connect(vc)
                        c_vc.connect(vl)  # Connect c + ac operations
                        c_vc.connect(vu)  # Connect c + ac operations
                        c_vc.connect(a_vl)  # Connect c + ac operations
                        c_vc.connect(a_vu)  # Connect c + ac operations
                        yield(c_vc.x)
                        c_vl = self.split_edge(vl.x, a_vl.x)
                        c_vl.connect(vco)
                        c_vc.connect(c_vl)  # Connect cN group vertices
                        yield(c_vl.x)
                        c_vu = self.split_edge(vu.x, a_vu.x) # yield at end of loop
                        c_vu.connect(vco)
                        # Connect remaining cN group vertices
                        c_vc.connect(c_vu)  # Connect cN group vertices
                        yield (c_vu.x)

                        a_vc = self.split_edge(a_vl.x, a_vu.x)  # is (a + vc) ?
                        a_vc.connect(vco)
                        a_vc.connect(c_vc)

                        # Storage for connecting c + ac operations:
                        ab_C.append((c_vc, vl, vu, a_vl, a_vu))

                        # Update the containers
                        Cox[i + 1].append(vl)
                        Cox[i + 1].append(vc)
                        Cox[i + 1].append(vu)
                        Ccx[i + 1].append(c_vl)
                        Ccx[i + 1].append(c_vc)
                        Ccx[i + 1].append(c_vu)
                        Cux[i + 1].append(a_vl)
                        Cux[i + 1].append(a_vc)
                        Cux[i + 1].append(a_vu)

                        # Update old containers
                        Cox[j].append(c_vl)  # !
                        Cox[j].append(a_vl)
                        Ccx[j].append(c_vc)  # !
                        Ccx[j].append(a_vc)  # !
                        Cux[j].append(c_vu)  # !
                        Cux[j].append(a_vu)

                        # Yield new points
                        yield(a_vc.x)

            except IndexError:
                if 1:
                    print(f'===')
                    print(f'i = {i}')
                    print(f'dim = {i + 1}')
                    print(f'===')
                    # TODO: The above routine works for symm inits, so maybe test if
                    # vl exists and then work from there?

                    if 1:
                        for vectors in ab_Cc:
                            # ab_C.append((c_vc, vl, vu, a_vl, a_vu))
                            # TODO: Make for loops instead of using variables
                            # ab_C.append((c_vc, vl, vu, a_vl, a_vu))
                        #    bc_vc = list(vectors[0].x)
                            #b_vl = list(vectors[1].x)
                            #b_vu = list(vectors[2].x)
                            ba_vl = list(vectors[3].x)
                            ba_vu = list(vectors[4].x)
                        #    bc_vc[i + 1] = vut[i + 1]
                            #b_vl[i + 1] = vut[i + 1]
                            #b_vu[i + 1] = vut[i + 1]
                            ba_vl[i + 1] = vut[i + 1]
                            ba_vu[i + 1] = vut[i + 1]
                        #    bc_vc = self.V[tuple(bc_vc)]
                        #    bc_vc.connect(vco)  # NOTE: Unneeded?
                        #    yield bc_vc

                            # Split to centre, call this centre group "d = 0.5*a"
                        #    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                         #   d_bc_vc = self.split_edge(vectors[1].x, bc_vc.x)

                            ba_vu = self.V[tuple(ba_vu)]
                            yield ba_vu
                            #    bc_vc.connect(ba_vu)  # Connect aN cross pairs
                            #    d_bc_vc.connect(ba_vu)  # Connect all to centroid
                            # Split the a + b edge of the initial triangulation:
                            d_bc_vc = self.split_edge(vectors[1].x, ba_vu.x)  # o-s
                            # ss_v = self.split_edge(b_vl.x, ba_vu.x)  # s-s
                            # yield os_v.x  # often equal to vco, but not always
                            # yield ss_v.x  # often equal to bc_vu, but not always
                            yield ba_vu
                        #    d_bc_vc.connect(bc_vc)
                            d_bc_vc.connect(vectors[1])  # Connect all to centroid
                            d_bc_vc.connect(vectors[2])  # Connect all to centroid
                            d_bc_vc.connect(vectors[3])  # Connect all to centroid
                            d_bc_vc.connect(vectors[4])  # Connect all to centroid
                            yield d_bc_vc.x
                            #b_vl = self.V[tuple(b_vl)]
                            #bc_vc.connect(b_vl)  # Connect aN cross pairs
                            #d_bc_vc.connect(b_vl)  # Connect all to centroid
                            #yield b_vl
                            #b_vu = self.V[tuple(b_vu)]
                            #bc_vc.connect(b_vu)  # Connect aN cross pairs
                            #d_bc_vc.connect(b_vu)  # Connect all to centroid
                            #yield b_vu
                            ba_vl = self.V[tuple(ba_vl)]
                            #bc_vc.connect(ba_vl)  # Connect aN cross pairs
                            #d_bc_vc.connect(ba_vl)  # Connect all to centroid
                            if 0:  # Needed for 3D tests to pass
                                self.split_edge(b_vu.x, ba_vl.x)
                            yield ba_vl
                         #   ba_vu = self.V[tuple(ba_vu)]
                        #    bc_vc.connect(ba_vu)  # Connect aN cross pairs
                        #    d_bc_vc.connect(ba_vu)  # Connect all to centroid
                            # Split the a + b edge of the initial triangulation:
                            #os_v = self.split_edge(vectors[1].x, ba_vu.x)  # o-s
                            #ss_v = self.split_edge(b_vl.x, ba_vu.x)  # s-s
                            #yield os_v.x  # often equal to vco, but not always
                            #yield ss_v.x  # often equal to bc_vu, but not always
                            #yield ba_vu

                            # Split remaining to centre, call this centre group "d = 0.5*a"
                        #    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                        #    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                        #    yield d_bc_vc.x
                            #d_b_vl = self.split_edge(vectors[1].x, b_vl.x)
                        #    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                            #d_bc_vc.connect(d_b_vl)  # Connect dN cross pairs
                            #yield d_b_vl.x
                            #d_b_vu = self.split_edge(vectors[2].x, b_vu.x)
                        #    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                            #d_bc_vc.connect(d_b_vu)  # Connect dN cross pairs
                            #yield d_b_vu.x
                            d_ba_vl = self.split_edge(vectors[3].x, ba_vl.x)
                        #    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                            #d_bc_vc.connect(d_ba_vl)  # Connect dN cross pairs
                            #yield d_ba_vl
                            d_ba_vu = self.split_edge(vectors[4].x, ba_vu.x)

                            d_ba_vc = self.split_edge(d_ba_vl.x, d_ba_vu.x)
                        #    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                        #    d_bc_vc.connect(d_ba_vu)  # Connect dN cross pairs
                            yield d_ba_vl
                            yield d_ba_vu
                            yield d_ba_vc

                            # ab_C.append((c_vc, vl, vu, a_vl, a_vu))  # CHECK

                            c_vc, vl, vu, a_vl, a_vu = vectors

                            # NEW: Try to split all possible combinations of new edges
                            # FIXED 4D TESTS!
                            # SLOW AS FUCK THOUGH!
                            if 1:

                                comb = [vl, vu, a_vl, a_vu,
                                        #b_vl,
                                        #b_vu,
                                        ba_vl,
                                        ba_vu]
                                comb_iter = itertools.combinations(comb, 2)
                                for vecs in comb_iter:
                                    self.split_edge(vecs[0].x, vecs[1].x)

                            if 0:  # DELETE:
                                self.split_edge(vl.x, vu.x)
                                self.split_edge(vl.x, a_vl.x)
                                self.split_edge(vl.x, a_vu.x)
                                self.split_edge(vu.x, a_vl.x)
                                self.split_edge(vu.x, a_vu.x)
                                self.split_edge(a_vl.x, a_vu.x)
                            # Add new list of cross pairs
                        #    ab_C.append((bc_vc, b_vl, b_vu, ba_vl, ba_vu))
                            #ab_C.append((d_bc_vc, d_b_vl, d_b_vu, d_ba_vl, d_ba_vu))
                            #ab_C.append((d_bc_vc, vectors[1], b_vl, a_vu, ba_vu))
                    #    ab_C.append((d_bc_vc, vu, b_vu, a_vl, ba_vl))
                        #TODO: ADD SYMMETRIC List

                    # Copy lists for iteration
                    cCox = Cox[i]
                    cCcx = Ccx[i]
                    cCux = Cux[i]
                    VL, VC, VU = cCox, cCcx, cCux
                    # for j, (VL, VU) in enumerate(zip(cC0x, cC1x)):
                    for k, (vl, vc, vu) in enumerate(zip(VL, VC, VU)):
                        print(f'vl = {vl.x}')
                        print(f'vc = {vc.x}')
                        print(f'vu = {vu.x}')
                        # Build aN vertices for each lower-upper pair in N:
                        # a_vl = list(vl.x)
                        a_vu = list(vu.x)
                        # a_vl[i + 1] = vut[i + 1]
                        a_vu[i + 1] = vut[i + 1]
                        # a_vl = self.V[tuple(a_vl)]

                        # Connect vertices in N to corresponding vertices
                        # in aN:
                        # TODO: CHECK:
                        #vl.connect(a_vl)
                        a_vu = self.V[tuple(a_vu)]
                        print(f'a_vu = {a_vu.x}')
                        yield a_vl.x
                        # Split the a + b edge of the initial triangulation:
                        c_vc = self.split_edge(vl.x, a_vu.x)
                        print(f'c_vc = {c_vc.x}')
                        self.split_edge(vl.x, vu.x)  # Equal to vc
                        c_vc.connect(vco)
                        c_vc.connect(vc)
                        c_vc.connect(vl)  # Connect c + ac operations
                        c_vc.connect(vu)  # Connect c + ac operations
                        #c_vc.connect(a_vl)  # Connect c + ac operations
                        c_vc.connect(a_vu)  # Connect c + ac operations
                        yield (c_vc.x)
                        print('--')
                        print(f'vu = {vu.x}')
                        print(f'a_vu = {a_vu.x}')

                        c_vu = self.split_edge(vu.x,
                                               a_vu.x)  # yield at end of loop
                        c_vu.connect(vco)
                        # Connect remaining cN group vertices
                        c_vc.connect(c_vu)  # Connect cN group vertices
                        print(f'c_vu.x= {c_vu.x}')
                        print('--')
                        yield (c_vu.x)

                        # TODO: CONTINUE HERE: ADD ALL ab_C pairs!

                        # Connect new vertex pair in aN:
                        # a_vl.connect(a_vu)

                        # Connect lower pair to upper (triangulation
                        # operation of a + b (two arbitrary operations):
                        #vl.connect(a_vu)
                        #ab_C.append((vl, a_vu))  # FIX

                        # TODO: LOOK AT THIS AGAIN:
                        # Update the containers

                        # C0x[i + 1].append(vl)
                    #    Cox[i + 1].append(vu)
                    #    Ccx[i + 1].append(c_vc)  #TODO: CHECK
                        # C1x[i + 1].append(a_vl)
                   #     Cux[i + 1].append(a_vu)

                        # Update old containers
                        # C0x[j].append(a_vl)
                        # C1x[j].append(a_vu)


                        # Update the containers
                      #  Cox[i + 1].append(vl)
                     #   Cox[i + 1].append(vc)
                        Cox[i + 1].append(vu)
                      #  Ccx[i + 1].append(c_vl)
                     #   Ccx[i + 1].append(c_vc)
                        Ccx[i + 1].append(c_vu)
                     #   Cux[i + 1].append(a_vl)
                      #  Cux[i + 1].append(a_vc)
                        Cux[i + 1].append(a_vu)

                        # Update old containers
                        #Cox[j].append(c_vl)  # !
                        #Cox[j].append(a_vl)
                        #Ccx[j].append(c_vc)  # !
                        #Ccx[j].append(a_vc)  # !
                        #Cux[j].append(c_vu)  # !
                        #Cux[j].append(a_vu)


                        # Yield new points
                     #   a_vu.connect(self.V[vut])

                        s_ab_C.append([c_vc, vl, vu, a_vu])

                        yield a_vu.x

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

        try:
            self.triangulated_vectors.remove((tuple(origin_c),
                                              tuple(supremum_c)))
        except ValueError:
            # Turn this into a logging warning?
            print('...')
            print('...')
            print('...')
            print('REMOVING FAILED')
            print(f' (tuple(origin), tuple(supremum))'
                  f'= {(tuple(origin), tuple(supremum))}')
            print('...')
            print('...')
            print('...')

        # Add newly triangulated vectors:
        for vs in sup_set:
            self.triangulated_vectors.append((tuple(vco.x), tuple(vs.x)))

        # Extra yield to ensure that the triangulation is completed
        if centroid:
            vcn_set = set()
            vcn_list = []
            c_nn_lists = []
            for vs in sup_set:
                # Build centroid
                c_nn = self.vpool(vco.x, vs.x)
                try:
                    c_nn.remove(vcn_set)
                except KeyError:
                    pass
                c_nn_lists.append(c_nn)

            for c_nn in c_nn_lists:
                try:
                    c_nn.remove(vcn_set)
                except KeyError:
                    pass

            for vs, c_nn in zip(sup_set, c_nn_lists):
                # Build centroid
                vcn = self.split_edge(vco.x, vs.x)
                vcn_set.add(vcn)
                try:  # Shouldn't be needed?
                    c_nn.remove(vcn_set)
                except KeyError:
                    pass
                for vnn in c_nn:
                    vcn.connect(vnn)
                yield vcn.x
        else:
            pass

        yield vut
        return

    def refine_star_old(self, v):
        """
        Refine the star domain of a vertex v
        :param v:
        :return:
        """
        # Copy lists before iteration
        vnn = copy.copy(v.nn)
        v1nn = []
        d_v0v1_set = set()
        for v1 in vnn:
            v1nn.append(copy.copy(v1.nn))

        for v1, v1nn in zip(vnn, v1nn):
            print('-')
            print(f'v1.x = {v1.x}')
            vnnu = v1nn.intersection(vnn)
            print(f'vnnu = {vnnu}')

            d_v0v1 = self.split_edge(v.x, v1.x)
            for o_d_v0v1 in d_v0v1_set:
                d_v0v1.connect(o_d_v0v1)
            d_v0v1_set.add(d_v0v1)
            for v2 in vnnu:
                print(f'v2 = {v2}')
                d_v1v2 = self.split_edge(v1.x, v2.x)
                d_v0v1.connect(d_v1v2)
        return

    @cache
    def split_edge(self, v1, v2):
        v1 = self.V[v1]
        v2 = self.V[v2]
    #    print(f'splitting {v1.x} --- {v2.x}')
        # Destroy original edge, if it exists:
        v1.disconnect(v2)
        # Compute vertex on centre of edge:
        try:
            vct = (v2.x_a - v1.x_a) / 2.0 + v1.x_a
        except TypeError:  # Allow for decimal operations
            #TODO: Only except float
            vct = (v2.x_a - v1.x_a) / decimal.Decimal(2.0) + v1.x_a
        vc = self.V[tuple(vct)]
        # Connect to original 2 vertices to the new centre vertex
        vc.connect(v1)
        vc.connect(v2)
        return vc

    def cut_g(self):
        self.V.process_pools()
        for v in self.V:
            #print(f'v.feasible = {v.feasible}')
            if not v.feasible:
                self.V.remove(v)

    # %% Refinement
    # % Refinement based on vector partitions
    def vtvs(self):
        """
        For every refinement generation, produce the `vpool` vertex clusters
        that is used to generate centroid vertices (1st stage) and then produce
        generators for the arguments to the `refine_local_space` method.
        :return:
        """
        vn_pool_sets = []
        for vp in self.triangulated_vectors:
            vn_pool_sets.append(self.vpool(*vp))
        return vn_pool_sets

    def tvs_gen(self, vn_pool_sets):
        """
        A generator to yield vpool objects for a generation of
        sub-triangulations. Preserving the original triangulated_vectors is
        needed to ensure refinement works correctly.
        :return:
        """
        for vn_pool in vn_pool_sets:
            yield vn_pool

        return

    def vc_gen(self, vn_pool_sets):
        """
        Generates the centroids from every generation of sub-triangulations
        :return:
        """
        for vp, vpool in zip(self.triangulated_vectors, vn_pool_sets):
            vn_pool_sets.append(self.vpool(*vp))

            origin, supremum = vp
            # Initiate vertices in case they don't exist
            vo = self.V[tuple(origin)]
            vs = self.V[tuple(supremum)]
            # Disconnect the origin and supremum
            vo.disconnect(vs)
            # Build centroid
            # vca = (vo.x_a + vs.x_a) / 2.0
            vca = (vs.x_a - vo.x_a) / 2.0 + vo.x_a
            vc = self.V[tuple(vca)]

            # Connect the origin and supremum to the centroid
            # vo.disconnect(vs)
            vc.connect(vo)
            vc.connect(vs)
            cvpool = copy.copy(vpool)
            for v in cvpool:
                v.connect(vc)
                #TODO: Check
                #v.connect(vo)
                #v.connect(vs)
                vn_pool = vpool - set([v])
                #for vn in vpool:
                #    v.connect(vn)



            yield vc.x

        return

    def refine_local_space_old(self, origin, supremum, vpool=None, vc_i=False):
        """
        Refines the inside the hyperrectangle captured by the vector

        #TODO: Ensure correct dimensions in input vectors

        :param origin: vector origin tuple/list
        :param supremum: vector supremum tuple/list
        :param vc_i: bool, Generate centroid if not generated in outside loop
                     (recommended to be set to True if the routine is called on
                      a new domain)
        :return:
        """
        if vpool is None:
            vn_pool = self.vpool(origin, supremum)
        else:
            vn_pool = vpool

        #%%
        vot = tuple(origin)
        vst = tuple(supremum)
        # Initiate vertices in case they don't exist
        vo = self.V[vot]
        vs = self.V[vst]
        #%%

        # Disconnect the origin and supremum
        vo.disconnect(vs)

        # Build centroid  #TODO: Done in outer loop and no longer needed
        # vca = (vo.x_a + vs.x_a) / 2.0
        vca = (vs.x_a - vo.x_a) / 2.0 + vo.x_a
        vc = self.V[tuple(vca)]

        # Connect the origin and supremum  to the centroid
        # vo.disconnect(vs)
        vc.connect(vo)
        vc.connect(vs)
        #TODO: We need to update the centroid nn set to connect to everything to
        #     ensure triangulation. Check if the following is consistent
        #     UPDATE: It was found to be inconsistent since the vertices in
        #     vpool is not connected to vc, there is potential to improve this
        #     routine by improving parrallel connections in _vertex.py
        #vc.nn.update(vpool)
        for v in vpool:
            v.connect(vc)
        yield vc.x
        # %%

        vn_done = set()
        cvn_pool = copy.copy(vn_pool)

        for vn in cvn_pool:
            # Disconnect with origin vertex
            vn.disconnect(vo)

            # Create the new vertex to connect to vo and von
            vjt = (vn.x_a - vo.x_a) / 2.0 + vo.x_a
            vj = self.V[tuple(vjt)]
            vj.connect(vo)
            vj.connect(vn)
            # Connect the vertices to the centroid (vo is already connected)
            vj.connect(vc)
            vn.connect(vc)
            yield vj.x

            # Disconnect with supremum vertex
            vn.disconnect(vs)

            # Create the new vertex to connect to vs and vn
            vkt = (vn.x_a - vs.x_a) / 2.0 + vs.x_a
            vk = self.V[tuple(vkt)]
            vk.connect(vs)
            vk.connect(vn)
            # Connect the vertices to the centroid (vo is already connected)
            vk.connect(vc)
            vn.connect(vc)
            yield vk.x

            # Add vn to vertices that have finished loop
            vn_done.add(vn)

            # Find intersecting neighbours
            vn_pool = vn_pool - set([vn])
            #print(f'vn_pool = {vn_pool}')
            for vnn in vn_pool:
                #NOTE: if vnn not vn, but might be faster to just leave out?
                #NOTE 2: Every vertex is connected with every other vertex,
                #        often the centroid is generated and connected again.
                #TODO: Ease the issue in NOTE 2 using lru_caches

                # Create the new vertex to connect to vn and vnn
                vlt = (vnn.x_a - vn.x_a) / 2.0 + vn.x_a
                vl = self.V[tuple(vlt)]
                vl.connect(vnn)
                vl.connect(vn)

                # Connect the vertices to the centroid (vn and vnn
                # is already connected)
                vl.connect(vc)
                #vnn.connect(vc)

                # Yield a tuple
                yield vl.x

                # Disconnect old neighbours
                vn.disconnect(vnn)

            # Append the newly triangulated search spaces for future refinement
            self.triangulated_vectors.append((vc.x, vn.x))

        self.triangulated_vectors.append((vc.x, vo.x))
        self.triangulated_vectors.append((vc.x, vs.x))

        # Remove the current vector from the external container
        self.triangulated_vectors.remove((origin, supremum))
        # Complete the routine in the generator
        yield vc.x
        return

    def vpool(self, origin, supremum):
        vot = tuple(origin)
        vst = tuple(supremum)
        # Initiate vertices in case they don't exist
        vo = self.V[vot]
        vs = self.V[vst]

        # Remove origin - supremum disconnect

        # Find the lower/upper bounds of the refinement hyperrectangle
        bl = list(vot)
        bu = list(vst)
        for i, (voi, vsi) in enumerate(zip(vot, vst)):
            if bl[i] > vsi:
                bl[i] = vsi
            if bu[i] < voi:
                bu[i] = voi

        # TODO: These for loops can easily be replaced by numpy operations,
        #      tests should be run to determine which method is faster.
        #      NOTE: This is mostly done with sets/lists because we aren't sure
        #            how well the numpy arrays will scale to thousands of
        #             dimensions.
        vn_pool = set()
        vn_pool.update(vo.nn)
        vn_pool.update(vs.nn)
        cvn_pool = copy.copy(vn_pool)
        for vn in cvn_pool:
            for i, xi in enumerate(vn.x):
                if bl[i] <= xi <= bu[i]:
                    pass
                else:
                    try:
                        vn_pool.remove(vn)
                    except KeyError:
                        pass  # NOTE: Not all neigbouds are in initial pool
        return vn_pool


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

    #@lru_cache(maxsize=None)  #TODO: Undo
    def generate_sub_cell_t1(self, origin, v_x):
        # TODO: Test if looping lists are faster
        return self.v_o - self.v_o * numpy.array(v_x)

    #@lru_cache(maxsize=None) #TODO: Undo
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
            #TODO: Note that scipy might be faster to add as an optional
            #      dependency
            sign_det_A_j0 = numpy.sign(numpy.linalg.det(numpy.delete(A_j0, d, 0)))
            #TODO: Note if sign_det_A_j0 == then the point is coplanar to the
            #      current simplex facet, so perhaps return True and attach?
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


    def in_st(self, v_x, v=None):
        """
        Insert a new vertex at location v_x into star domain, if a vertex `v` is
        specified then the routine skips the distance computation and is
        significantly faster.

        :param v_x:
        :param v: Vertex, object, if the v is the near
        :return:
        """
        if v is not None:
            pass
        else:
            pass
            # Distance computation

    def refine_star(self, v):
        """
        Refine the star domain of a vertex v
        :param v:
        :return:
        """
        if 1:
            # Copy lists before iteration
            vnn = copy.copy(v.nn)
            v1nn = []
            d_v0v1_set = set()
            for v1 in vnn:
                v1nn.append(copy.copy(v1.nn))

            for v1, v1nn in zip(vnn,
                                v1nn):  # TODO: What the fuck " v1nn in zip(vnn, v1nn)"  ????
                vnnu = v1nn.intersection(vnn)

                d_v0v1 = self.split_edge(v.x, v1.x)
                d_v0v1_set.add(d_v0v1)

                for v2 in vnnu:
                    d_v1v2 = self.split_edge(v1.x, v2.x)
                    d_v0v1.connect(d_v1v2)

            # Connect d_v0v1_set
            v1nn = []
            v2nn = []
            d_v0v1_list = list(d_v0v1_set)  # need ordered here
            for v2 in d_v0v1_list:
                v1nn.append(copy.copy(v2.nn))
                v2nn.append(copy.copy(v2.nn))
            for v1, v1n in zip(d_v0v1_list, v1nn):
                for v2, v2n in zip(d_v0v1_list, v2nn):
                    if v1 is not v2:
                        nset = v1n.intersection(v2n) - set([v])
                        if len(nset) > 0:
                            pass
                            v1.connect(v2)

        if 0:
            # Copy lists before iteration
            vnn = copy.copy(v.nn)
            v1nn = []
            for v1 in vnn:
                v1nn.append(copy.copy(v1.nn))

            for v1, v1nn in zip(vnn, v1nn):
                print('-')
                print(f'v1.x = {v1.x}')
                vnnu = v1nn.intersection(vnn)
                print(f'vnnu = {vnnu}')

                d_v0v1 = self.split_edge(v.x, v1.x)
                for v2 in vnnu:
                    print(f'v2 = {v2}')
                    # d_v0v1.connect(v2)
                    d_v1v2 = self.split_edge(v1.x, v2.x)
                    d_v0v1.connect(d_v1v2)
                    v.connect(d_v1v2)

        if 0:
            for v1, v1nn in zip(vnn, v1nn):
                vnnu = v1nn.intersection(vnn)
                print(f'vnnu = {vnnu}')
                print(f'v1.x = {v1.x}')
                print(f'v1.x = {v1.x}')

                comb_iter = itertools.combinations(vnnu, self.dim)
                for s in comb_iter:
                    print(f's = {s}')
                    for v2 in s:
                        v_v1 = self.split_edge(v.x, v1.x)
                        v1_v2 = self.split_edge(v1.x, v2.x)
                        v_v1.connect(v)
                        v1_v2.connect(v)
                        v_v1.connect(v1_v2)

        return

    def refine_all_star(self, exclude=set()):
        """
        Refine the star domain of all vertices
        :param v: VertexObject
        :param exclude: set, exclude subdividing any neighbours in this set
        :return:
        """
        # Vcopy = copy.copy(self.HC.V)
        V0l = list(self.V)
        V0nn = []
        V1nn = []
        for ind, v in enumerate(V0l):
            V0nn.append(copy.copy(list(v.nn)))
            V1nn.append([])
            for v1 in v.nn:
                V1nn[ind].append(copy.copy(list(v1.nn)))

        ind = -1
        for v, vnn in zip(V0l, V0nn):
            ind += 1
            d_v0v1_set = set()
            # d_v1v2_set = set()  # Only used to track new vertices
            for v1, v1nn in zip(vnn, V1nn[ind]):
                v1nn = set(v1nn)
                vnnu = v1nn.intersection(vnn)
                d_v0v1 = self.split_edge(v.x, v1.x)
                # d_v0v1_set.add(d_v0v1)
                for v2 in vnnu:
                    pass
                    d_v1v2 = self.split_edge(v1.x, v2.x)
                    d_v0v1.connect(d_v1v2)
                    # d_v1v2_set.add(d_v1v2)  # Only used to track new vertices

        # Vnew = d_v0v1_set.union(d_v1v2_set)
        # print(f'Vnew = {Vnew}')
        # print(f'len(Vnew) = {len(Vnew)}')
        return  # Vnew

    def boundary(self, V=None, pp=True):
        """
        Compute the boundary of a set of vertices

        :param V: Iterable object containing vertices, if None entire complex is
                  used.
        :return: dV, boundary set of V
        """
        if V is None:
            V = self.V
        dV = set()  # Known boundary vertices
        for v in V:
            s_it = itertools.combinations(v.nn, self.dim)
            valid_s = []
            for s in s_it:
                s_it_valid = True
                for v2 in s:
                    for v3 in s:
                        if v3 is not v2:
                            if v2 in v3.nn:
                                pass
                            else:
                                s_it_valid = False
                if s_it_valid:
                    valid_s.append(s)

            # Now parse through
            for s in valid_s:
                snn = set(s[0].nn)
                for v2 in s[1:]:
                    snn = snn.intersection(v2.nn)

                snn = snn - set(s) - set([v])
                # Check if simplex is a boundary simplex:
                if len(snn) == 0:
                    for v2 in s:
                        dV.add(v2)
        return dV

    def pproc_boundary(self, V=None, pp=True):
        """
        Compute the boundary of a set of vertices using multiprocessing.

        :param V: Iterable object containing vertices, if None entire complex is
                  used.
        :return: dV, boundary set of V
        """
        try:
            self.pool = self.V.pool
        except AttributeError:
            self.pool = mp.Pool(processes=self.workers)
        if V is None:
            V = self.V
        dV = set()  # Known boundary vertices

        vlist = V.cache.keys()
        G = self.pool.map(fdum, vlist)
        for v, _ in zip(self.V, G):
            s_it = itertools.combinations(v.nn, self.dim)
            valid_s = []
            for s in s_it:
                s_it_valid = True
                for v2 in s:
                    for v3 in s:
                        if v3 is not v2:
                            if v2 in v3.nn:
                                pass
                            else:
                                s_it_valid = False
                if s_it_valid:
                    valid_s.append(s)

            # Now parse through
            for s in valid_s:
                snn = set(s[0].nn)
                for v2 in s[1:]:
                    snn = snn.intersection(v2.nn)

                snn = snn - set(s) - set([v])
                # Check if simplex is a boundary simplex:
                if len(snn) == 0:
                    for v2 in s:
                        dV.add(v2)
        return dV

    def boundary_old_probably(self, V=None):
        """
        Compute the boundary of a set of vertices

        :param V: Iterable object containing vertices, if None entire complex is
                  used.
        :return: dV, boundary set of V
        """
        dV = set()
        if V is None:
            V = self.V

        for v in V:
            print(f'====')
            print(f'v = {v.x}')
            v_nn = copy.copy(v.nn)
            for v2 in v_nn:
                if v in v_nn:
                    pass


            if 0:
                comb_iter = itertools.combinations(v_nn, self.dim)
                for s in comb_iter:
                    print('=')
                    print(f's = {s}')
                    ss = set(s)
                    for v2 in s:
                        print(f'v2.x = {v2.x}')
                    continue
                    print('-')
                    for v2 in s:
                        print('--')
                        print(f'v2.x = {v2.x}')
                        print(f'v2.nn = {v2.nn}')
                        v2_nn = copy.copy(v2.nn - ss - set((v,)))
                        #for v3 in v2.nn:
                        for v3 in v2_nn:
                            print(f'v3.x = {v3.x}')
                        print(f's.issubset(v2.nn) = {set(s).issubset(v2.nn) }')
                        #if set(s).issubset(v2.nn):
                        if set(s).issubset(v2_nn):
                            print('!!!!!')

            #for v2 in v_nn:
            #    v2_nn = copy.copy(v.nn)
           #     comb_iter = itertools.combinations(v2_nn, self.dim)
            #    for s in comb_iter:
            #        print(f's = {s}')

                #for v3 in v2_nn:
                #    print(f'')
               #     if v3 in v_nn:
               #         print(f'v3 = {v3.x}')
               #         print(f'v3 = {v3}')
               #         print(f'v_nn = {v_nn}')
                        # v.disconnect(v3)
               #         v2.disconnect(v3)

        if 0:
            dV = set()
            if V is None:
                V = self.V

            V_nn = {}

            #TODO: Make copies of lists
            for v in V:
                print(f'v.x = {v.x}')
                V_nn[v.x] = copy.copy(v.nn)

            print(f'V_nn = {V_nn}')
            for v in V:
                print(f'-')
                print(f'v = {v.x}')
                v_nn = V_nn[v.x]#copy.copy(v.nn)
                for v2 in v_nn:
                    v2_nn = V_nn[v2.x]#copy.copy(v.nn)
                    for v3 in v2_nn:
                        print(f'')
                        if v3 in v_nn:
                            print(f'v3 = {v3.x}')
                            print(f'v3 = {v3}')
                            print(f'v_nn = {v_nn}')
                            #v.disconnect(v3)
                            v2.disconnect(v3)

            for v in V:
                if len(v.nn) == 0:
                    pass#self.V.remove(v)

    def boundary_d_old(self, V=None):
        """
        Compute the boundary self.boundary and then delete all vertices
        not in the boundary of V. Faster than boundary, but does not preserve
        pruned vertices.

        :param V:
        :return: dV, boundary set of V
        """
        dV = set()  # Known boundary vertices
        iV = set()  # Known inner vertices
        for v in V:
            v_nn = copy.copy(v.nn)
            for v2 in v_nn:
                v2_nn = copy.copy(v.nn)
                for v3 in v2_nn:
                    if v3 == v:
                        pass

    def boundary_d(self, V=None):
        """
        Compute the boundary self.boundary and then delete all vertices
        not in the boundary of V. Faster than boundary, but does not preserve
        pruned vertices.

        :param V:
        :return: dV, boundary set of V
        """
        from itertools import combinations
        dV = set()  # Known boundary vertices
        iV = set()  # Known inner vertices
        for v in V:
        #    print(f'-')
        #    print(f'v.x = {v.x}')
            s_pool = []
            s_it = combinations(v.nn, self.dim)
            valid_s = []
            for s in s_it:
                #snn = set()
                snn = set()
                #print(f's = {s}')
                s_it_valid = True
                for v2 in s:
                    #snn = snn.union(v2.nn - set([v]))
                    #snn = snn.union(v2.nn - set([v]))
                    for v3 in s:
                        if v3 is not v2:
                            if v2 in v3.nn:
                                pass
                            else:
                                s_it_valid = False
                #snn = snn.intersection(v.nn)
                #print(f'len(snn) = {len(snn)}')
                #snn = snn - set(s)
                #print(f'snn = {snn}')
                #print(f'len(snn) - s = {len(snn)}')
                #if len(snn) == 0:
                #    valid_s.append(s)
                if s_it_valid:
                        valid_s.append(s)
        #    print(f'valid_s = {valid_s}')
        #    print(f'len(valid_s) = {len(valid_s)}')

            # Now parse through
            for s in valid_s:
                snn = set(s[0].nn)
                for v2 in s[1:]:
                    snn = snn.intersection(v2.nn)

                snn = snn - set(s) - set([v])
                #print(f'snn = {snn}')
                if len(snn) == 0:
                    for v2 in s:
                        dV.add(v2)

        #print(f'dV = {dV}')
        #print(f'len(dV) = {len(dV)}')

        # Delete all vertices not in V:
        cV = copy.copy(self.V)
        #for v in dV:
       #     print(f'v in dV = {v.x}')

        for v in cV:
            #print(f'v = {v.x} in dV = {v in dV}')
            if not (v in dV):
                self.V.remove(v)
                    #for v
            #for v2 in v_nn:
            #    v2_nn = copy.copy(v.nn)
           #    for v3 in v2_nn:
            #        if v3 == v:
            #            pass

    def boundary_d2(self, V=None):
        """
        Compute the boundary self.boundary and then delete all vertices
        not in the boundary of V. Faster than boundary, but does not preserve
        pruned vertices.

        :param V:
        :return: dV, boundary set of V
        """
        from itertools import combinations
        dV = set()  # Known boundary vertices
        iV = set()  # Known inner vertices
        vlist = []
        for v in V:
            vlist.append(v)

        for v in vlist:
            #    print(f'-')
            #    print(f'v.x = {v.x}')
            s_pool = []
            s_it = combinations(v.nn, self.dim)
            valid_s = []
            for s in s_it:
                # snn = set()
                snn = set()
                # print(f's = {s}')
                s_it_valid = True
                for v2 in s:
                    # snn = snn.union(v2.nn - set([v]))
                    # snn = snn.union(v2.nn - set([v]))
                    for v3 in s:
                        if v3 is not v2:
                            if v2 in v3.nn:
                                pass
                            else:
                                s_it_valid = False
                # snn = snn.intersection(v.nn)
                # print(f'len(snn) = {len(snn)}')
                # snn = snn - set(s)
                # print(f'snn = {snn}')
                # print(f'len(snn) - s = {len(snn)}')
                # if len(snn) == 0:
                #    valid_s.append(s)
                if s_it_valid:
                    valid_s.append(s)
            #    print(f'valid_s = {valid_s}')
            #    print(f'len(valid_s) = {len(valid_s)}')

            # Now parse through
            for s in valid_s:
                snn = set(s[0].nn)
                for v2 in s[1:]:
                    snn = snn.intersection(v2.nn)

                snn = snn - set(s) - set([v])
                # print(f'snn = {snn}')
                if len(snn) == 0:
                    for v2 in s:
                        dV.add(v2)
                else:
                    for v2 in s:
                        self.V.remove(v2)
        # print(f'dV = {dV}')
        # print(f'len(dV) = {len(dV)}')

        # Delete all vertices not in V:
        cV = copy.copy(self.V)
        # for v in dV:
        #     print(f'v in dV = {v.x}')

        if 0:
            for v in cV:
                # print(f'v = {v.x} in dV = {v in dV}')
                if not (v in dV):
                    self.V.remove(v)
                    # for v
            # for v2 in v_nn:
            #    v2_nn = copy.copy(v.nn)
        #    for v3 in v2_nn:
        #        if v3 == v:
        #            pass
    # --- DEC thin wrappers (delegate to hyperct.dec) ---

    def clifford(self, dim, q=''):
        return _dec.clifford(self, dim, q)

    def sharp(self, v_x):
        return _dec.sharp(self, v_x)

    @staticmethod
    def flat(form, dim):
        return _dec.flat(form, dim)

    # --- Plotting thin wrappers (delegate to hyperct._plotting) ---

    def plot_complex(self, **kwargs):
        return _plotting.plot_complex(self, **kwargs)

    def plot_save_figure(self, strpath):
        return _plotting.plot_save_figure(self, strpath)

    def plot_clean(self, del_ax=True, del_fig=True):
        return _plotting.plot_clean(self, del_ax, del_fig)

    def plot_contour(self, bounds, func, func_args=()):
        return _plotting.plot_contour(self, bounds, func, func_args)

    def plot_complex_surface(self, fig, ax, directed=True, pointsize=5,
                             color_e=None, color_f=None, min_points=[]):
        return _plotting.plot_complex_surface(
            self, fig, ax, directed, pointsize, color_e, color_f, min_points)

    def plot_field_surface(self, fig, ax, bounds, func, func_args=(),
                           proj_dim=2, color=None):
        return _plotting.plot_field_surface(
            self, fig, ax, bounds, func, func_args, proj_dim, color)

    def plot_field_grids(self, bounds, func, func_args):
        return _plotting.plot_field_grids(self, bounds, func, func_args)

    @staticmethod
    def plot_directed_edge(f_v1, f_v2, x_v1, x_v2, mut_scale=20,
                           proj_dim=2, color=None):
        return _plotting.plot_directed_edge(
            f_v1, f_v2, x_v1, x_v2, mut_scale, proj_dim, color)

    @staticmethod
    def plot_min_points(axes, min_points, proj_dim=2, point_color=None,
                        pointsize=5):
        return _plotting.plot_min_points(
            axes, min_points, proj_dim, point_color, pointsize)

    def animate_complex(self, update_state, **kwargs):
        return _plotting.animate_complex(self, update_state, **kwargs)

    # %% Conversions
    def incidence_array(self):
        """
        Construct v-v incidence array
        :return: self.incidence_structure
        """

        self.incidence_structure = numpy.zeros([self.V.size(), self.V.size()], dtype=int)

        for v in self.V:
            for v2 in v.nn:
                self.incidence_structure[v.index, v2.index] = 1

        return self.incidence_structure

    def incidence_array_directed(self):
        """
        Construct v-v incidence array
        Complex must be supplied with sfield and directed
        :return: self.incidence_structure
        """

        self.incidence_structure = numpy.zeros([self.V.size(), self.V.size()], dtype=int)

        for v in self.V:
            for v2 in v.nn:
                if v.f <= v2.f:
                    self.incidence_structure[v.index, v2.index] = 1
                else:
                    self.incidence_structure[v.index, v2.index] = -1

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


    def vf_to_vv(self, vertices, simplices):
        """
        Convert a vertex-face mesh to a vertex-vertex mesh used by this class

        :param vertices: list of vertices
        :param simplices: list of simplices
        :return:
        """
        if self.dim > 1:
            for s in simplices:
                edges = itertools.combinations(s, self.dim)
                for e in edges:
                    self.V[tuple(vertices[e[0]])].connect(
                        self.V[tuple(vertices[e[1]])])
        else:
            for e in simplices:
                self.V[tuple(vertices[e[0]])].connect(
                    self.V[tuple(vertices[e[1]])])
                #print(f'e = {e}')
            #for v in s:
                #print(f'v = {v}')
                #print(f's = {s}')
                #print(f'vertices[v] = {vertices[v]}')
                #print(f'tuple(vertices[v])= {tuple(vertices[v])}')
                #self.V[tuple(vertices[v])]
        return

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
        #TODO: UNTESindexTED FOR DIMENSIONS HIGHER THAN 2
        self.vertices_fm = []  # Vertices (A list of ob
        self.simplices_fm = []  # Faces
        self.simplices_fm_i = []

        # TODO: Add in field
        for v in self.V.cache:  # Note that cache is an OrderedDict
            self.vertices_fm.append(v)

            #TODO: Should this not be outside the initial loop?
            simplex = (self.dim + 1) * [None]  # Predetermined simplex sizes
            simplex[0] = self.V[v]

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
                                    build_simpl_i[2] = v3.index #TODO: Check missing index 1?
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

            # Append the newly built simplex
            self.simplices_fm.append(sl)

        return

    #TODO: face_vertex_mesh

    # %% Data persistence
    def save_complex(self, HC=None, fn='cdata.json'):
        """
        TODO: Save the complex to file using pickle
        https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
        :param fn: str, filename
        :return:
        """
        if HC == None:
            HC = self

        data = collections.OrderedDict()
        HCckey = copy.copy(HC.V.cache)
        print(f'HC.V.cache.keys() = {HC.V.cache.keys()}')
        for v in HCckey:
            print('.')
            print(f'v = {v}')
            print(f'type(v) = {type(v)}')
            data[str(v)] = {}
            print(f'HC.V[v].nn = {HC.V[v].nn}')
            nn_keys = []
            for v2 in HC.V[v].nn:
                nn_keys.append(v2.x)

            print(f'{nn_keys}')
            data[str(v)]['nn'] = nn_keys
            #print(f'vars(HC.V[str(v)]) = {vars(HC.V[str(v)])}')
            print(f'vars(HC.V[v]) = {vars(HC.V[v])}')
            for key in vars(HC.V[v]):
                if key not in ['nn', 'x', 'x_a', 'hash']:
                    #print(f'key = {key}')
                    print(f'val = {vars(HC.V[v])[key]}')
                    data[str(v)][key] = vars(HC.V[v])[key]
                    if key == 'f':
                        data[str(v)][key] = float(vars(HC.V[v])[key])
                    if key == 'index':
                        data[str(v)][key] = int(vars(HC.V[v])[key])
                        #data[str(v)][key] = float(vars(HC.V[v])[key])

        print(f'data = {data}')
        with open(fn, 'w') as fp:
            json.dump(data, fp)

        return

    def load_complex(self, fn='cdata.json'):
        """
        TODO: Load the complex from file using pickle
        :param fn: str, filename
        :return:
        """
        logging.info('Loading complex data...')
        with open(fn, 'r') as fp:
            data = json.load(fp, object_pairs_hook=collections.OrderedDict)
       # print(f'data = {data}')
        logging.info('Complex data successfully loaded, building Python '
                    'objects...')
        for vt in data.keys():
            v = self.V[eval(vt)]
            for vnt in data[vt]['nn']:
                v.connect(self.V[tuple(vnt)])

            for key, value in data[vt].items():
                if key not in ['nn']:
                    setattr(v, key, value)
                #print(f'val = {vars(data[v])[key]}')
                #data[str(v)][key] = vars(data[v])[key]

        logging.info("Complex cache constructions completed, purging pools...")
        try:
            self.V.recompute_pools()
        except AttributeError:  # Non-field complexes have no pools
            pass
        logging.info("Succesfully loaded complex to memory.")
        #TODO: Clear f and g pools in HC.V

