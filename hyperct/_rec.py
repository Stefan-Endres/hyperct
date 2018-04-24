"""Hyperrectangle triangulations"""
# Import common functions
from hyperct._triangulation import *

# Import the base cube function
from hyperct._cube import *


# Other imports
import numpy
import logging
import sys
import copy

class VertexRec:
    def __init__(self, x, bounds=None, func=None, func_args=(), g_cons=None,
                 g_cons_args=(), nn=None, Ind=None):
        """
        Hyperrectangle vertices

        :param x:
        :param bounds:
        :param func:
        :param func_args:
        :param g_cons:
        :param g_cons_args:
        :param nn:
        :param Ind:
        """
        import numpy
        self.x = x
        self.order = sum(x)
        if bounds is None:
            x_a = numpy.array(x, dtype=float)
        else:
            x_a = numpy.array(x, dtype=float)
            for i in range(len(bounds)):
                x_a[i] = (x_a[i] * (bounds[i][1] - bounds[i][0])
                          + bounds[i][0])

                # print(f'x = {x}; x_a = {x_a}')
        # TODO: Make saving the array structure optional
        self.x_a = x_a
        print(f"self.x_a = {self.x_a}")
        # Note Vertex is only initiate once for all x so only
        # evaluated once
        if func is not None:
            if g_cons is not None:
                self.feasible = True
                for ind, g in enumerate(g_cons):
                    if g(self.x_a, *g_cons_args[ind]) < 0.0:
                        self.f = numpy.inf
                        self.feasible = False
                if self.feasible:
                    self.f = func(x_a, *func_args)

            else:
                self.f = func(x_a, *func_args)

        if nn is not None:
            self.nn = nn
        else:
            self.nn = set()

        self.fval = None
        self.check_min = True

        # Index:
        if Ind is not None:
            self.Ind = Ind

    def __hash__(self):
        # return hash(tuple(self.x))
        return hash(self.x)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

            # self.min = self.minimiser()
            if self.minimiser():
                # if self.f > v.f:
                #    self.min = False
                # else:
                v.min = False
                v.check_min = False

            # TEMPORARY
            self.check_min = True
            v.check_min = True

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)
            self.check_min = True
            v.check_min = True

    def minimiser(self):
        # NOTE: This works pretty well, never call self.min,
        #       call this function instead
        if self.check_min:
            # Check if the current vertex is a minimiser
            # self.min = all(self.f <= v.f for v in self.nn)
            self.min = True
            for v in self.nn:
                # if self.f <= v.f:
                # if self.f > v.f: #TODO: LAST STABLE
                if self.f >= v.f:  # TODO: AttributeError: 'Vertex' object has no attribute 'f'
                    # if self.f >= v.f:
                    self.min = False
                    break

            self.check_min = False

        return self.min


class VertexCacheRec:
    def __init__(self, func=None, func_args=(), bounds=None, g_cons=None,
                 g_cons_args=(), indexed=True):

        self.cache = {}
        # self.cache = set()
        self.func = func
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.func_args = func_args
        self.bounds = bounds
        self.nfev = 0
        self.size = 0
        print(f"BOUNDS = {bounds}")

        if indexed:
            self.Index = -1

    def __getitem__(self, x, indexed=True):
        try:
            return self.cache[x]
        except KeyError:
            if indexed:
                self.Index += 1
                xval = VertexRec(x, bounds=self.bounds,
                              func=self.func, func_args=self.func_args,
                              g_cons=self.g_cons,
                              g_cons_args=self.g_cons_args,
                              Ind=self.Index)
            else:
                xval = VertexRec(x, bounds=self.bounds,
                              func=self.func, func_args=self.func_args,
                              g_cons=self.g_cons,
                              g_cons_args=self.g_cons_args)

            # logging.info("New generated vertex at x = {}".format(x))
            # NOTE: Surprisingly high performance increase if logging is commented out
            self.cache[x] = xval

            # TODO: Check
            if self.func is not None:
                if self.g_cons is not None:
                    # print(f'xval.feasible = {xval.feasible}')
                    if xval.feasible:
                        self.nfev += 1
                        self.size += 1
                    else:
                        self.size += 1
                else:
                    self.nfev += 1
                    self.size += 1

            return self.cache[x]

