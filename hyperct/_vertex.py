import numpy
from abc import ABC, abstractmethod


"""Vertex objects"""
class VertexBase(ABC):
    def __init__(self, x, nn=None, index=None):
        self.x = x
        #self.order = sum(x)  #TODO: Delete if we can't prove the order triangulation conjecture

        if nn is not None:
            self.nn = nn
        else:
            self.nn = set()

        self.index = index

    def __hash__(self):
        return hash(self.x)

    @abstractmethod
    def connect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    @abstractmethod
    def disconnect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    def print_out(self):
        print("Vertex: {}".format(self.x))
        constr = 'Connections: '
        for vc in self.nn:
            constr += '{} '.format(vc.x)

        print(constr)
        #print('Order = {}'.format(self.order))


class VertexCube(VertexBase):
    """Vertex class to be used for a pure simplicial complex with no associated
    differential geometry (single level domain that exists in R^n)"""
    def __init__(self, x, nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)


class VertexScalarField(VertexBase):
    """Add homology properties of a scalar field f: R^n --> R associated with
    the geometry built from the VertexBase class"""

    def __init__(self, x, field=None, nn=None, index=None, field_args=(),
                 g_cons=None, g_cons_args=()):
        """
        :param x: tuple, vector of vertex coordinates
        :param field: function, a scalar field f: R^n --> R associated with
    the geometry
        :param nn: list, optional, list of nearest neighbours
        :param index: int, optional, index of the vertex
        :param field_args: tuple, additional arguments to be passed to field
        :param g_cons: function, constraints on the vertex
        :param g_cons_args: tuple, additional arguments to be passed to g_cons

        """
        super().__init__(x, nn=nn, index=index)

        self.x_a = numpy.array(x)  # Array version of the hashed tuple

        # Note Vertex is only initiated once for all x so only
        # evaluated once
        self.feasible = True

        print(g_cons)
        if g_cons is not None:
            for g, args in zip(g_cons, g_cons_args):
                if g(self.x_a, *args) < 0.0:
                    self.f = numpy.inf
                    self.feasible = False
                    break

        #TODO: add possible h_cons tolerance check

        if self.feasible:
            try: #TODO: Remove exception handling?
                self.f = field(self.x_a, *field_args)
            except TypeError:
                print(f"WARNING: field function not found at x = {self.x_a}")
                self.f = numpy.inf

        self.fval = None
        self.check_min = True

    def __hash__(self):
        return hash(self.x)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

            if self.minimiser():
                v._min = False
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
        """Check whether this vertex is strictly less than all its neighbours"""
        if self.check_min:
            self._min = all(self.f < v.f for v in self.nn)
            self.check_min = False

        return self._min

class VertexVectorField(VertexBase):
    """Add homology properties of a scalar field f: R^n --> R^m associated with
    the geometry built from the VertexBase class"""

    def __init__(self, x, sfield=None, vfield=None, field_args=(),
                 vfield_args=(), g_cons=None,
                 g_cons_args=(), nn=None, index=None):
        super(VertexVectorField, self).__init__(x, nn=nn, index=index)

        raise NotImplementedError("This class is still a work in progress")

"""
Cache objects
"""
class VertexCacheBase(object):
    def __init__(self):

        self.cache = {}
        self.nfev = 0  # Feasible points
        self.size = 0  # Total size of cache
        self.index = -1

        #TODO: Define a getitem method based on if indexing is on or not so
        # that we do not have to do an if check every call (does the python
        # compiler make this irrelevant or not?) and in addition whether or not
        # we have defined a field function.

class VertexCacheIndex(VertexCacheBase):
    def __init__(self):
        super().__init__()
        self.Vertex = VertexCube

    def __getitem__(self, x):  #TODO: Check if no_index is significant speedup
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, index=self.index)
            # logging.info("New generated vertex at x = {}".format(x))
            # NOTE: Surprisingly high performance increase if logging is commented out
            self.cache[x] = xval
            return self.cache[x]

class VertexCacheField(VertexCacheBase):
    def __init__(self, field, field_args=(), g_cons=None,
                 g_cons_args=()):
        super().__init__()
        self.index = -1
        self.Vertex = VertexScalarField
        self.field = field
        self.field_args = field_args
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def __getitem__(self, x, nn=None): #TODO: Test to add optional nn argument?
        #NOTE: To use nn arg do ex. V.__getitem__((1,2,3), [3,4,7])
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, field=self.field, nn=nn, index=self.index,
                               field_args=self.field_args,
                               g_cons=self.g_cons, g_cons_args=self.g_cons_args)

            self.cache[x] = xval

        # TODO: Check
        if self.field is not None:
            if self.g_cons is not None:
                if xval.feasible:
                    self.nfev += 1
                    self.size += 1
                else:
                    self.size += 1
            else:
                self.nfev += 1
                self.size += 1

        return self.cache[x]

if __name__ == '__main__':  # TODO: Convert these to unittests
    v1 = VertexCube((1,2,-3.3))

    print(v1)
    print(v1.x)

    Vertex = VertexCube

    v1 = Vertex((1, 2, 3))
    v1 = Vertex((1, 2, 3))
    print(v1)
    #print(v1.x_a)

    def func(x):
        import numpy
        return numpy.sum((x - 3) ** 2) + 2.0 * (x[0] + 10)


    def g_cons(x):  # (Requires n > 2)
        import numpy
        # return x[0] - 0.5 * x[2] + 0.5
        return x[0]  # + x[2] #+ 0.5


    v1 = VertexScalarField((1, 2, -3.3), func)
    print(v1)
    print(v1.x)
    print(v1.x_a)

    def func(x):
        import numpy
        return numpy.sum((x - 3) ** 2) + 2.0 * (x[0] + 10)


    def g_cons(x):  # (Requires n > 2)
        import numpy
        # return x[0] - 0.5 * x[2] + 0.5
        return x[0]  # + x[2] #+ 0.5

    #V = VertexCache()
    V = VertexCacheField(func)
    print(V)
    V[(1,2,3)]
    V[(1,2,3)]
    V.__getitem__((1,2,3), None)
    V.__getitem__((1,2,3), [3,4,7])
    #TODO: ADD THIS TO COMPLEX:
