"""Add homology properties of a scalar field associated with the geometry built
from the classes in _triangulation.py """
import numpy
from hyperct._vertex import VertexBase

class VertexField(VertexBase):
    def __init__(self, x, field=None, field_args=(), g_cons=None,
                 g_cons_args=(), nn=None, index=None):
        super(VertexBase, self).__init__(x, nn=nn)

        self.x_a = numpy.array(x)  # Array version of the hashed tuple

        # Note Vertex is only initiated once for all x so only
        # evaluated once
        if field is not None:
            self.feasible = True
            if g_cons is not None:
                for g, args in zip(g_cons, g_cons_args):
                    if g(self.x_a, *args) < 0.0:
                        self.f = numpy.inf
                        self.feasible = False
                        break
            if self.feasible:
                self.f = field(self.x_a, *field_args)

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

