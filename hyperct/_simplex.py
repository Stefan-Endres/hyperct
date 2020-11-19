import collections
from abc import ABC, abstractmethod
import logging
import copy
import numpy as np
from functools import partial
import multiprocessing as mp

#from hyperct._field import *

"""Simplex objects"""
class SimplexBase(ABC):
    def __init__(self, V):
        self.dim = len(V)
        self.V = V  # set of vertices

        # Compute unordered hash
        hkey = []
        for v in V:
            hkey.append(v.x)

        hkey.sort()
        #hkey.sort(key = len)
        self.hash = hkey


        if 0:
            self.x = x
            self.hash = hash(self.x)  # Save precomputed hash
            #self.orderv = sum(x)  #TODO: Delete if we can't prove the order triangulation conjecture

            if nn is not None:
                self.nn = set(nn)  # can use .indexupdate to add a new list
            else:
                self.nn = set()

            self.index = index

    def __hash__(self):
        return self.hash

    #@abstractmethod
    #def connect(self, s):
    #    raise NotImplementedError("This method is only implemented with an "
    #                              "associated child of the base class.")

    #@abstractmethod
    #def disconnect(self, s):
   #     raise NotImplementedError("This method is only implemented with an "
    #                              "associated child of the base class.")

    def print_out(self):
        print("Vertex: {}".format(self.x))
        constr = 'Connections: '
        for vc in self.nn:
            constr += '{} '.format(vc.x)

        print(constr)
        #print('Order = {}'.format(self.order))

    def star(self):
        """
        Returns the star domain st(v) of the vertex.

        :param v: The vertex v in st(v)
        :return: st, a set containing all the vertices in st(v)
        """
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

        #self.st = self.nn
        #self.st.add(self)
        #return self.st

class SimplexOrdered(SimplexBase):
    def __init__(self, V):
        super().__init__(V)
        pass
