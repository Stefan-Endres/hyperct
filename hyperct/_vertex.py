from abc import ABC, abstractmethod

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
    def __init__(self, x, nn=None):
        super(VertexCube, self).__init__(x, nn=nn)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)

if __name__ == '__main__':
    v1 = VertexCube((1,2,-3.3))
    v1 = VertexField((1,2,-3.3))
    print(v1)
    print(v1.x)

    Vertex = VertexCube

    v1 = Vertex((1, 2, 3))
    v1 = Vertex((1, 2, 3))
    print(v1)
    #print(v1.x_a)