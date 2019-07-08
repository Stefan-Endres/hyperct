from hyperct._complex import *

def g_cons(x):  # (Requires n > 2)
    import numpy
    # return x[0] - 0.5 * x[2] + 0.5
    #time.sleep(1)
    return x[0]  # + x[2] #+ 0.5


def func(x):
    #time.sleep(1)
    #if x[0] == 0:
    #    raise FloatingPointError
    return (-(x[1] + 47.0)
            * numpy.sin(numpy.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0))))
            - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0))))
            )#/x[0]

n = 2
gen = 2 # 7
bounds = [(-100.0, 100.0), (-100.0, 100.0)]


def test_triangulation(n, gen, bound=None):
    if bounds == None:
        HC_ref = Complex(n)
    HC_ref.load_complex(fn='../test_2_2_3D_cube_splits.json')



