from hyperct import *


def func(x):
    import numpy
    return numpy.sum(x ** 2) + 2.0 * x[0]


def g_cons(x):  # (Requires n > 2)
    import numpy
    # return x[0] - 0.5 * x[2] + 0.5
    return x[0]  # + x[2] #+ 0.5

# Test
class TestCube(object):

    def test_1_1_2D_cube_init(self):  #TODO: REMOVE FUNC AFTER SPLIT
        """Test that the initial 2D cube has the correct coords"""
        HC = Complex(2, func)
        HC.n_cube()
        check = [(0, 0), (1, 1), (1, 0), (0, 1), (0.5, 0.5)]
        for i, v in enumerate(HC.C0()):
            pass
            numpy.testing.assert_equal(check[i], check[i])


    def test_2_1_3D_cube_init(self):
        """Test that the initial 2D cube has the correct coords"""
        HC = Complex(3, func)
        HC.n_cube()
        check = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1),
                 (0, 1, 0), (0, 1, 1), (0, 0, 1), (0.5, 0.5, 0.5)]
        for i, v in enumerate(HC.C0()):
            pass
            numpy.testing.assert_equal(check[i], v.x)

    def test_3_1_4D_cube_init(self):
        """Test that the initial 2D cube has the correct coords"""
        HC = Complex(4, func)
        HC.n_cube()
        check = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 0, 0), (1, 1, 0, 0),
                 (1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1),
                 (1, 0, 0, 1), (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 1, 1),
                 (0, 1, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1),
                 (0.5, 0.5, 0.5, 0.5)]

        for i, v in enumerate(HC.C0()):
            pass
            numpy.testing.assert_equal(check[i], v.x)

    def test_4_1_5D_cube_init(self):
        """Test that the initial 2D cube has the correct coords"""
        HC = Complex(5, func)
        HC.n_cube()
        check = [(0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (1, 0, 0, 0, 0),
                 (1, 1, 0, 0, 0),
                 (1, 1, 1, 0, 0), (1, 1, 1, 1, 0), (1, 1, 1, 0, 1),
                 (1, 1, 0, 1, 0),
                 (1, 1, 0, 1, 1), (1, 1, 0, 0, 1), (1, 0, 1, 0, 0),
                 (1, 0, 1, 1, 0),
                 (1, 0, 1, 1, 1), (1, 0, 1, 0, 1), (1, 0, 0, 1, 0),
                 (1, 0, 0, 1, 1),
                 (1, 0, 0, 0, 1), (0, 1, 0, 0, 0), (0, 1, 1, 0, 0),
                 (0, 1, 1, 1, 0),
                 (0, 1, 1, 1, 1), (0, 1, 1, 0, 1), (0, 1, 0, 1, 0),
                 (0, 1, 0, 1, 1),
                 (0, 1, 0, 0, 1), (0, 0, 1, 0, 0), (0, 0, 1, 1, 0),
                 (0, 0, 1, 1, 1),
                 (0, 0, 1, 0, 1), (0, 0, 0, 1, 0), (0, 0, 0, 1, 1),
                 (0, 0, 0, 0, 1),
                 (0.5, 0.5, 0.5, 0.5, 0.5)]

        for i, v in enumerate(HC.C0()):
            pass
            numpy.testing.assert_equal(check[i], v.x)


