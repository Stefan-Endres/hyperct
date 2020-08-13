"""
Unittests


Note to set the following logging level to see console output:

logging.getLogger().setLevel(logging.INFO)
"""

import os
import logging
import unittest

from hyperct._complex import *

import pytest

logging.getLogger().setLevel(logging.INFO) #TODO: REMOVE

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


def test_triangulation(n=2, gen=0, bounds=None, symmetry=None):
    # Generate new reference complex
    HC_ref = Complex(n, domain=bounds)

    # Load test data
    if symmetry is None:
        path = os.path.join(os.path.dirname(__file__), 'test_data',
                            f'test_{n + 1}_{gen + 1}_{n}D_cube_gen_{gen}.json')
    else:
        path = os.path.join(os.path.dirname(__file__), 'test_data',
                            f'test_{n + 1}_{gen + 1}_{n}D_symm_gen_{gen}.json')

    HC_ref.load_complex(fn=path)

    # Generate data containers for test comparisons
    check = []  # Check if vertex in complex list
    nn_checks = {}  # Check if vertex has correct neighbours dict
    for v in HC_ref.V:
        check.append(v.x)
        nn_checks[v.x] = [vnn.x for vnn in v.nn]

    # Generate new test complex
    HC = Complex(n, domain=bounds, symmetry=symmetry)
    HC.triangulate()
    for i in range(gen):
        HC.refine_all()

    # Test that all the correct vertices are present
    for i, v in enumerate(HC.V.cache):
        logging.info(f'Test if generated v.x = {v} is in reference complex')
        numpy.testing.assert_equal(v in check, True)
        logging.info(f'Test passed')

    for i, v in enumerate(check):
        # Unordered check 2:
        logging.info(f'Test if reference v.x = {v} is in generated complex')
        numpy.testing.assert_equal(v in HC.V.cache, True)
        logging.info(f'Test passed')

    for v in nn_checks:
        nn_test = []
        for v2 in HC.V[v].nn:
            nn_test.append(v2.x)

        nn_t = numpy.array(nn_test)[numpy.lexsort(numpy.rot90(nn_test))]
        nn_c = numpy.array(nn_checks[v])[numpy.lexsort(numpy.rot90(nn_checks[v]))]
        logging.info('-' * len(f'Testing neighbours of {v}'))
        logging.info(f'Testing neighbours of {v}')
        logging.info('-' * len(f'Testing neighbours of {v}'))
        logging.info(f'Lexicographical arrays should match:')
        logging.info(f'Nearest neighbours generated by current test = {nn_t}')
        logging.info(f'Reference neighbours (correct values) = {nn_c}')

        try:
            numpy.testing.assert_equal(nn_t, nn_c)
        except AssertionError as e:
            logging.info(f'Test failed, searching for defects...')
            cset = set()
            gsv = numpy.array([-numpy.inf,]*n)
            osv = numpy.array([numpy.inf,]*n)
            for v in nn_c:
                cset.add(tuple(v))
                for i, vi in enumerate(v):
                    if vi > gsv[i]:
                        gsv[i] = vi
                    if vi < osv[i]:
                        osv[i] = vi
            logging.info(f' Approximate triangulation reference vectors:'
                  f' origin = {osv}'
                  f' supremum = {gsv}')
            tset = set()
            gsv = numpy.array([-numpy.inf,]*n)  # Rest vectors to find out area
            osv = numpy.array([numpy.inf,]*n)
            for v in nn_t:
                tset.add(tuple(v))
                for i, vi in enumerate(v):
                    if vi > gsv[i]:
                        gsv[i] = vi
                    if vi < osv[i]:
                        osv[i] = vi
                if tuple(v) not in cset:
                    logging.info(f'{v} should not be in v.nn')

            logging.info(f' Approximate triangulation vectors computed in test:'
                  f' origin = {osv}'
                  f' supremum = {gsv}')
            for v in nn_c:
                if tuple(v) not in tset:
                    logging.info(f'{v} missing from v.nn')

            raise(e)

        logging.info(f'Test passed')
        logging.info('-' * len(f'Test passed'))
        logging.info('.')


class TestCube(object):
    def test_1_1_2D_cube_init(self):
        """Test that the initial 2D cube has the correct vertices"""
        test_triangulation(2, 0)

    def test_1_2_2D_cube_splits(self):
        """Test that the 2D cube subtriangulations has the correct vertices,
           testing 1 generation of subtriangulations"""
        test_triangulation(2, 1)

    def test_1_3_2D_cube_splits(self):
        """Test that the 2D cube subtriangulations has the correct vertices,
           testing 2 generations of subtriangulations"""
        test_triangulation(2, 2)

    def test_2_1_3D_cube_init(self):
        """Test that the initial 3D cube has the correct vertices"""
        test_triangulation(3, 0)

    def test_2_2_3D_cube_splits(self):
        """Test that the 3D cube subtriangulations has the correct vertices,
           testing 1 generation of subtriangulations"""
        test_triangulation(3, 1)

    def test_2_3_3D_cube_splits(self):
        """Test that the 3D cube subtriangulations has the correct vertices,
           testing 2 generations of subtriangulations"""
        test_triangulation(3, 2)

    def test_3_1_4D_cube_init(self):
        """Test that the initial 4D cube has the correct vertices"""
        test_triangulation(4, 0)

    def test_3_2_4D_cube_splits(self):
        """Test that the 4D cube subtriangulations has the correct vertices,
           testing 1 generation of subtriangulations"""
        test_triangulation(4, 1)

    #@unittest.skip("Skipping slow test")
    #@pytest.mark.slow
    def test_3_3_4D_cube_splits(self):
        """Test that the 4D cube subtriangulations has the correct vertices,
           testing 2 generations of subtriangulations"""
        test_triangulation(4, 2)

    def test_4_1_5D_cube_init(self):
        """Test that the initial 5D cube has the correct vertices"""
        test_triangulation(5, 0)

    @pytest.mark.slow
    def test_4_2_5D_cube_splits(self):
        """Test that the 5D cube subtriangulations has the correct vertices,
           testing 1 generation of subtriangulations"""
        test_triangulation(5, 1)

    @pytest.mark.slow
    @unittest.skip("Skipping slow test")
    def test_4_3_5D_cube_splits(self):
        """Test that the 5D cube subtriangulations has the correct vertices,
           testing 2 generations of subtriangulations"""
        test_triangulation(5, 2)


    @unittest.skip("Skipping slow test")
    def test_5_1_6D_cube_init(self):
        "Test that the initial 6D cube has the correct vertices"
        test_triangulation(6, 0)

    @unittest.skip("Skipping slow test")
    def test_5_2_6D_cube_splits(self):
        """Test that the 6D cube subtriangulations has the correct vertices,
           testing 1 generation of subtriangulations"""
        test_triangulation(6, 1)

    @unittest.skip("Skipping slow test")
    def test_6_1_7D_cube_init(self):
        "Test that the initial 7D cube has the correct vertices"
        test_triangulation(7, 0)

    @unittest.skip("Skipping slow test")
    def test_6_2_7D_cube_splits(self):
        """Test that the 7D cube subtriangulations has the correct vertices,
           testing 1 generation of subtriangulations"""
        test_triangulation(7, 1)

    @unittest.skip("Skipping slow test")
    def test_7_1_8D_cube_init(self):
        "Test that the initial 8D cube has the correct vertices"
        test_triangulation(8, 0)

    @unittest.skip("Skipping slow test")
    def test_8_1_9D_cube_init(self):
        "Test that the initial 9D cube has the correct vertices"
        test_triangulation(9, 0)

    @unittest.skip("Skipping slow test")
    def test_9_1_10D_cube_init(self):
        "Test that the initial 10D cube has the correct vertices"
        test_triangulation(10, 0)

    @unittest.skip("Skipping slow test")
    def test_99_1_11D_cube_init(self):
        "Test that the initial 11D cube has the correct vertices"
        test_triangulation(11, 0)

class TestSymmetry(object):
    def test_1_1_2D_symm_init(self):
        """Test that the initial 2D symmetric cube has the correct vertices"""
        symmetry = [0,]*2
        test_triangulation(2, 0, symmetry=symmetry)
