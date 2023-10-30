"""
Automatic unit testing of check_grid_specification()
"""

import unittest

import numpy as np

from milhoja import check_grid_specification


class TestCheckGridSpecification(unittest.TestCase):
    def setUp(self):
        self.__good = {
            "dimension": 3,
            "nxb": 8,
            "nyb": 16,
            "nzb": 4,
            "nguardcells": 1
        }

        # Confirm base spec is correct
        check_grid_specification(self.__good)

    def testKeys(self):
        # Too few
        bad_spec = self.__good.copy()
        del bad_spec["nxb"]
        with self.assertRaises(ValueError):
            check_grid_specification(bad_spec)

        # Too many
        bad_spec = self.__good.copy()
        bad_spec["fail"] = 1.1
        with self.assertRaises(ValueError):
            check_grid_specification(bad_spec)

        # Right number, but incorrect key name
        bad_spec = self.__good.copy()
        del bad_spec["nxb"]
        bad_spec["fail"] = 1.1
        with self.assertRaises(ValueError):
            check_grid_specification(bad_spec)

    def testDimension(self):
        ok_spec = self.__good.copy()
        # These are correct for all dimensions
        ok_spec["nyb"] = 1
        ok_spec["nzb"] = 1
        for good in [1, 2, 3]:
            ok_spec["dimension"] = good
            check_grid_specification(ok_spec)

        bad_spec = self.__good.copy()
        for bad in [None, "fail", np.nan, np.inf, [], [1]]:
            bad_spec["dimension"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec)
        for bad in [-1, 0, 4]:
            bad_spec["dimension"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec)

    def testNxb(self):
        ok_spec = self.__good.copy()
        # These are correct for all dimensions
        ok_spec["nyb"] = 1
        ok_spec["nzb"] = 1
        for dim in [1, 2, 3]:
            ok_spec["dimension"] = dim
            for good in [1, 2, 3, 10, 101, 222]:
                ok_spec["nxb"] = good
                check_grid_specification(ok_spec)

        bad_spec = self.__good.copy()
        for bad in [None, "fail", np.nan, np.inf, [], [1]]:
            bad_spec["nxb"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec)
        for bad in [-1, 0]:
            bad_spec["nxb"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec)

    def testNyb(self):
        ok_spec = self.__good.copy()
        # This is correct for all dimensions
        ok_spec["nzb"] = 1
        ok_spec["dimension"] = 1
        ok_spec["nyb"] = 1
        check_grid_specification(ok_spec)
        for dim in [2, 3]:
            ok_spec["dimension"] = dim
            for good in [1, 2, 3, 10, 101, 222]:
                ok_spec["nyb"] = good
                check_grid_specification(ok_spec)

        bad_spec = self.__good.copy()
        for bad in [None, "fail", np.nan, np.inf, [], [1]]:
            bad_spec["nyb"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec)
        for bad in [-1, 0]:
            bad_spec["nyb"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec)

        bad_spec = self.__good.copy()
        bad_spec["dimension"] = 1
        for bad in [2, 3, 10, 101, 222]:
            bad_spec["nyb"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec)

    def testNzb(self):
        ok_spec = self.__good.copy()
        # This is correct for all dimensions
        ok_spec["nyb"] = 1
        for dim in [1, 2]:
            ok_spec["nzb"] = 1
            check_grid_specification(ok_spec)
        ok_spec["dimension"] = 3
        for good in [1, 2, 3, 10, 101, 222]:
            ok_spec["nzb"] = good
            check_grid_specification(ok_spec)

        bad_spec = self.__good.copy()
        for bad in [None, "fail", np.nan, np.inf, [], [1]]:
            bad_spec["nzb"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec)
        for bad in [-1, 0]:
            bad_spec["nzb"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec)

        bad_spec = self.__good.copy()
        bad_spec["nyb"] = 1
        for dim in [1, 2]:
            bad_spec["dimension"] = dim
            for bad in [2, 3, 10, 101, 222]:
                bad_spec["nzb"] = bad
                with self.assertRaises(ValueError):
                    check_grid_specification(bad_spec)

    def testNGuardcells(self):
        ok_spec = self.__good.copy()
        for good in [0, 1, 2, 3, 10, 101, 222]:
            ok_spec["nguardcells"] = good
            check_grid_specification(ok_spec)

        bad_spec = self.__good.copy()
        for bad in [None, "fail", np.nan, np.inf, [], [1]]:
            bad_spec["nguardcells"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec)
        for bad in [-2, -1]:
            bad_spec["nguardcells"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec)
