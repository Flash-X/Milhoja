"""
Automatic unit testing of check_grid_specification()
"""

import copy
import unittest

from milhoja import (
    LOG_LEVEL_NONE, BasicLogger,
    check_grid_specification
)
from milhoja.tests import (
    NOT_INT_LIST, NOT_CLASS_LIST
)


class TestCheckGridSpecification(unittest.TestCase):
    """
    .. todo::
        * Adjust so that this checks check_grid_specification()
          indirectly by ensuring that SubroutineGroup correctly detects
          input failures.
    """
    def setUp(self):
        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        self.__good = {
            "dimension": 3,
            "nxb": 8,
            "nyb": 16,
            "nzb": 4,
            "nguardcells": 1
        }

        # Confirm base spec is correct
        check_grid_specification(self.__good, self.__logger)

    def testBadLogger(self):
        for bad in NOT_CLASS_LIST:
            with self.assertRaises(TypeError):
                check_grid_specification(self.__good, bad)

    def testKeys(self):
        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["nxb"]
        with self.assertRaises(ValueError):
            check_grid_specification(bad_spec, self.__logger)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(self.__good) < len(bad_spec))
        with self.assertRaises(ValueError):
            check_grid_specification(bad_spec, self.__logger)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["nxb"]
        bad_spec["fail"] = 1.1
        self.assertEqual(len(self.__good), len(bad_spec))
        with self.assertRaises(ValueError):
            check_grid_specification(bad_spec, self.__logger)

    def testDimension(self):
        good_spec = copy.deepcopy(self.__good)
        # These are correct for all dimensions
        good_spec["nyb"] = 1
        good_spec["nzb"] = 1
        for good in [1, 2, 3]:
            good_spec["dimension"] = good
            check_grid_specification(good_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in NOT_INT_LIST:
            bad_spec["dimension"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec, self.__logger)
        for bad in [-1, 0, 4]:
            bad_spec["dimension"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec, self.__logger)

    def testNxb(self):
        good_spec = copy.deepcopy(self.__good)
        # These are correct for all dimensions
        good_spec["nyb"] = 1
        good_spec["nzb"] = 1
        for dim in [1, 2, 3]:
            good_spec["dimension"] = dim
            for good in [1, 2, 3, 10, 101, 222]:
                good_spec["nxb"] = good
                check_grid_specification(good_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in NOT_INT_LIST:
            bad_spec["nxb"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec, self.__logger)
        for bad in [-1, 0]:
            bad_spec["nxb"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec, self.__logger)

    def testNyb(self):
        good_spec = copy.deepcopy(self.__good)
        # This is correct for all dimensions
        good_spec["nzb"] = 1
        good_spec["dimension"] = 1
        good_spec["nyb"] = 1
        check_grid_specification(good_spec, self.__logger)
        for dim in [2, 3]:
            good_spec["dimension"] = dim
            for good in [1, 2, 3, 10, 101, 222]:
                good_spec["nyb"] = good
                check_grid_specification(good_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in NOT_INT_LIST:
            bad_spec["nyb"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec, self.__logger)
        for bad in [-1, 0]:
            bad_spec["nyb"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_spec["dimension"] = 1
        for bad in [2, 3, 10, 101, 222]:
            bad_spec["nyb"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec, self.__logger)

    def testNzb(self):
        good_spec = copy.deepcopy(self.__good)
        # This is correct for all dimensions
        good_spec["nyb"] = 1
        for dim in [1, 2]:
            good_spec["nzb"] = 1
            check_grid_specification(good_spec, self.__logger)
        good_spec["dimension"] = 3
        for good in [1, 2, 3, 10, 101, 222]:
            good_spec["nzb"] = good
            check_grid_specification(good_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in NOT_INT_LIST:
            bad_spec["nzb"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec, self.__logger)
        for bad in [-1, 0]:
            bad_spec["nzb"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_spec["nyb"] = 1
        for dim in [1, 2]:
            bad_spec["dimension"] = dim
            for bad in [2, 3, 10, 101, 222]:
                bad_spec["nzb"] = bad
                with self.assertRaises(ValueError):
                    check_grid_specification(bad_spec, self.__logger)

    def testNGuardcells(self):
        good_spec = copy.deepcopy(self.__good)
        for good in [0, 1, 2, 3, 10, 101, 222]:
            good_spec["nguardcells"] = good
            check_grid_specification(good_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in NOT_INT_LIST:
            bad_spec["nguardcells"] = bad
            with self.assertRaises(TypeError):
                check_grid_specification(bad_spec, self.__logger)
        for bad in [-2, -1]:
            bad_spec["nguardcells"] = bad
            with self.assertRaises(ValueError):
                check_grid_specification(bad_spec, self.__logger)
