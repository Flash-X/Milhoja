"""
Automatic unit testing of check_grid_data_specification()
"""

import copy
import unittest

import numpy as np
import itertools as it

from milhoja import (
    LOG_LEVEL_NONE,
    GRID_DATA_ARGUMENT, SCRATCH_ARGUMENT,
    LogicError,
    BasicLogger,
    check_grid_data_specification
)


class TestCheckGridDataSpecification(unittest.TestCase):
    def setUp(self):
        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        self.__name = "U"
        self.__index = 1
        self.__good = {
            "source": GRID_DATA_ARGUMENT,
            "structure_index": ["center", 1],
            "R": [1]
        }
        check_grid_data_specification(self.__name, self.__good,
                                      self.__index, self.__logger)

    def testBadSource(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["source"]
        with self.assertRaises(ValueError):
            check_grid_data_specification(self.__name, bad_spec,
                                          self.__index, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
            bad_spec["source"] = bad
            with self.assertRaises(TypeError):
                check_grid_data_specification(self.__name, bad_spec,
                                              self.__index, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_spec["source"] = SCRATCH_ARGUMENT
        with self.assertRaises(LogicError):
            check_grid_data_specification(self.__name, bad_spec,
                                          self.__index, self.__logger)

    def testBadLogger(self):
        for bad in [None, 1, 1.1, "fail", np.nan, np.inf, [], [1], (), (1,)]:
            with self.assertRaises(TypeError):
                check_grid_data_specification(self.__name, self.__good,
                                              self.__index, bad)

    def testGoodSpecs(self):
        name = "unk"
        good_spaces = ["CenTER", "center",
                       "FluxX", "fluxx", "FluxY", "fluxy", "FlUxZ", "fluxz"]
        good_index = [1]
        good_var_base = [0, 1]
        good_all = list(it.product(good_spaces, good_index, good_var_base))
        n_cases = len(good_spaces) * len(good_index) * len(good_var_base)
        self.assertEqual(len(good_all), n_cases)

        for space, index, var_base in good_all:
            good_vars = list(range(var_base, 12))
            for access in ["R", "RW", "W"]:
                good_spec = {
                    "source": GRID_DATA_ARGUMENT,
                    "structure_index": [space, index],
                    access: good_vars
                }
                check_grid_data_specification(name, good_spec,
                                              var_base, self.__logger)

            for a1, a2 in it.combinations(["R", "RW", "W"], 2):
                # Must be disjoint, but make out of order and non-contiguous.
                # Include the index base.
                vars1 = [var_base, 3, 5]
                vars2 = [9, 2]
                good_spec = {
                    "source": GRID_DATA_ARGUMENT,
                    "structure_index": [space, index],
                    a1: vars1, a2: vars2
                }
                check_grid_data_specification(name, good_spec,
                                              var_base, self.__logger)

                good_spec = {
                    "source": GRID_DATA_ARGUMENT,
                    "structure_index": [space, index],
                    a1: vars2, a2: vars1
                }
                check_grid_data_specification(name, good_spec,
                                              var_base, self.__logger)

            # Must be disjoint, but make out of order and non-contiguous.
            # Include the index base.
            vars1 = [99, 3, 5]
            vars2 = [9, 2]
            vars3 = [var_base, 101]
            for a1, a2, a3 in it.permutations(["R", "RW", "W"]):
                good_spec = {
                    "source": GRID_DATA_ARGUMENT,
                    "structure_index": [space, index],
                    a1: vars1, a2: vars2, a3: vars3
                }
                check_grid_data_specification(name, good_spec,
                                              var_base, self.__logger)

    def testKeys(self):
        # Missing mandatory keys
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["structure_index"]
        with self.assertRaises(ValueError):
            check_grid_data_specification(self.__name, bad_spec,
                                          self.__index, self.__logger)

        # No R/RW/W keys
        bad_spec = copy.deepcopy(self.__good)
        self.assertTrue("R" in bad_spec)
        self.assertTrue("RW" not in bad_spec)
        self.assertTrue("W" not in bad_spec)
        del bad_spec["R"]
        with self.assertRaises(ValueError):
            check_grid_data_specification(self.__name, bad_spec,
                                          self.__index, self.__logger)

        # Too many keys
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(bad_spec) > len(self.__good))
        with self.assertRaises(ValueError):
            check_grid_data_specification(self.__name, bad_spec,
                                          self.__index, self.__logger)

        # Right number of keys, but bad key
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["structure_index"]
        bad_spec["fail"] = 1.1
        self.assertEqual(len(bad_spec), len(self.__good))
        with self.assertRaises(ValueError):
            check_grid_data_specification(self.__name, bad_spec,
                                          self.__index, self.__logger)

    def testBadStructureIndex(self):
        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, "fail", 1, 1.1, np.nan, np.inf, (), (1,)]:
            bad_spec["structure_index"] = bad
            with self.assertRaises(TypeError):
                check_grid_data_specification(self.__name, bad_spec,
                                              self.__index, self.__logger)
        for bad in ([], ["center"], ["center", 1, None]):
            bad_spec["structure_index"] = bad
            with self.assertRaises(ValueError):
                check_grid_data_specification(self.__name, bad_spec,
                                              self.__index, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
            bad_spec["structure_index"][0] = bad
            with self.assertRaises(TypeError):
                check_grid_data_specification(self.__name, bad_spec,
                                              self.__index, self.__logger)
        bad_spec["structure_index"][0] = "fail"
        with self.assertRaises(ValueError):
            check_grid_data_specification(self.__name, bad_spec,
                                          self.__index, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, "fail", 1.1, np.nan, np.inf, [], [1], (), (1,)]:
            bad_spec["structure_index"][1] = bad
            with self.assertRaises(TypeError):
                check_grid_data_specification(self.__name, bad_spec,
                                              self.__index, self.__logger)
        for bad in [-1, 0, 2, 3]:
            bad_spec["structure_index"][1] = bad
            with self.assertRaises(ValueError):
                check_grid_data_specification(self.__name, bad_spec,
                                              self.__index, self.__logger)

    def testBadAccessPatterns(self):
        for access in ["R", "RW", "W"]:
            bad_spec = copy.deepcopy(self.__good)
            for bad in [None, "fail", 1, 1.1, np.nan, np.inf, (), (1,)]:
                bad_spec[access] = bad
                with self.assertRaises(TypeError):
                    check_grid_data_specification(self.__name, bad_spec,
                                                  self.__index, self.__logger)

            bad_spec = copy.deepcopy(self.__good)
            bad_spec[access] = []
            with self.assertRaises(ValueError):
                check_grid_data_specification(self.__name, bad_spec,
                                              self.__index, self.__logger)

            for bad in [None, "fail", 1.1, np.nan, np.inf, (), (1,)]:
                bad_spec[access] = [3, 1, bad, 2]
                with self.assertRaises(TypeError):
                    check_grid_data_specification(self.__name, bad_spec,
                                                  self.__index, self.__logger)

        # Check for bad access patterns based on variable index set's base
        for access in ["R", "RW", "W"]:
            bad_spec = copy.deepcopy(self.__good)
            index = 1
            for bad in [-2, -1, 0]:
                bad_spec[access] = [bad]
                with self.assertRaises(ValueError):
                    check_grid_data_specification(self.__name, bad_spec,
                                                  index, self.__logger)

                bad_spec[access] = [3, 1, bad, 2]
                with self.assertRaises(ValueError):
                    check_grid_data_specification(self.__name, bad_spec,
                                                  index, self.__logger)

            index = 0
            for bad in [-2, -1]:
                bad_spec[access] = [bad]
                with self.assertRaises(ValueError):
                    check_grid_data_specification(self.__name, bad_spec,
                                                  index, self.__logger)

                bad_spec[access] = [3, 1, bad, 2]
                with self.assertRaises(ValueError):
                    check_grid_data_specification(self.__name, bad_spec,
                                                  index, self.__logger)

        # Variables can have only one access pattern
        common = 3
        for a1, a2 in it.combinations(["R", "RW", "W"], 2):
            bad_spec = copy.deepcopy(self.__good)
            del bad_spec["R"]
            bad_spec[a1] = [1, common, 5]
            bad_spec[a2] = [9, 2, common]
            with self.assertRaises(ValueError):
                check_grid_data_specification(self.__name, bad_spec,
                                              self.__index, self.__logger)
