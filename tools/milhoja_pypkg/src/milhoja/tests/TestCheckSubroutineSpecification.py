"""
Automatic unit testing of check_subroutine_specification()
"""

import copy
import json
import unittest

import numpy as np

from pathlib import Path

from milhoja import (
    LOG_LEVEL_NONE,
    BasicLogger,
    check_subroutine_specification
)


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_RUNTIME_PATH = _DATA_PATH.joinpath("runtime")


class TestCheckSubroutineSpecification(unittest.TestCase):
    def setUp(self):
        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        # Reference must be full operation specification and ideally
        # contain multiple subroutine specs
        ref_fname = "Math_op1.json"
        with open(_RUNTIME_PATH.joinpath(ref_fname), "r") as fptr:
            good = json.load(fptr)
        self.__index = good["operation"]["variable_index_base"]
        self.__name = "StaticPhysicsRoutines::computeLaplacianDensity"
        self.__good = good["operation"][self.__name]

        # Confirm base spec is correct
        check_subroutine_specification(self.__name, self.__good,
                                       self.__index, self.__logger)

    def testBadLogger(self):
        for bad in [None, 1, 1.1, "fail", np.nan, np.inf, [], [1], (), (1,)]:
            with self.assertRaises(TypeError):
                check_subroutine_specification(self.__name, self.__good,
                                               self.__index, bad)

    def testBadSource(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["argument_specifications"]["lo"]["source"]
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
            bad_spec["argument_specifications"]["lo"]["source"] = bad
            with self.assertRaises(TypeError):
                check_subroutine_specification(self.__name, bad_spec,
                                               self.__index, self.__logger)

    def testKeys(self):
        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["interface_file"]
        self.assertTrue(len(bad_spec) < len(self.__good))
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(bad_spec) > len(self.__good))
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["interface_file"]
        bad_spec["fail"] = 1.1
        self.assertEqual(len(bad_spec), len(self.__good))
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

    def testInterface(self):
        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
            bad_spec["interface_file"] = bad
            with self.assertRaises(TypeError):
                check_subroutine_specification(self.__name, bad_spec,
                                               self.__index, self.__logger)

        bad_spec["interface_file"] = ""
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

    def testArgumentList(self):
        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, "fail", 1, 1.1, np.nan, np.inf, (), (1,)]:
            bad_spec["argument_list"] = bad
            with self.assertRaises(TypeError):
                check_subroutine_specification(self.__name, bad_spec,
                                               self.__index, self.__logger)

        for i in range(len(self.__good["argument_list"])):
            for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
                bad_spec = copy.deepcopy(self.__good)
                bad_spec["argument_list"][i] = bad
                with self.assertRaises(TypeError):
                    check_subroutine_specification(self.__name, bad_spec,
                                                   self.__index, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_list = bad_spec["argument_list"]
        bad_list[1] = bad_list[2]
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

    def testArgumentSpecifications(self):
        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, "fail", 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
            bad_spec["argument_specifications"] = bad
            with self.assertRaises(TypeError):
                check_subroutine_specification(self.__name, bad_spec,
                                               self.__index, self.__logger)

        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["argument_specifications"]["hi"]
        n_bad = len(bad_spec["argument_specifications"])
        n_good = len(self.__good["argument_specifications"])
        self.assertTrue(n_bad < n_good)
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["argument_specifications"]["fail"] = {}
        n_bad = len(bad_spec["argument_specifications"])
        n_good = len(self.__good["argument_specifications"])
        self.assertTrue(n_bad > n_good)
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

        # Right number, but incorrect key
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["argument_specifications"]["hi"]
        bad_spec["argument_specifications"]["fail"] = {}
        n_bad = len(bad_spec["argument_specifications"])
        n_good = len(self.__good["argument_specifications"])
        self.assertEqual(n_bad, n_good)
        with self.assertRaises(ValueError):
            check_subroutine_specification(self.__name, bad_spec,
                                           self.__index, self.__logger)

    def testChecksEachArgument(self):
        for arg in self.__good["argument_list"]:
            bad_spec = copy.deepcopy(self.__good)
            del bad_spec["argument_specifications"][arg]["source"]
            with self.assertRaises(ValueError):
                check_subroutine_specification(self.__name, bad_spec,
                                               self.__index, self.__logger)
