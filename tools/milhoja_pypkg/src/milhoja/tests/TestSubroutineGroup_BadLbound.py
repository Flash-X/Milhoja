"""
Automatic unit testing of SubroutineGroup's error checking of lbound
argument specifications.
"""

import copy
import json
import unittest

from pathlib import Path

from milhoja import (
    LOG_LEVEL_NONE,
    LBOUND_ARGUMENT,
    LogicError,
    BasicLogger,
    SubroutineGroup
)
from milhoja.tests import NOT_STR_LIST


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_FAKE_PATH = _DATA_PATH.joinpath("fake")


class TestSubroutineGroup_BadLbound(unittest.TestCase):
    def setUp(self):
        self.__VAR = "loCoeffs"
        self.__SUBR = "functionA"

        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        fname = _FAKE_PATH.joinpath("FakeA_op1.json")
        with open(fname, "r") as fptr:
            self.__good = json.load(fptr)
        SubroutineGroup(self.__good, self.__logger)

        arg_specs = self.__good[self.__SUBR]["argument_specifications"]
        var_spec = arg_specs[self.__VAR]
        self.assertEqual(LBOUND_ARGUMENT, var_spec["source"])

    def testBadSource(self):
        bad_spec = copy.deepcopy(self.__good)
        bad_arg_specs = bad_spec[self.__SUBR]["argument_specifications"]
        bad_var_spec = bad_arg_specs[self.__VAR]
        del bad_var_spec["source"]
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_arg_specs = bad_spec[self.__SUBR]["argument_specifications"]
        bad_var_spec = bad_arg_specs[self.__VAR]
        for bad in NOT_STR_LIST:
            bad_var_spec["source"] = bad
            with self.assertRaises(TypeError):
                SubroutineGroup(bad_spec, self.__logger)

    def testKeys(self):
        good_arg_specs = self.__good[self.__SUBR]["argument_specifications"]
        good_var_spec = good_arg_specs[self.__VAR]
        n_good = len(good_var_spec)

        # Too few keys
        bad_spec = copy.deepcopy(self.__good)
        bad_arg_specs = bad_spec[self.__SUBR]["argument_specifications"]
        bad_var_spec = bad_arg_specs[self.__VAR]
        del bad_var_spec["array"]
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        # Too many keys
        bad_spec = copy.deepcopy(self.__good)
        bad_arg_specs = bad_spec[self.__SUBR]["argument_specifications"]
        bad_var_spec = bad_arg_specs[self.__VAR]
        bad_var_spec["fail"] = 1.1
        self.assertTrue(len(bad_var_spec) > n_good)
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        # Right number of keys, but bad key
        bad_spec = copy.deepcopy(self.__good)
        bad_arg_specs = bad_spec[self.__SUBR]["argument_specifications"]
        bad_var_spec = bad_arg_specs[self.__VAR]
        del bad_var_spec["array"]
        bad_var_spec["fail"] = 1.1
        self.assertEqual(len(bad_var_spec), n_good)
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

    def testArray(self):
        HAVE_LBOUND = {
            "coefficients", "yCoords", "zAreas", "volumes", "U", "flX",
            "scratch4D", "scratch5D"
        }
        COULD_HAVE_LBOUND = {
            "xCoords", "zCoords", "xAreas", "yAreas", "flY", "flZ"
        }
        CANNOT_HAVE_LBOUND = {
            "dt",
            "gid", "deltas", "lo", "hi", "ubdd", "interior",
            "loCoeffs", "loYCoords", "loZAreas", "loVolumes",
            "loU", "loFl", "lo_S4D", "lo_S5D"
        }
        expected = set(self.__good[self.__SUBR]["argument_specifications"])
        result = HAVE_LBOUND.union(COULD_HAVE_LBOUND, CANNOT_HAVE_LBOUND)
        self.assertEqual(expected, result)

        good_spec = copy.deepcopy(self.__good)
        good_arg_specs = good_spec[self.__SUBR]["argument_specifications"]
        good_var_spec = good_arg_specs[self.__VAR]
        for each in COULD_HAVE_LBOUND:
            good_var_spec["array"] = each
            SubroutineGroup(good_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_arg_specs = bad_spec[self.__SUBR]["argument_specifications"]
        bad_var_spec = bad_arg_specs[self.__VAR]
        for bad in NOT_STR_LIST:
            bad_var_spec["array"] = bad
            with self.assertRaises(TypeError):
                SubroutineGroup(bad_spec, self.__logger)

        bad_var_spec["array"] = ""
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)
        bad_var_spec["array"] = "notInFuntionA_arg_lisT!"
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        # We test here that "dt", which is an external scalar, is rejected
        bad_spec = copy.deepcopy(self.__good)
        bad_arg_specs = bad_spec[self.__SUBR]["argument_specifications"]
        bad_var_spec = bad_arg_specs[self.__VAR]
        for each in CANNOT_HAVE_LBOUND:
            self.assertTrue(each in bad_arg_specs)
            bad_var_spec["array"] = each
            if each == self.__VAR:
                # It's a different type of error if an lbound is set to itself
                with self.assertRaises(LogicError):
                    SubroutineGroup(bad_spec, self.__logger)
            else:
                with self.assertRaises(ValueError):
                    SubroutineGroup(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_arg_specs = bad_spec[self.__SUBR]["argument_specifications"]
        bad_arg_specs["loU"]["array"] = "xCoords"
        bad_arg_specs["loCoeffs"]["array"] = "xCoords"
        with self.assertRaises(LogicError):
            SubroutineGroup(bad_spec, self.__logger)
