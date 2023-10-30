"""
Automatic unit testing of check_operation_specification()
"""

import copy
import json
import unittest

import numpy as np

from pathlib import Path

from milhoja import check_operation_specification


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_RUNTIME_PATH = _DATA_PATH.joinpath("runtime")


class TestCheckOperationSpecification(unittest.TestCase):
    def setUp(self):
        # Reference must be full operation specification and ideally
        # have both external and scratch variables
        ref_fname = "Math_op1.json"
        with open(_RUNTIME_PATH.joinpath(ref_fname), "r") as fptr:
            self.__good = json.load(fptr)

        # Confirm base spec is correct
        check_operation_specification(self.__good)

    def testKeys(self):
        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["format"]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["fail"] = 1.1
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["format"]
        bad_spec["fail"] = 1.1
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

    def testExternalsOptional(self):
        bad_spec = copy.deepcopy(self.__good)
        self.assertTrue("external" in bad_spec["operation"])

        del bad_spec["operation"]["external"]
        self.assertFalse("external" in bad_spec["operation"])
        check_operation_specification(bad_spec)

    def testExternalsKeys(self):
        VAR = "_dt"

        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"]["external"][VAR]["type"]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["operation"]["external"][VAR]["fail"] = 1.1
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"]["external"][VAR]["type"]
        bad_spec["operation"]["external"][VAR]["fail"] = 1.1
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

    def testExternalsValues(self):
        VAR = "_dt"

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1.1, np.nan, np.inf, [], [1]]:
            bad_spec["operation"]["external"][VAR]["type"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec)

        bad_spec["operation"]["external"][VAR]["type"] = "fail"
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1.1, "fail", np.nan, np.inf, (), (1,)]:
            bad_spec["operation"]["external"][VAR]["extents"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec)

        bad_spec["operation"]["external"][VAR]["extents"] = [1.1]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

    def testScratchOptional(self):
        bad_spec = copy.deepcopy(self.__good)
        self.assertTrue("scratch" in bad_spec["operation"])

        del bad_spec["operation"]["scratch"]
        self.assertFalse("scratch" in bad_spec["operation"])
        check_operation_specification(bad_spec)

    def testScratchKeys(self):
        VAR = "_scratch3D"

        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"]["scratch"][VAR]["type"]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["operation"]["scratch"][VAR]["fail"] = 1.1
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"]["scratch"][VAR]["type"]
        bad_spec["operation"]["scratch"][VAR]["fail"] = 1.1
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

    def testScratchValues(self):
        VAR = "_scratch3D"

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1.1, np.nan, np.inf, [], [1]]:
            bad_spec["operation"]["scratch"][VAR]["type"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec)

        bad_spec["operation"]["scratch"][VAR]["type"] = "fail"
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1.1, np.nan, np.inf, [], [1]]:
            bad_spec["operation"]["scratch"][VAR]["extents"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1.1, np.nan, np.inf, [], [1]]:
            bad_spec["operation"]["scratch"][VAR]["lbound"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec)
