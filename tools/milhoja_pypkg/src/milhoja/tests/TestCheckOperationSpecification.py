"""
Automatic unit testing of check_operation_specification()
"""

import copy
import json
import unittest

import numpy as np
import itertools as it

from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from milhoja import (
    LOG_LEVEL_NONE,
    EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT,
    LogicError,
    BasicLogger,
    check_operation_specification
)


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_FAKE_PATH = _DATA_PATH.joinpath("fake")


class TestCheckOperationSpecification(unittest.TestCase):
    def setUp(self):
        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        # Reference must be full operation specification, have both external
        # and scratch variables, and contain multiple subroutines
        ref_fname = "Math_op1.json"
        with open(_FAKE_PATH.joinpath(ref_fname), "r") as fptr:
            self.__good = json.load(fptr)

        # Confirm base spec is correct
        check_operation_specification(self.__good, self.__logger)

        # Manually determined full set of subroutines in operation
        self.__internals = {
            "StaticPhysicsRoutines::computeLaplacianDensity",
            "StaticPhysicsRoutines::computeLaplacianEnergy",
            "StaticPhysicsRoutines::computeLaplacianFusedKernels"
        }
        # Crosscheck with reality
        keys_all = {"name", "variable_index_base",
                    EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT}
        keys_all = keys_all.union(self.__internals)
        self.assertEqual(set(self.__good["operation"]), keys_all)

    def testBadLogger(self):
        for bad in [None, 1, 1.1, "fail", np.nan, np.inf, [], [1], (), (1,)]:
            with self.assertRaises(TypeError):
                check_operation_specification(self.__good, bad)

    def testKeys(self):
        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["format"]
        self.assertTrue(len(self.__good) > len(bad_spec))
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(self.__good) < len(bad_spec))
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["format"]
        bad_spec["fail"] = 1.1
        self.assertEqual(len(self.__good), len(bad_spec))
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testMissingName(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"]["name"]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testName(self):
        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 0, 1.1, np.nan, np.inf, (), (1,)]:
            bad_spec["operation"]["name"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

        bad_spec["operation"]["name"] = ""
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testVariableIndexBase(self):
        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1.1, "fail", np.nan, np.inf, (), (1,)]:
            bad_spec["operation"]["variable_index_base"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

        for bad in [-100, -2, -1, 2, 3, 4]:
            bad_spec["operation"]["variable_index_base"] = bad
            with self.assertRaises(ValueError):
                check_operation_specification(bad_spec, self.__logger)

    def testMissingVariableIndexBase(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"]["variable_index_base"]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testNoSubroutineSpecifications(self):
        bad_spec = copy.deepcopy(self.__good)
        for subroutine in self.__internals:
            del bad_spec["operation"][subroutine]
        with self.assertRaises(LogicError):
            check_operation_specification(bad_spec, self.__logger)

    def testChecksAllSubroutines(self):
        for subroutine in self.__internals:
            bad_spec = copy.deepcopy(self.__good)
            del bad_spec["operation"][subroutine]["interface_file"]
            with self.assertRaises(ValueError):
                check_operation_specification(bad_spec, self.__logger)

    def testEmptyExternals(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"][EXTERNAL_ARGUMENT]["_dt"]
        self.assertEqual(0, len(bad_spec["operation"][EXTERNAL_ARGUMENT]))
        with self.assertRaises(LogicError):
            check_operation_specification(bad_spec, self.__logger)

    def testExternalsKeys(self):
        VAR = "_dt"

        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["type"]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["fail"] = 1.1
        n_good = len(self.__good["operation"][EXTERNAL_ARGUMENT][VAR])
        n_bad = len(bad_spec["operation"][EXTERNAL_ARGUMENT][VAR])
        self.assertTrue(n_good < n_bad)
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["type"]
        bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["fail"] = 1.1
        n_good = len(self.__good["operation"][EXTERNAL_ARGUMENT][VAR])
        n_bad = len(bad_spec["operation"][EXTERNAL_ARGUMENT][VAR])
        self.assertEqual(n_good, n_bad)
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testExternalsValues(self):
        VAR = "_dt"
        VALID_TYPES = ["real", "integer", "logical"]
        VALID_EXTENTS = [[], [1], [3], [2, 3], list(range(1, 100))]
        GOOD_ARGS = list(it.product(VALID_TYPES, VALID_EXTENTS))
        self.assertEqual(len(VALID_TYPES) * len(VALID_EXTENTS),
                         len(GOOD_ARGS))

        # Confirm correct specifications accepted
        good_spec = copy.deepcopy(self.__good)
        for var_type, extents in GOOD_ARGS:
            good_spec["operation"][EXTERNAL_ARGUMENT][VAR]["type"] = var_type
            good_spec["operation"][EXTERNAL_ARGUMENT][VAR]["extents"] = extents
            check_operation_specification(good_spec, self.__logger)

        # Check bad specifications
        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1.1, np.nan, np.inf, [], [1]]:
            bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["type"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

        bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["type"] = "fail"
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1.1, "fail", np.nan, np.inf, (), (1,)]:
            bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["extents"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

            bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["extents"] = [bad]
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

            bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["extents"] = [bad, 1]
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

        for bad in [-1111, -1, 0]:
            bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["extents"] = [bad]
            with self.assertRaises(ValueError):
                check_operation_specification(bad_spec, self.__logger)

            bad_spec["operation"][EXTERNAL_ARGUMENT][VAR]["extents"] = [bad, 1]
            with self.assertRaises(ValueError):
                check_operation_specification(bad_spec, self.__logger)

    def testMissingHighLevelExternals(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"][EXTERNAL_ARGUMENT]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testEmptyScratch(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"][SCRATCH_ARGUMENT]["_scratch3D"]
        del bad_spec["operation"][SCRATCH_ARGUMENT]["_scratch4D"]
        self.assertEqual(0, len(bad_spec["operation"][SCRATCH_ARGUMENT]))
        with self.assertRaises(LogicError):
            check_operation_specification(bad_spec, self.__logger)

    def testBadExternal(self):
        VAR = "dt"

        bad = "!FAIL!"
        bad_spec = copy.deepcopy(self.__good)
        self.assertTrue(bad not in bad_spec["operation"][EXTERNAL_ARGUMENT])

        subroutine = "StaticPhysicsRoutines::computeLaplacianFusedKernels"
        bad_sub_spec = bad_spec["operation"][subroutine]
        bad_sub_spec["argument_specifications"][VAR]["name"] = bad
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testExternalWarning(self):
        VAR = "dt"

        expected = "external variables not used in any subroutine"

        # Remove external variable from argument list
        bad_spec = copy.deepcopy(self.__good)
        subroutine = "StaticPhysicsRoutines::computeLaplacianFusedKernels"
        bad_sub_spec = bad_spec["operation"][subroutine]
        bad_arg_list = [e for e in bad_sub_spec["argument_list"] if e != VAR]
        bad_sub_spec["argument_list"] = bad_arg_list
        del bad_sub_spec["argument_specifications"][VAR]
        with redirect_stdout(StringIO()) as msg:
            check_operation_specification(bad_spec, self.__logger)

        result = msg.getvalue().strip()
        self.assertTrue("WARNING" in result)
        self.assertTrue(expected in result)

    def testScratchKeys(self):
        VAR = "_scratch3D"

        # Too few
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"][SCRATCH_ARGUMENT][VAR]["type"]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["operation"][SCRATCH_ARGUMENT][VAR]["fail"] = 1.1
        n_good = len(self.__good["operation"][SCRATCH_ARGUMENT][VAR])
        n_bad = len(bad_spec["operation"][SCRATCH_ARGUMENT][VAR])
        self.assertTrue(n_good < n_bad)
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"][SCRATCH_ARGUMENT][VAR]["type"]
        bad_spec["operation"][SCRATCH_ARGUMENT][VAR]["fail"] = 1.1
        n_good = len(self.__good["operation"][SCRATCH_ARGUMENT][VAR])
        n_bad = len(bad_spec["operation"][SCRATCH_ARGUMENT][VAR])
        self.assertEqual(n_good, n_bad)
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testScratchValues(self):
        VAR = "_scratch3D"

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1]]:
            bad_spec["operation"][SCRATCH_ARGUMENT][VAR]["type"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

        bad_spec["operation"][SCRATCH_ARGUMENT][VAR]["type"] = "fail"
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1]]:
            bad_spec["operation"][SCRATCH_ARGUMENT][VAR]["extents"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1]]:
            bad_spec["operation"][SCRATCH_ARGUMENT][VAR]["lbound"] = bad
            with self.assertRaises(TypeError):
                check_operation_specification(bad_spec, self.__logger)

    def testBadScratch(self):
        VAR = "scratch"

        bad = "!FAIL!"
        bad_spec = copy.deepcopy(self.__good)
        self.assertTrue(bad not in bad_spec["operation"][SCRATCH_ARGUMENT])

        subroutine = "StaticPhysicsRoutines::computeLaplacianDensity"
        bad_sub_spec = bad_spec["operation"][subroutine]
        bad_arg_specs = bad_sub_spec["argument_specifications"]
        bad_arg_specs[VAR]["name"] = bad
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testMissingHighLevelScratch(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["operation"][SCRATCH_ARGUMENT]
        with self.assertRaises(ValueError):
            check_operation_specification(bad_spec, self.__logger)

    def testScratchWarning(self):
        VAR = "scratch"

        expected = "scratch variable _scratch4D not used in any subroutine"

        # Remove external variable from argument list
        bad_spec = copy.deepcopy(self.__good)
        subroutine = "StaticPhysicsRoutines::computeLaplacianFusedKernels"
        bad_sub_spec = bad_spec["operation"][subroutine]
        bad_arg_list = bad_sub_spec["argument_list"]
        bad_sub_spec["argument_list"] = [e for e in bad_arg_list if e != VAR]
        del bad_sub_spec["argument_specifications"][VAR]
        with redirect_stdout(StringIO()) as msg:
            check_operation_specification(bad_spec, self.__logger)

        result = msg.getvalue().strip()
        self.assertTrue("WARNING" in result)
        self.assertTrue(expected in result)
