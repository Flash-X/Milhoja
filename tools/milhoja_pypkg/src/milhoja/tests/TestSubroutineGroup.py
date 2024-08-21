"""
Automatic unit testing of SubroutineGroup
"""

import os
import copy
import json
import shutil
import unittest

from pathlib import Path

from milhoja import (
    LOG_LEVEL_NONE,
    EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT,
    BasicLogger,
    SubroutineGroup
)
from milhoja.tests import (
    NOT_STR_LIST, NOT_DICT_LIST, NOT_CLASS_LIST
)


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_FAKE_PATH = _DATA_PATH.joinpath("fake")


class TestSubroutineGroup(unittest.TestCase):
    def setUp(self):
        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        self.__subroutines = [
            "StaticPhysicsRoutines::computeLaplacianDensity",
            "StaticPhysicsRoutines::computeLaplacianEnergy",
            "StaticPhysicsRoutines::computeLaplacianFusedKernels"
        ]

        # Confirm correct
        self.__fname = _FAKE_PATH.joinpath("Math_op1.json")
        with open(self.__fname, "r") as fptr:
            self.__good = json.load(fptr)
        self.__group = SubroutineGroup.from_milhoja_json(
            self.__fname, self.__logger
        )
        SubroutineGroup(self.__good, self.__logger)

    def testFromMilhojaJson(self):
        dst = Path.cwd().joinpath("delete_me")
        if dst.exists():
            shutil.rmtree(dst)
        os.makedirs(dst)

        BAD_FNAME = "FAIL.FAIL.loog"
        TMP_FNAME = dst.joinpath(BAD_FNAME)

        # Correct use already tested in setUp()

        # Bad filename argument
        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                SubroutineGroup.from_milhoja_json(bad, self.__logger)

        bad = Path.cwd().joinpath(BAD_FNAME)
        self.assertFalse(bad.is_file())
        with self.assertRaises(ValueError):
            SubroutineGroup.from_milhoja_json(bad, self.__logger)
        with self.assertRaises(ValueError):
            SubroutineGroup.from_milhoja_json(BAD_FNAME, self.__logger)
        with self.assertRaises(ValueError):
            SubroutineGroup.from_milhoja_json(dst, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_spec["frmat"] = bad_spec["format"]
        del bad_spec["format"]
        with open(TMP_FNAME, "w") as fptr:
            json.dump(bad_spec, fptr)
        with self.assertRaises(ValueError):
            SubroutineGroup.from_milhoja_json(TMP_FNAME, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_spec["format"][0] = "!Not a Format!"
        with open(TMP_FNAME, "w") as fptr:
            json.dump(bad_spec, fptr)
        with self.assertRaises(ValueError):
            SubroutineGroup.from_milhoja_json(TMP_FNAME, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_spec["format"][1] = "!Not a Version!"
        with open(TMP_FNAME, "w") as fptr:
            json.dump(bad_spec, fptr)
        with self.assertRaises(ValueError):
            SubroutineGroup.from_milhoja_json(TMP_FNAME, self.__logger)

        # Bad logger argument
        for bad in NOT_CLASS_LIST:
            with self.assertRaises(TypeError):
                SubroutineGroup.from_milhoja_json(self.__fname, bad)

        shutil.rmtree(dst)

    def testConstruction(self):
        # Correct construction already tested in setUp()

        # Bad group argument
        for bad in NOT_DICT_LIST:
            with self.assertRaises(TypeError):
                SubroutineGroup(bad, self.__logger)

        # Bad logger argument
        for bad in NOT_CLASS_LIST:
            with self.assertRaises(TypeError):
                SubroutineGroup(self.__good, bad)

        # Bad spec tested in TestSubroutine_Bad*.py test cases

    def testContains(self):
        for good in self.__subroutines:
            self.assertTrue(good in self.__group)
        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                self.assertFalse(bad in self.__group)
        for bad in ["nope", "fail", "why?!"]:
            self.assertFalse(bad in self.__group)

        # These are keys in the spec, but not subroutines
        not_subroutines = {"name", "variable_index_base",
                           EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT}
        for bad in not_subroutines:
            self.assertFalse(bad in self.__group)

    def testName(self):
        self.assertEqual("base_op1", self.__group.name)

    def testVariableIndexBase(self):
        # TODO: This should change it to 0 and recheck.
        self.assertEqual(1, self.__group.variable_index_base)

    def testSubroutines(self):
        result = self.__group.subroutines
        self.assertEqual(len(self.__subroutines), len(result))
        self.assertEqual(len(result), len(set(result)))
        self.assertEqual(set(self.__subroutines), set(result))

    def testSpecification(self):
        # Confirm correct
        group_spec = self.__group.specification

        # This assumes that __good is Milhoja-internal format spec
        self.assertEqual(self.__good, group_spec)
        result = SubroutineGroup(group_spec, self.__logger)
        self.assertEqual(self.__good, result.specification)

        expected = {
            "format", "name", "variable_index_base",
            EXTERNAL_ARGUMENT,
            SCRATCH_ARGUMENT
        }.union(self.__group.subroutines)
        self.assertEqual(expected, set(group_spec))

        # Confirm consistency with finer-grained getters
        expected = self.__group.group_external_variables
        self.assertEqual(set(expected),
                         set(group_spec[EXTERNAL_ARGUMENT]))

        expected = self.__group.group_scratch_variables
        self.assertEqual(set(expected),
                         set(group_spec[SCRATCH_ARGUMENT]))

        for subroutine in self.__group.subroutines:
            expected = self.__group.subroutine_specification(subroutine)
            result = group_spec[subroutine]
            self.assertEqual(expected, result)

        # Confirm that we cannot alter original spec in object
        #
        # If you remove deepcopy in function, then this will fail.
        group_spec["name"] = "do not fail please"
        result = self.__group.specification
        self.assertNotEqual(group_spec, result)

    def testGroupExternalVariables(self):
        self.assertEqual({"_dt"}, self.__group.group_external_variables)

    def testExternalSpecification(self):
        # Confirm correct
        arg_spec = self.__group.external_specification("_dt")
        expected = {"type": "real", "extents": "(2, 3)"}
        self.assertEqual(expected, arg_spec)

        # Confirm that we cannot alter original spec in object
        #
        # If you remove copy in function, then this will fail.
        arg_spec["type"] = "integer"
        self.assertNotEqual(arg_spec,
                            self.__group.external_specification("_dt"))

        # Bad argument
        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                self.__group.external_specification(bad)

        with self.assertRaises(ValueError):
            self.__group.external_specification("!no_way this is a valid n@me?")

    def testGroupScratchVariables(self):
        expected = {"_scratch3D", "_scratch4D"}
        self.assertEqual(expected, self.__group.group_scratch_variables)

    def testScratchSpecification(self):
        # Confirm correct
        arg_spec = self.__group.scratch_specification("_scratch3D")
        expected = {
            "type": "real",
            "extents": "(8, 16, 1)",
            "lbound": "(tile_lo)"
        }
        self.assertEqual(expected, arg_spec)

        arg_spec = self.__group.scratch_specification("_scratch4D")
        expected = {
            "type": "real",
            "extents": "(8, 16, 1, 2)",
            "lbound": "(tile_lo, 1)"
        }
        self.assertEqual(expected, arg_spec)

        # Confirm that we cannot alter original spec in object
        #
        # If you remove copy in function, then this will fail.
        arg_spec["type"] = "integer"
        self.assertNotEqual(arg_spec,
                            self.__group.scratch_specification("_scratch4D"))

        # Bad argument
        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                self.__group.scratch_specification(bad)

        with self.assertRaises(ValueError):
            self.__group.scratch_specification("!no_way this is a valid n@me?")

    def testSubroutineSpecification(self):
        # Confirm correct
        for subroutine in self.__group.subroutines:
            expected = {"interface_file",
                        "argument_list",
                        "argument_specifications"}
            result = self.__group.subroutine_specification(subroutine)
            self.assertEqual(expected, set(result))

            expected = subroutine.replace("StaticPhysicsRoutines::", "") + ".h"
            expected = expected.replace("Kernels", "")
            self.assertEqual(expected, result["interface_file"])

            # Confirm consistency with finer-grained getter functions
            expected = self.__group.argument_list(subroutine)
            self.assertEqual(expected, result["argument_list"])
            self.assertEqual(set(expected),
                             set(result["argument_specifications"]))
            for argument in self.__group.argument_list(subroutine):
                expected = self.__group.argument_specification(subroutine,
                                                               argument)
                self.assertEqual(expected,
                                 result["argument_specifications"][argument])

            # Confirm that we cannot alter original spec in object
            #
            # If you remove copy in function, then this will fail.
            result["interface_file"] = "doNot_fail_please"
            self.assertNotEqual(
                result,
                self.__group.subroutine_specification(subroutine)
            )

        # Bad subroutine name
        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                self.__group.subroutine_specification(bad)

        with self.assertRaises(ValueError):
            self.__group.subroutine_specification("NoWaYThiS_Is_SubR#")

    def testArgumentList(self):
        # Confirm correct
        subroutine = "StaticPhysicsRoutines::computeLaplacianDensity"
        expected = ["lo", "hi", "U", "scratch", "deltas"]
        result = self.__group.argument_list(subroutine)
        self.assertEqual(expected, result)

        subroutine = "StaticPhysicsRoutines::computeLaplacianEnergy"
        result = self.__group.argument_list(subroutine)
        self.assertEqual(expected, result)

        subroutine = "StaticPhysicsRoutines::computeLaplacianFusedKernels"
        expected = ["dt"] + expected
        result = self.__group.argument_list(subroutine)
        self.assertEqual(expected, result)

        # Confirm that we cannot alter original spec in object
        #
        # If you remove copy in function, then this will fail.
        result[0] = "do not fail please"
        self.assertNotEqual(result, self.__group.argument_list(subroutine))

        # Bad subroutine name
        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                self.__group.argument_list(bad)

        with self.assertRaises(ValueError):
            self.__group.argument_list("NotASubroutineNameIdBet")

    def testArgumentSpecification(self):
        # Confirm correct
        good_sub = "StaticPhysicsRoutines::computeLaplacianDensity"
        good_arg = "deltas"
        expected = {"source": "tile_deltas"}
        arg_spec = self.__group.argument_specification(good_sub, good_arg)
        self.assertEqual(expected, arg_spec)

        good_sub = "StaticPhysicsRoutines::computeLaplacianEnergy"
        good_arg = "U"
        expected = {
            "source": "grid_data",
            "structure_index": ["CENTER", 1],
            "rw": [2]
        }
        arg_spec = self.__group.argument_specification(good_sub, good_arg)
        self.assertEqual(expected, arg_spec)

        good_sub = "StaticPhysicsRoutines::computeLaplacianFusedKernels"
        good_arg = "dt"
        expected = {"source": EXTERNAL_ARGUMENT, "name": "_dt"}
        arg_spec = self.__group.argument_specification(good_sub, good_arg)
        good_arg = "scratch"
        expected = {"source": SCRATCH_ARGUMENT, "name": "_scratch4D"}
        arg_spec = self.__group.argument_specification(good_sub, good_arg)
        self.assertEqual(expected, arg_spec)

        # Confirm that we cannot alter original spec in object
        #
        # If you remove deepcopy in function, then this will fail.
        arg_spec["name"] = arg_spec["name"] + "NoFailPlea_se"
        self.assertNotEqual(
            arg_spec,
            self.__group.argument_specification(good_sub, good_arg)
        )

        # Bad subroutine name
        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                self.__group.argument_specification(bad, good_arg)

        with self.assertRaises(ValueError):
            self.__group.argument_specification("NotAVariable_namE", good_arg)

        # Bad argument name
        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                self.__group.argument_specification(good_sub, bad)

        with self.assertRaises(ValueError):
            self.__group.argument_specification(good_sub, "NotAnArgUMeNT_N@mE")
