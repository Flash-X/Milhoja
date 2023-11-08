"""
Automatic unit testing of correct detection/reporting of bad subroutine
specifications by SubroutineGroup
"""

import copy
import json
import unittest

from pathlib import Path

from milhoja import (
    LOG_LEVEL_NONE,
    BasicLogger,
    SubroutineGroup
)
from milhoja.tests import (
    NOT_STR_LIST,
    NOT_LIST_LIST, NOT_DICT_LIST,
    NOT_CLASS_LIST
)


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_RUNTIME_PATH = _DATA_PATH.joinpath("runtime")


class TestSubroutineGroup_BadSubroutines(unittest.TestCase):
    def setUp(self):
        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        # Reference must be full operation specification and ideally
        # contain multiple subroutine specs
        ref_fname = _RUNTIME_PATH.joinpath("Math_op1.json")
        with open(ref_fname, "r") as fptr:
            self.__good = json.load(fptr)
        self.__name = "StaticPhysicsRoutines::computeLaplacianDensity"

        # Confirm base spec is correct
        SubroutineGroup.from_milhoja_json(ref_fname, self.__logger)
        SubroutineGroup(self.__good, self.__logger)

    def testBadLogger(self):
        for bad in NOT_CLASS_LIST:
            with self.assertRaises(TypeError):
                SubroutineGroup(self.__good, bad)

    def testBadSource(self):
        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        del bad_sub_spec["argument_specifications"]["lo"]["source"]
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        for bad in NOT_STR_LIST:
            bad_sub_spec["argument_specifications"]["lo"]["source"] = bad
            with self.assertRaises(TypeError):
                SubroutineGroup(bad_spec, self.__logger)

    def testKeys(self):
        # Too few
        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        del bad_sub_spec["interface_file"]
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        bad_sub_spec["fail"] = 1.1
        self.assertTrue(len(bad_sub_spec) >
                        len(self.__good["operation"][self.__name]))
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        # Right number, but incorrect key name
        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        del bad_sub_spec["interface_file"]
        bad_sub_spec["fail"] = 1.1
        self.assertEqual(len(bad_sub_spec),
                         len(self.__good["operation"][self.__name]))
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

    def testInterface(self):
        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        for bad in NOT_STR_LIST:
            bad_sub_spec["interface_file"] = bad
            with self.assertRaises(TypeError):
                SubroutineGroup(bad_spec, self.__logger)

        bad_sub_spec["interface_file"] = ""
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

    def testArgumentList(self):
        good_sub_spec = self.__good["operation"][self.__name]

        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        for bad in NOT_LIST_LIST:
            bad_sub_spec["argument_list"] = bad
            with self.assertRaises(TypeError):
                SubroutineGroup(bad_spec, self.__logger)

        for i in range(len(good_sub_spec["argument_list"])):
            for bad in NOT_STR_LIST:
                bad_spec = copy.deepcopy(self.__good)
                bad_sub_spec = bad_spec["operation"][self.__name]
                bad_sub_spec["argument_list"][i] = bad
                with self.assertRaises(TypeError):
                    SubroutineGroup(bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        bad_list = bad_sub_spec["argument_list"]
        bad_list[1] = bad_list[2]
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

    def testArgumentSpecifications(self):
        good_sub_spec = self.__good["operation"][self.__name]
        n_good = len(good_sub_spec["argument_specifications"])

        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        for bad in NOT_DICT_LIST:
            bad_sub_spec["argument_specifications"] = bad
            with self.assertRaises(TypeError):
                SubroutineGroup(bad_spec, self.__logger)

        # Too few
        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        del bad_sub_spec["argument_specifications"]["hi"]
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        # Too many
        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        bad_sub_spec["argument_specifications"]["fail"] = {}
        n_bad = len(bad_sub_spec["argument_specifications"])
        self.assertTrue(n_bad > n_good)
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

        # Right number, but incorrect key
        bad_spec = copy.deepcopy(self.__good)
        bad_sub_spec = bad_spec["operation"][self.__name]
        del bad_sub_spec["argument_specifications"]["hi"]
        bad_sub_spec["argument_specifications"]["fail"] = {}
        n_bad = len(bad_sub_spec["argument_specifications"])
        self.assertEqual(n_bad, n_good)
        with self.assertRaises(ValueError):
            SubroutineGroup(bad_spec, self.__logger)

    def testChecksEachArgument(self):
        good_sub_spec = self.__good["operation"][self.__name]

        for arg in good_sub_spec["argument_list"]:
            bad_spec = copy.deepcopy(self.__good)
            bad_sub_spec = bad_spec["operation"][self.__name]
            del bad_sub_spec["argument_specifications"][arg]["source"]
            with self.assertRaises(ValueError):
                SubroutineGroup(bad_spec, self.__logger)
