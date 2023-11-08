"""
Automatic unit testing of check_external_specification()
"""

import copy
import unittest

from milhoja import (
    LOG_LEVEL_NONE,
    EXTERNAL_ARGUMENT,
    SCRATCH_ARGUMENT,
    LogicError,
    BasicLogger,
    check_external_specification
)
from milhoja.tests import (
    NOT_STR_LIST, NOT_CLASS_LIST
)


class TestCheckExternalSpecification(unittest.TestCase):
    def setUp(self):
        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        self.__name = "optional"
        self.__good = {"source": EXTERNAL_ARGUMENT, "name": "unimportant"}
        check_external_specification(self.__name, self.__good, self.__logger)

    def testBadSource(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["source"]
        with self.assertRaises(ValueError):
            check_external_specification(self.__name, bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in NOT_STR_LIST:
            bad_spec["source"] = bad
            with self.assertRaises(TypeError):
                check_external_specification(self.__name, bad_spec,
                                             self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        bad_spec["source"] = SCRATCH_ARGUMENT
        with self.assertRaises(LogicError):
            check_external_specification(self.__name, bad_spec, self.__logger)

    def testBadLogger(self):
        for bad in NOT_CLASS_LIST:
            with self.assertRaises(TypeError):
                check_external_specification(self.__name, self.__good, bad)

    def testKeys(self):
        # Too few keys
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["name"]
        with self.assertRaises(ValueError):
            check_external_specification(self.__name, bad_spec, self.__logger)

        # Too many keys
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(bad_spec) > len(self.__good))
        with self.assertRaises(ValueError):
            check_external_specification(self.__name, bad_spec, self.__logger)

        # Right number of keys, but bad key
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["name"]
        bad_spec["fail"] = 1.1
        self.assertEqual(len(bad_spec), len(self.__good))
        with self.assertRaises(ValueError):
            check_external_specification(self.__name, bad_spec, self.__logger)

    def testName(self):
        bad_spec = copy.deepcopy(self.__good)
        for bad in NOT_STR_LIST:
            bad_spec["name"] = bad
            with self.assertRaises(TypeError):
                check_external_specification(self.__name, bad_spec,
                                             self.__logger)
