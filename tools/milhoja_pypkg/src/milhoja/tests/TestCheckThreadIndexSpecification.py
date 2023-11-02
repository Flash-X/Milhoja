"""
Automatic unit testing of check_thread_index_specification()
"""

import copy
import unittest

import numpy as np

from milhoja import (
    LOG_LEVEL_NONE,
    TILE_LO_ARGUMENT, THREAD_INDEX_ARGUMENT,
    LogicError,
    BasicLogger,
    check_thread_index_specification
)


class TestCheckThreadIndexSpecification(unittest.TestCase):
    def setUp(self):
        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        self.__name = "thread_index"
        self.__good = {"source": THREAD_INDEX_ARGUMENT}
        check_thread_index_specification(self.__name, self.__good,
                                         self.__logger)

    def testBadSource(self):
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["source"]
        with self.assertRaises(ValueError):
            check_thread_index_specification(self.__name, bad_spec,
                                             self.__logger)

        bad_spec = copy.deepcopy(self.__good)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
            bad_spec["source"] = bad
            with self.assertRaises(TypeError):
                check_thread_index_specification(self.__name, bad_spec,
                                                 self.__logger)

        bad_spec["source"] = TILE_LO_ARGUMENT
        with self.assertRaises(LogicError):
            check_thread_index_specification(self.__name, bad_spec,
                                             self.__logger)

    def testBadLogger(self):
        for bad in [None, 1, 1.1, "fail", np.nan, np.inf, [], [1], (), (1,)]:
            with self.assertRaises(TypeError):
                check_thread_index_specification(self.__name, self.__good, bad)

    def testKeys(self):
        # Too few keys checked in testBadSource

        # Too many keys
        bad_spec = copy.deepcopy(self.__good)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(bad_spec) > len(self.__good))
        with self.assertRaises(ValueError):
            check_thread_index_specification(self.__name, bad_spec,
                                             self.__logger)

        # Right number of keys, but bad key
        bad_spec = copy.deepcopy(self.__good)
        del bad_spec["source"]
        bad_spec["fail"] = 1.1
        self.assertEqual(len(bad_spec), len(self.__good))
        with self.assertRaises(ValueError):
            check_thread_index_specification(self.__name, bad_spec,
                                             self.__logger)
