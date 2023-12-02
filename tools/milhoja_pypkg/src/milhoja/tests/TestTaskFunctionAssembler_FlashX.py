"""
Automatic unit testing of TaskFunctionAssembler for use with Flash-X test
problems
"""

import os
import json
import shutil
import unittest

from pathlib import Path

from milhoja import (
    LOG_LEVEL_NONE,
    BasicLogger
)
from milhoja.tests import (
    generate_flashx_cpu_tf_specs, generate_flashx_gpu_tf_specs
)

_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_FLASHX_PATH = _DATA_PATH.joinpath("FlashX")


class TestTaskFunctionAssembler_FlashX(unittest.TestCase):
    def setUp(self):
        self.__dst = Path.cwd().joinpath("delete_me")
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.makedirs(self.__dst)

        self.__logger = BasicLogger(LOG_LEVEL_NONE)

    def testSedovCpu(self):
        for dimension in [1, 2, 3]:
            if dimension == 1:
                nxb, nyb, nzb = (32, 1, 1)
            elif dimension == 2:
                nxb, nyb, nzb = (8, 8, 1)
            elif dimension == 3:
                nxb, nyb, nzb = (16, 16, 16)

            filename = f"cpu_tf_hydro_{dimension}D.json"
            tf_spec_fname = self.__dst.joinpath(filename)
            self.assertFalse(tf_spec_fname.exists())
            filename = generate_flashx_cpu_tf_specs(
                         dimension, [nxb, nyb, nzb], _FLASHX_PATH,
                         self.__dst, False, self.__logger
                       )
            self.assertEqual(tf_spec_fname, filename)
            self.assertTrue(tf_spec_fname.is_file())

            filename = f"REF_cpu_tf_hydro_{dimension}D.json"
            with open(_FLASHX_PATH.joinpath(filename), "r") as fptr:
                expected = json.load(fptr)
            with open(tf_spec_fname, "r") as fptr:
                result = json.load(fptr)

            # ----- TEST AT FINER SCALE AS THIS CAN HELP DEBUG FAILURES
            key = "format"
            self.assertTrue(key in expected)
            self.assertTrue(key in result)
            self.assertEqual(expected[key], result[key])

            groups_all = ["grid", "task_function", "data_item", "subroutines"]

            self.maxDiff = None

            self.assertEqual(len(expected), len(result))
            for group in groups_all:
                # print(group)
                # print(expected[group])
                # print(result[group])
                self.assertTrue(group in expected)
                self.assertTrue(group in result)

                self.assertEqual(len(expected[group]), len(result[group]))
                for key in expected[group]:
                    # print(group, key)
                    # print(expected[group][key])
                    # print(result[group][key])
                    self.assertTrue(key in result[group])
                    self.assertEqual(expected[group][key], result[group][key])

            # ----- DEBUG AT COARSEST SCALE AS INTENDED
            self.assertEqual(expected, result)

    def testSedovGpu(self):
        for dimension in [1, 2, 3]:
            if dimension == 1:
                nxb, nyb, nzb = (32, 1, 1)
            elif dimension == 2:
                nxb, nyb, nzb = (8, 8, 1)
            elif dimension == 3:
                nxb, nyb, nzb = (16, 16, 16)

            filename = f"gpu_tf_hydro_{dimension}D.json"
            tf_spec_fname = self.__dst.joinpath(filename)
            self.assertFalse(tf_spec_fname.exists())
            filename = generate_flashx_gpu_tf_specs(
                         dimension, [nxb, nyb, nzb], _FLASHX_PATH,
                         self.__dst, False, self.__logger
                       )
            self.assertEqual(tf_spec_fname, filename)
            self.assertTrue(tf_spec_fname.is_file())

            filename = f"REF_gpu_tf_hydro_{dimension}D.json"
            with open(_FLASHX_PATH.joinpath(filename), "r") as fptr:
                expected = json.load(fptr)
            with open(tf_spec_fname, "r") as fptr:
                result = json.load(fptr)

            # ----- TEST AT FINER SCALE AS THIS CAN HELP DEBUG FAILURES
            key = "format"
            self.assertTrue(key in expected)
            self.assertTrue(key in result)
            self.assertEqual(expected[key], result[key])

            groups_all = ["grid", "task_function", "data_item", "subroutines"]

            self.maxDiff = None

            self.assertEqual(len(expected), len(result))
            for group in groups_all:
                # print(group)
                # print(expected[group])
                # print(result[group])
                self.assertTrue(group in expected)
                self.assertTrue(group in result)

                self.assertEqual(len(expected[group]), len(result[group]))
                for key in expected[group]:
                    # print(group, key)
                    # print(expected[group][key])
                    # print(result[group][key])
                    self.assertTrue(key in result[group])
                    self.assertEqual(expected[group][key], result[group][key])

            # ----- DEBUG AT COARSEST SCALE AS INTENDED
            self.assertEqual(expected, result)
