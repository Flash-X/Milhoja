"""
Automatic unit testing of TaskFunctionAssembler
"""

import os
import json
import shutil
import unittest

from pathlib import Path

import milhoja.tests


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_SEDOV_PATH = _DATA_PATH.joinpath("Sedov")
_RUNTIME_PATH = _DATA_PATH.joinpath("runtime")


class TestTaskFunctionAssembler(unittest.TestCase):
    def setUp(self):
        self.__dst = Path.cwd().joinpath("delete_me")
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.makedirs(self.__dst)

        self.__logger = milhoja.BasicLogger(milhoja.LOG_LEVEL_NONE)

        gpu_spec_fname = self.__dst.joinpath("gpu_tf_hydro_3D.json")
        self.assertFalse(gpu_spec_fname.exists())
        filename = milhoja.tests.generate_sedov_gpu_tf_specs(
                     3, [16, 16, 16], _SEDOV_PATH,
                     self.__dst, False, self.__logger
                   )
        self.assertEqual(gpu_spec_fname, filename)
        self.assertTrue(gpu_spec_fname.is_file())

        with open(gpu_spec_fname, "r") as fptr:
            tf_spec = json.load(fptr)
        tf_call_graph = tf_spec["task_function"]["subroutine_call_graph"]
        op_spec_json = self.__dst.joinpath("Hydro_op1_Fortran_3D.json")
        self.__Sedov = milhoja.TaskFunctionAssembler.from_milhoja_json(
            "gpu_tf_hydro", tf_call_graph, op_spec_json, self.__logger
        )

    def testDummyArguments(self):
        expected = [
            "hydro_op1_dt",
            "tile_deltas", "tile_hi", "tile_lo",
            "CC_1",
            "hydro_op1_auxC",
            "hydro_op1_flX", "hydro_op1_flY", "hydro_op1_flZ"
        ]
        self.assertEqual(expected, self.__Sedov.dummy_arguments)

    def testTileMetadata(self):
        expected = {"tile_lo", "tile_hi", "tile_deltas"}
        self.assertEqual(expected, self.__Sedov.tile_metadata_arguments)

    def testGridData(self):
        result = self.__Sedov.grid_data_structures
        self.assertEqual(1, len(result))
        self.assertTrue("CENTER" in result)
        self.assertEqual(set([1]), result["CENTER"])

    def testScratchArguments(self):
        expected = {
            "hydro_op1_auxC",
            "hydro_op1_flX", "hydro_op1_flY", "hydro_op1_flZ"
        }
        self.assertEqual(expected, self.__Sedov.scratch_arguments)

    def testExternalArguments(self):
        expected = {"hydro_op1_dt"}
        self.assertEqual(expected, self.__Sedov.external_arguments)

    def testRuntimeCpu(self):
        OVERWRITE = False
        EXPECTED = [
            "cpu_tf_ic.json",
            "cpu_tf_dens.json", "cpu_tf_ener.json", "cpu_tf_fused.json",
            "cpu_tf_analysis.json"
        ]

        # Start clean
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.makedirs(self.__dst)

        for each in EXPECTED:
            filename = self.__dst.joinpath(each)
            self.assertFalse(filename.exists())

        milhoja.tests.generate_runtime_cpu_tf_specs(
            _RUNTIME_PATH, self.__dst,
            OVERWRITE, self.__logger
        )

        for each in EXPECTED:
            filename = self.__dst.joinpath(each)
            self.assertTrue(filename.is_file())

            ref_fname = "REF_" + each
            with open(_RUNTIME_PATH.joinpath(ref_fname), "r") as fptr:
                reference = json.load(fptr)
            with open(filename, "r") as fptr:
                result = json.load(fptr)

            # ----- TEST AT FINER SCALE AS THIS CAN HELP DEBUG FAILURES
            key = "format"
            self.assertTrue(key in reference)
            self.assertTrue(key in result)
            self.assertEqual(reference[key], result[key])

            groups_all = [
                "grid", "task_function", "data_item", "subroutines"
            ]

            self.assertEqual(len(reference), len(result))
            for group in groups_all:
                # print(group)
                # print(expected[group])
                # print(result[group])
                self.assertTrue(group in reference)
                self.assertTrue(group in result)

                self.assertEqual(len(reference[group]), len(result[group]))
                for key in reference[group]:
                    # print(group, key)
                    # print(reference[group][key])
                    # print(result[group][key])
                    self.assertTrue(key in result[group])
                    self.assertEqual(reference[group][key],
                                     result[group][key])

            # ----- DEBUG AT COARSEST SCALE AS INTENDED
            self.assertEqual(reference, result)

    def testSedovCpu(self):
        OVERWRITE = False
        EXPECTED = [
            "cpu_tf_ic_{}D.json", "cpu_tf_hydro_{}D.json", "cpu_tf_IQ_{}D.json"
        ]

        # Start clean
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.makedirs(self.__dst)

        for dimension in [1, 2, 3]:
            # Generate all task function specification files
            nxb = 16
            nyb = 16 if dimension >= 2 else 1
            nzb = 16 if dimension == 3 else 1

            for each in EXPECTED:
                filename = self.__dst.joinpath(each.format(dimension))
                self.assertFalse(filename.exists())

            milhoja.tests.generate_sedov_cpu_tf_specs(
                dimension, [nxb, nyb, nzb],
                _SEDOV_PATH, self.__dst,
                OVERWRITE, self.__logger
            )

            for each in EXPECTED:
                filename = self.__dst.joinpath(each.format(dimension))
                self.assertTrue(filename.is_file())

                ref_fname = "REF_" + each.format(dimension)
                with open(_SEDOV_PATH.joinpath(ref_fname), "r") as fptr:
                    reference = json.load(fptr)
                with open(filename, "r") as fptr:
                    result = json.load(fptr)

                # ----- TEST AT FINER SCALE AS THIS CAN HELP DEBUG FAILURES
                key = "format"
                self.assertTrue(key in reference)
                self.assertTrue(key in result)
                self.assertEqual(reference[key], result[key])

                groups_all = [
                    "grid", "task_function", "data_item", "subroutines"
                ]

                self.assertEqual(len(reference), len(result))
                for group in groups_all:
                    # print(group)
                    # print(expected[group])
                    # print(result[group])
                    self.assertTrue(group in reference)
                    self.assertTrue(group in result)

                    self.assertEqual(len(reference[group]), len(result[group]))
                    for key in reference[group]:
                        # print(group, key)
                        # print(reference[group][key])
                        # print(result[group][key])
                        self.assertTrue(key in result[group])
                        self.assertEqual(reference[group][key],
                                         result[group][key])

                # ----- DEBUG AT COARSEST SCALE AS INTENDED
                self.assertEqual(reference, result)

    def testSedovGpu(self):
        # Start clean
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.makedirs(self.__dst)

        for dimension in [1, 2, 3]:
            nxb = 16
            nyb = 16 if dimension >= 2 else 1
            nzb = 16 if dimension == 3 else 1

            filename = f"gpu_tf_hydro_{dimension}D.json"
            tf_spec_fname = self.__dst.joinpath(filename)
            self.assertFalse(tf_spec_fname.exists())
            filename = milhoja.tests.generate_sedov_gpu_tf_specs(
                         dimension, [nxb, nyb, nzb], _SEDOV_PATH,
                         self.__dst, False, self.__logger
                       )
            self.assertEqual(tf_spec_fname, filename)
            self.assertTrue(tf_spec_fname.is_file())

            filename = f"REF_gpu_tf_hydro_{dimension}D.json"
            with open(_SEDOV_PATH.joinpath(filename), "r") as fptr:
                expected = json.load(fptr)
            with open(tf_spec_fname, "r") as fptr:
                result = json.load(fptr)

            # ----- TEST AT FINER SCALE AS THIS CAN HELP DEBUG FAILURES
            key = "format"
            self.assertTrue(key in expected)
            self.assertTrue(key in result)
            self.assertEqual(expected[key], result[key])

            groups_all = ["grid", "task_function", "data_item", "subroutines"]

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
