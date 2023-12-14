"""
Automatic unit testing of TaskFunctionAssembler
"""

import os
import json
import shutil
import unittest

from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from milhoja import (
    LOG_LEVEL_NONE,
    LogicError,
    BasicLogger,
    SubroutineGroup,
    TaskFunctionAssembler
)
from milhoja.tests import (
    NOT_STR_LIST, NOT_LIST_LIST, NOT_BOOL_LIST, NOT_CLASS_LIST,
    generate_runtime_cpu_tf_specs,
    generate_sedov_cpu_tf_specs,
    generate_sedov_gpu_tf_specs,
    generate_flashx_gpu_tf_specs
)


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_SEDOV_PATH = _DATA_PATH.joinpath("Sedov")
_RUNTIME_PATH = _DATA_PATH.joinpath("runtime")
_FLASHX_PATH = _DATA_PATH.joinpath("FlashX")


class TestTaskFunctionAssembler(unittest.TestCase):
    def setUp(self):
        self.__dst = Path.cwd().joinpath("delete_me")
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.makedirs(self.__dst)

        self.__GRID_SPEC = {
            "dimension": 3,
            "nxb": 16,
            "nyb": 16,
            "nzb": 16,
            "nguardcells": 1
        }
        self.__GRID_JSON = self.__dst.joinpath("grid.json")
        with open(self.__GRID_JSON, "w") as fptr:
            json.dump(self.__GRID_SPEC, fptr)

        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        gpu_spec_fname = self.__dst.joinpath("gpu_tf_hydro_3D.json")
        self.assertFalse(gpu_spec_fname.exists())
        block_size = [
            self.__GRID_SPEC["nxb"],
            self.__GRID_SPEC["nyb"],
            self.__GRID_SPEC["nzb"]
        ]
        filename = generate_flashx_gpu_tf_specs(
                     self.__GRID_SPEC["dimension"], block_size,
                     _FLASHX_PATH, self.__dst, False, self.__logger
                   )
        self.assertEqual(gpu_spec_fname, filename)
        self.assertTrue(gpu_spec_fname.is_file())

        with open(gpu_spec_fname, "r") as fptr:
            tf_spec = json.load(fptr)
        self.__call_graph = tf_spec["task_function"]["subroutine_call_graph"]

        self.__group_json = self.__dst.joinpath("Hydro_op1_3D.json")
        self.__Sedov = TaskFunctionAssembler.from_milhoja_json(
            "gpu_tf_hydro", self.__call_graph,
            [self.__group_json], self.__GRID_JSON,
            self.__logger
        )

    def testFromMilhojaJson(self):
        TaskFunctionAssembler.from_milhoja_json(
            "gpu_tf_hydro", self.__call_graph,
            [self.__group_json], self.__GRID_JSON,
            self.__logger
        )

        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                TaskFunctionAssembler.from_milhoja_json(
                    bad, self.__call_graph,
                    [self.__group_json], self.__GRID_JSON,
                    self.__logger
                )

        with self.assertRaises(ValueError):
            TaskFunctionAssembler.from_milhoja_json(
                "", self.__call_graph,
                [self.__group_json], self.__GRID_JSON,
                self.__logger
            )

        # TODO: Bad internal call graphs

        for bad in NOT_LIST_LIST:
            with self.assertRaises(TypeError):
                TaskFunctionAssembler.from_milhoja_json(
                    "gpu_tf_hydro", self.__call_graph,
                    bad, self.__GRID_JSON,
                    self.__logger
                )

        with self.assertRaises(ValueError):
            TaskFunctionAssembler.from_milhoja_json(
                "gpu_tf_hydro", self.__call_graph,
                [], self.__GRID_JSON,
                self.__logger
            )

        for bad in ["", self.__dst.joinpath("nope.h5"), self.__dst]:
            with self.assertRaises(ValueError):
                TaskFunctionAssembler.from_milhoja_json(
                    "gpu_tf_hydro", self.__call_graph,
                    [bad], self.__GRID_JSON,
                    self.__logger
                )

        for bad in NOT_STR_LIST:
            with self.assertRaises(TypeError):
                TaskFunctionAssembler.from_milhoja_json(
                    "gpu_tf_hydro", self.__call_graph,
                    [self.__group_json], bad,
                    self.__logger
                )

        for bad in ["", self.__dst.joinpath("nope.h5"), self.__dst]:
            with self.assertRaises(ValueError):
                TaskFunctionAssembler.from_milhoja_json(
                    "gpu_tf_hydro", self.__call_graph,
                    [self.__group_json], bad,
                    self.__logger
                )

        for bad in NOT_CLASS_LIST:
            with self.assertRaises(TypeError):
                TaskFunctionAssembler.from_milhoja_json(
                    "gpu_tf_hydro", self.__call_graph,
                    [self.__group_json], self.__GRID_JSON,
                    bad
                )

    def testConstruction(self):
        group = SubroutineGroup.from_milhoja_json(self.__group_json,
                                                  self.__logger)
        TaskFunctionAssembler(
            "gpu_tf_hydro", self.__call_graph,
            [group], self.__GRID_SPEC,
            self.__logger
        )

        for bad in NOT_LIST_LIST:
            with self.assertRaises(TypeError):
                TaskFunctionAssembler(
                    "gpu_tf_hydro", self.__call_graph,
                    bad, self.__GRID_SPEC,
                    self.__logger
                )

        with self.assertRaises(TypeError):
            TaskFunctionAssembler(
                "gpu_tf_hydro", self.__call_graph,
                group, self.__GRID_SPEC,
                self.__logger
            )

        # Both groups have same name
        with self.assertRaises(LogicError):
            TaskFunctionAssembler(
                "gpu_tf_hydro", self.__call_graph,
                [group, group], self.__GRID_SPEC,
                self.__logger
            )

        for bad in NOT_CLASS_LIST:
            with self.assertRaises(TypeError):
                TaskFunctionAssembler(
                    "gpu_tf_hydro", self.__call_graph,
                    [bad], self.__GRID_SPEC,
                    self.__logger
                )

        for bad in NOT_CLASS_LIST:
            with self.assertRaises(TypeError):
                TaskFunctionAssembler(
                    "gpu_tf_hydro", self.__call_graph,
                    [group], self.__GRID_SPEC,
                    bad
                )

        good_spec = group.specification
        group_2 = SubroutineGroup(good_spec, self.__logger)
        TaskFunctionAssembler(
            "gpu_tf_hydro", self.__call_graph,
            [group_2], self.__GRID_SPEC,
            self.__logger
        )
        # Two groups containing subroutines with same names
        # Rename to avoid same name error
        good_spec["name"] += "_next!"
        group_2 = SubroutineGroup(good_spec, self.__logger)
        with self.assertRaises(LogicError):
            TaskFunctionAssembler(
                "gpu_tf_hydro", self.__call_graph,
                [group, group_2], self.__GRID_SPEC,
                self.__logger
            )

    def testDummyArguments(self):
        expected = [
            "external_hydro_op1_dt",
            "tile_deltas", "tile_hi", "tile_lo",
            "CC_1",
            "scratch_hydro_op1_auxC",
            "scratch_hydro_op1_flX",
            "scratch_hydro_op1_flY",
            "scratch_hydro_op1_flZ",
            "lbdd_CC_1",
            "lbdd_scratch_hydro_op1_auxC",
            "lbdd_scratch_hydro_op1_flX",
            "lbdd_scratch_hydro_op1_flY",
            "lbdd_scratch_hydro_op1_flZ"
        ]
        self.assertEqual(expected, self.__Sedov.dummy_arguments)

    def testArgumentSpecifications(self):
        with self.assertRaises(ValueError):
            self.__Sedov.argument_specification("HeckNo!")

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
            "scratch_hydro_op1_auxC",
            "scratch_hydro_op1_flX",
            "scratch_hydro_op1_flY",
            "scratch_hydro_op1_flZ"
        }
        self.assertEqual(expected, self.__Sedov.scratch_arguments)

    def testExternalArguments(self):
        expected = {"external_hydro_op1_dt"}
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

        generate_runtime_cpu_tf_specs(
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

            generate_sedov_cpu_tf_specs(
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
            filename = generate_sedov_gpu_tf_specs(
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

    def testToMilhojaJson(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        tf_partial_spec = {
            "task_function": {
                "language":       "Fortran",
                "processor":      "CPU",
                "cpp_header":     "cpu_tf_test_Cpp2C.h",
                "cpp_source":     "cpu_tf_test_Cpp2C.cxx",
                "c2f_source":     "cpu_tf_test_C2F.F90",
                "fortran_source": "cpu_tf_test_mod.F90"
            },
            "data_item": {
                "type":           "TileWrapper",
                "byte_alignment": -1,
                "header":         "Tile_cpu_tf_test.h",
                "source":         "Tile_cpu_tf_test.cxx",
                "module":         ""
            }
        }
        self.assertFalse(TF_PARTIAL_JSON.exists())
        with open(TF_PARTIAL_JSON, "w") as fptr:
            json.dump(tf_partial_spec, fptr)
        self.assertTrue(TF_PARTIAL_JSON.is_file())

        # Confirm correct arguments/functionality
        self.assertFalse(FILENAME.exists())
        self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)
        self.assertTrue(FILENAME.is_file())

        # Confirm we won't overwrite but does warn
        with self.assertRaises(RuntimeError):
            with redirect_stdout(StringIO()) as msg:
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)
        warn = msg.getvalue().strip()
        self.assertTrue("WARNING" in warn)
        self.assertTrue(warn.endswith(f"{FILENAME} already exists"))

        # Confirm will overwrite but still warns
        with redirect_stdout(StringIO()) as msg:
            self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, True)
            warn = msg.getvalue().strip()
        self.assertTrue("WARNING" in warn)
        self.assertTrue(warn.endswith(f"{FILENAME} already exists"))

    def testToMilhojaJsonErrors(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        tf_partial_spec = {
            "task_function": {
                "language":       "Fortran",
                "processor":      "CPU",
                "cpp_header":     "cpu_tf_test_Cpp2C.h",
                "cpp_source":     "cpu_tf_test_Cpp2C.cxx",
                "c2f_source":     "cpu_tf_test_C2F.F90",
                "fortran_source": "cpu_tf_test_mod.F90"
            },
            "data_item": {
                "type":           "TileWrapper",
                "byte_alignment": -1,
                "header":         "Tile_cpu_tf_test.h",
                "source":         "Tile_cpu_tf_test.cxx",
                "module":         ""
            }
        }
        self.assertFalse(TF_PARTIAL_JSON.exists())
        with open(TF_PARTIAL_JSON, "w") as fptr:
            json.dump(tf_partial_spec, fptr)
        self.assertTrue(TF_PARTIAL_JSON.is_file())

        # Confirm correct arguments/functionality
        self.assertFalse(FILENAME.exists())
        self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)
        self.assertTrue(FILENAME.is_file())
        os.remove(FILENAME)

        # Bad output filename
        for bad in NOT_STR_LIST:
            self.assertFalse(FILENAME.exists())
            with self.assertRaises(TypeError):
                self.__Sedov.to_milhoja_json(bad, TF_PARTIAL_JSON, False)
            self.assertFalse(FILENAME.exists())

        # Bad partial TF spec filename
        for bad in NOT_STR_LIST:
            self.assertFalse(FILENAME.exists())
            with self.assertRaises(TypeError):
                self.__Sedov.to_milhoja_json(FILENAME, bad, False)
            self.assertFalse(FILENAME.exists())

        # partial TF spec file already exists
        self.assertFalse(FILENAME.exists())
        with self.assertRaises(ValueError):
            self.__Sedov.to_milhoja_json(FILENAME, FILENAME, False)
        self.assertFalse(FILENAME.exists())

        # Bad overwrite argument
        for bad in NOT_BOOL_LIST:
            self.assertFalse(FILENAME.exists())
            with self.assertRaises(TypeError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, bad)
            self.assertFalse(FILENAME.exists())
