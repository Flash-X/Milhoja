"""
Automatic testing of creating a single TF specification from two subroutine
groups.  This exercises both the TaskFunction and TaskFuncionAssembler classes
using pathological subroutine group specifications that hopefully stress test
the design and implementation of the class better than real use cases would.
"""

import os
import json
import copy
import shutil
import unittest

from pathlib import Path

from milhoja import (
    LOG_LEVEL_NONE,
    GRID_DATA_ARGUMENT,
    EXTERNAL_ARGUMENT,
    SCRATCH_ARGUMENT,
    TILE_GRID_INDEX_ARGUMENT,
    TILE_LEVEL_ARGUMENT,
    TILE_DELTAS_ARGUMENT,
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT,
    TILE_COORDINATES_ARGUMENT,
    TILE_FACE_AREAS_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT,
    TILE_ARGUMENTS_ALL,
    BasicLogger,
    SubroutineGroup,
    TaskFunction,
    TaskFunctionAssembler,
)


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_FAKE_PATH = _DATA_PATH.joinpath("fake")


class TestMultipleSubroutineGroups(unittest.TestCase):
    def setUp(self):
        self.__GRID_SPEC = {
            "dimension": 2,
            "nxb": 8,
            "nyb": 16,
            "nzb": 1,
            "nguardcells": 1
        }

        # Don't remove on tearDown() so that users can check the results of the
        # last test at the very least.
        self.__dst = Path.cwd().joinpath("delete_me")
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.mkdir(self.__dst)

        TF_NAME = "TestCombined"
        GROUP1_FILENAME = _FAKE_PATH.joinpath("FakeA_op1.json").resolve()
        GROUP2_FILENAME = _FAKE_PATH.joinpath("FakeB_op2.json").resolve()
        INTERNAL_CALL_GRAPH = ["functionC", "functionA"]
        TF_PARTIAL_SPEC = {
            "task_function": {
                "language": "C++",
                "processor": "CPU",
                "cpp_header": "cpu_tf_combined.h",
                "cpp_source": "cpu_tf_combined.cpp",
                "c2f_source": "",
                "fortran_source": ""
            },
            "data_item": {
                "type": "TileWrapper",
                "byte_alignment": -1,
                "header": "Tile_cpu_tf_combined.h",
                "source": "Tile_cpu_tf_combined.cpp"
            }
        }
        TF_PARTIAL_SPEC_FILENAME = \
            self.__dst.joinpath("cpu_tf_combined_partial.json")
        TF_SPEC_FILENAME = self.__dst.joinpath("cpu_tf_combined.json")

        NO_OVERWRITE = False

        with open(TF_PARTIAL_SPEC_FILENAME, "w") as fptr:
            json.dump(TF_PARTIAL_SPEC, fptr)

        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        group_1 = SubroutineGroup.from_milhoja_json(GROUP1_FILENAME,
                                                    self.__logger)
        group_2 = SubroutineGroup.from_milhoja_json(GROUP2_FILENAME,
                                                    self.__logger)
        groups = [group_1, group_2]

        self.assertTrue(not TF_SPEC_FILENAME.exists())
        assembler = TaskFunctionAssembler(
                TF_NAME, INTERNAL_CALL_GRAPH,
                groups, self.__GRID_SPEC,
                self.__logger
            )
        assembler.to_milhoja_json(
            TF_SPEC_FILENAME, TF_PARTIAL_SPEC_FILENAME, NO_OVERWRITE
        )
        self.assertTrue(TF_SPEC_FILENAME.is_file())

        self.__tf_spec = TaskFunction.from_milhoja_json(
            TF_SPEC_FILENAME
        )

        dummy = self.__tf_spec.subroutine_dummy_arguments("functionA")
        actual = self.__tf_spec.subroutine_actual_arguments("functionA")
        self.__fcnA = dict(zip(actual, dummy))

        dummy = self.__tf_spec.subroutine_dummy_arguments("functionC")
        actual = self.__tf_spec.subroutine_actual_arguments("functionC")
        self.__fcnC = dict(zip(actual, dummy))

    def testSameVariableIndexBase(self):
        GROUP1_FNAME = _FAKE_PATH.joinpath("FakeA_op1.json").resolve()
        GROUP2_FNAME = _FAKE_PATH.joinpath("FakeB_op2.json").resolve()
        INTERNAL_CALL_GRAPH = ["functionC", "functionA"]

        group_1 = SubroutineGroup.from_milhoja_json(GROUP1_FNAME, self.__logger)
        group_2 = SubroutineGroup.from_milhoja_json(GROUP2_FNAME, self.__logger)
        groups_all = [group_1, group_2]

        group_spec_1 = group_1.specification
        group_spec_2 = group_2.specification
        self.assertEqual(1, group_spec_1["variable_index_base"])
        self.assertEqual(1, group_spec_2["variable_index_base"])
        TaskFunctionAssembler("DifferentBase", INTERNAL_CALL_GRAPH,
                              groups_all, self.__GRID_SPEC, self.__logger)

        bad_spec = copy.deepcopy(group_spec_1)
        bad_spec["variable_index_base"] = 0
        bad_group = SubroutineGroup(bad_spec, self.__logger)
        with self.assertRaises(NotImplementedError):
            TaskFunctionAssembler("DifferentBase", INTERNAL_CALL_GRAPH,
                                  [bad_group, group_2], self.__GRID_SPEC,
                                  self.__logger)

    def testTileArguments(self):
        # TODO: Replace some of this with code that updates the lo/hi and
        # confirms correctness?
        result = self.__tf_spec.tile_metadata_arguments
        self.assertEqual(12, len(result))

        # ----- INCLUDED ONLY FROM FUNCTIONA
        singletonsA = {TILE_GRID_INDEX_ARGUMENT,
                       TILE_LO_ARGUMENT, TILE_INTERIOR_ARGUMENT}
        for key in singletonsA:
            self.assertTrue(key in result)
            # TF dummy variable name same as argument source
            self.assertEqual([key], result[key])
            self.assertTrue(key in self.__fcnA)
            self.assertTrue(key not in self.__fcnC)

        expected = [
            ("gid", TILE_GRID_INDEX_ARGUMENT),
            ("lo", TILE_LO_ARGUMENT),
            ("interior", TILE_INTERIOR_ARGUMENT)
        ]
        for dummy, actual in expected:
            self.assertEqual(dummy, self.__fcnA[actual])

        # ----- INCLUDED ONLY FROM FUNCTIONC
        singletonsC = {TILE_LEVEL_ARGUMENT,
                       TILE_LBOUND_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT}
        for key in singletonsC:
            self.assertTrue(key in result)
            # TF dummy variable name same as argument source
            self.assertEqual([key], result[key])
            self.assertTrue(key not in self.__fcnA)
            self.assertTrue(key in self.__fcnC)

        expected = [
            ("level", TILE_LEVEL_ARGUMENT),
            ("loGC", TILE_LBOUND_ARGUMENT),
            ("arrayBdds", TILE_ARRAY_BOUNDS_ARGUMENT)
        ]
        for dummy, actual in expected:
            self.assertEqual(dummy, self.__fcnC[actual])

        # ----- INCLUDED FORM BOTH
        # Singletons
        singletonsBoth = {TILE_HI_ARGUMENT, TILE_UBOUND_ARGUMENT,
                          TILE_DELTAS_ARGUMENT}
        for key in singletonsBoth:
            self.assertTrue(key in result)
            # TF dummy variable name same as argument source
            self.assertEqual([key], result[key])
            self.assertTrue(key in self.__fcnA)
            self.assertTrue(key in self.__fcnC)

        expected = [
            ("hi", TILE_HI_ARGUMENT),
            ("ubdd", TILE_UBOUND_ARGUMENT),
            ("deltas", TILE_DELTAS_ARGUMENT)
        ]
        for dummy, actual in expected:
            self.assertEqual(dummy, self.__fcnA[actual])

        expected = [
            ("hi", TILE_HI_ARGUMENT),
            ("hiGC", TILE_UBOUND_ARGUMENT),
            ("deltas", TILE_DELTAS_ARGUMENT)
        ]
        for dummy, actual in expected:
            self.assertEqual(dummy, self.__fcnC[actual])

        # Coordinates are complicated ...
        expected = {"tile_xCoords_center",
                    "tile_yCoords_left",
                    "tile_zCoords_right"}
        self.assertTrue(TILE_COORDINATES_ARGUMENT in result)
        self.assertEqual(expected, set(result[TILE_COORDINATES_ARGUMENT]))
        for actual in expected:
            self.assertTrue(key in self.__fcnA)
            self.assertTrue(key in self.__fcnC)
        self.assertEqual("xCoords", self.__fcnA["tile_xCoords_center"])
        self.assertEqual("xCenters", self.__fcnC["tile_xCoords_center"])
        self.assertEqual("yCoords", self.__fcnA["tile_yCoords_left"])
        self.assertEqual("yLefts", self.__fcnC["tile_yCoords_left"])
        self.assertEqual("zCoords", self.__fcnA["tile_zCoords_right"])
        self.assertEqual("zRights", self.__fcnC["tile_zCoords_right"])

        expected = {
            "source": TILE_COORDINATES_ARGUMENT,
            "axis": "I",
            "edge": "center",
            "lo": TILE_LBOUND_ARGUMENT,
            "hi": TILE_UBOUND_ARGUMENT,
        }
        arg_spec = self.__tf_spec.argument_specification("tile_xCoords_center")
        self.assertEqual(expected, arg_spec)

        expected = {
            "source": TILE_COORDINATES_ARGUMENT,
            "axis": "J",
            "edge": "left",
            "lo": TILE_LBOUND_ARGUMENT,
            "hi": TILE_UBOUND_ARGUMENT,
        }
        arg_spec = self.__tf_spec.argument_specification("tile_yCoords_left")
        self.assertEqual(expected, arg_spec)

        expected = {
            "source": TILE_COORDINATES_ARGUMENT,
            "axis": "K",
            "edge": "right",
            "lo": TILE_LO_ARGUMENT,
            "hi": TILE_UBOUND_ARGUMENT,
        }
        arg_spec = self.__tf_spec.argument_specification("tile_zCoords_right")
        self.assertEqual(expected, arg_spec)

        # Face areas less so ...
        expected = {"tile_xFaceAreas",
                    "tile_yFaceAreas",
                    "tile_zFaceAreas"}
        self.assertTrue(TILE_FACE_AREAS_ARGUMENT in result)
        self.assertEqual(expected, set(result[TILE_FACE_AREAS_ARGUMENT]))
        for key in expected:
            self.assertTrue(key in self.__fcnA)
            self.assertTrue(key in self.__fcnC)
        self.assertEqual("xAreas", self.__fcnA["tile_xFaceAreas"])
        self.assertEqual("xFaces", self.__fcnC["tile_xFaceAreas"])
        self.assertEqual("yAreas", self.__fcnA["tile_yFaceAreas"])
        self.assertEqual("yFaces", self.__fcnC["tile_yFaceAreas"])
        self.assertEqual("zAreas", self.__fcnA["tile_zFaceAreas"])
        self.assertEqual("zFaces", self.__fcnC["tile_zFaceAreas"])

        expected = {
            "source": TILE_FACE_AREAS_ARGUMENT,
            "axis": "I",
            "lo": TILE_LBOUND_ARGUMENT,
            "hi": TILE_HI_ARGUMENT,
        }
        arg_spec = self.__tf_spec.argument_specification("tile_xFaceAreas")
        self.assertEqual(expected, arg_spec)

        expected = {
            "source": TILE_FACE_AREAS_ARGUMENT,
            "axis": "J",
            "lo": TILE_LO_ARGUMENT,
            "hi": TILE_HI_ARGUMENT,
        }
        arg_spec = self.__tf_spec.argument_specification("tile_yFaceAreas")
        self.assertEqual(expected, arg_spec)

        expected = {
            "source": TILE_FACE_AREAS_ARGUMENT,
            "axis": "K",
            "lo": TILE_LBOUND_ARGUMENT,
            "hi": TILE_UBOUND_ARGUMENT,
        }
        arg_spec = self.__tf_spec.argument_specification("tile_zFaceAreas")
        self.assertEqual(expected, arg_spec)

        # Volumes are a breeze relatively speaking ...
        expected = {TILE_CELL_VOLUMES_ARGUMENT}
        self.assertTrue(TILE_CELL_VOLUMES_ARGUMENT in result)
        self.assertEqual(expected, set(result[TILE_CELL_VOLUMES_ARGUMENT]))
        self.assertTrue(TILE_CELL_VOLUMES_ARGUMENT in self.__fcnA)
        self.assertTrue(TILE_CELL_VOLUMES_ARGUMENT in self.__fcnC)
        self.assertEqual("volumes", self.__fcnA[TILE_CELL_VOLUMES_ARGUMENT])
        self.assertEqual("cellVols", self.__fcnC[TILE_CELL_VOLUMES_ARGUMENT])

        expected = {
            "source": TILE_CELL_VOLUMES_ARGUMENT,
            "lo": TILE_LBOUND_ARGUMENT,
            "hi": TILE_UBOUND_ARGUMENT,
        }
        key = TILE_CELL_VOLUMES_ARGUMENT
        arg_spec = self.__tf_spec.argument_specification(key)
        self.assertEqual(expected, arg_spec)

        # ----- SANITY CHECKS
        others = set([TILE_COORDINATES_ARGUMENT,
                      TILE_FACE_AREAS_ARGUMENT,
                      TILE_CELL_VOLUMES_ARGUMENT])

        self.assertTrue(singletonsA.intersection(singletonsC) == set())
        self.assertTrue(singletonsA.intersection(singletonsBoth) == set())
        self.assertTrue(singletonsC.intersection(singletonsBoth) == set())

        all_in_test = singletonsA.union(singletonsC)
        all_in_test = all_in_test.union(singletonsBoth)
        all_in_test = all_in_test.union(others)
        self.assertTrue(all_in_test == TILE_ARGUMENTS_ALL)

    def testGridDataArguments(self):
        # tile_in
        expected = {"FLX_1"}
        tile_in = self.__tf_spec.tile_in_arguments
        self.assertEqual(expected, tile_in)

        expected = {
            "source": GRID_DATA_ARGUMENT,
            "structure_index": ["FLUXX", 1],
            "variables_in": [1, 3]
        }
        arg_spec = self.__tf_spec.argument_specification("FLX_1")
        self.assertEqual(expected, arg_spec)
        self.assertEqual("flX", self.__fcnA["FLX_1"])
        self.assertEqual("xFlux", self.__fcnC["FLX_1"])

        # tile_in_out
        expected = {"FLZ_1", "CC_1"}
        tile_in_out = self.__tf_spec.tile_in_out_arguments
        self.assertEqual(expected, tile_in_out)

        expected = {
            "source": GRID_DATA_ARGUMENT,
            "structure_index": ["CENTER", 1],
            "variables_in": [2, 6],
            "variables_out": [1, 3]
        }
        arg_spec = self.__tf_spec.argument_specification("CC_1")
        self.assertEqual(expected, arg_spec)
        self.assertEqual("U", self.__fcnA["CC_1"])
        self.assertEqual("solnData", self.__fcnC["CC_1"])

        expected = {
            "source": GRID_DATA_ARGUMENT,
            "structure_index": ["FLUXZ", 1],
            "variables_in": [4, 5],
            "variables_out": [1, 2]
        }
        arg_spec = self.__tf_spec.argument_specification("FLZ_1")
        self.assertEqual(expected, arg_spec)
        self.assertEqual("flZ", self.__fcnA["FLZ_1"])
        self.assertEqual("zFlux", self.__fcnC["FLZ_1"])

        # tile_out
        expected = {"FLY_1"}
        tile_out = self.__tf_spec.tile_out_arguments
        self.assertEqual(expected, tile_out)

        expected = {
            "source": GRID_DATA_ARGUMENT,
            "structure_index": ["FLUXY", 1],
            "variables_out": [1, 5]
        }
        arg_spec = self.__tf_spec.argument_specification("FLY_1")
        self.assertEqual(expected, arg_spec)
        self.assertEqual("flY", self.__fcnA["FLY_1"])
        self.assertEqual("yFlux", self.__fcnC["FLY_1"])

    def testExternalArguments(self):
        expected = {"external_fakeA_op1_dt", "external_fakeB_op2_dt",
                    "external_fakeA_op1_coeffs", "external_fakeB_op2_coeffs"}
        externals = self.__tf_spec.external_arguments
        self.assertEqual(expected, externals)

        # These appear to be the same, but we cannot know that at this level,
        # so they are two variables.
        expected = {
            "source": EXTERNAL_ARGUMENT,
            "type": "milhoja::Real",
            "extents": []
        }
        keyA = "external_fakeA_op1_dt"
        keyC = "external_fakeB_op2_dt"
        arg_spec_A = self.__tf_spec.argument_specification(keyA)
        arg_spec_C = self.__tf_spec.argument_specification(keyC)
        self.assertEqual(expected, arg_spec_A)
        self.assertEqual(arg_spec_A, arg_spec_C)
        self.assertTrue(keyA not in self.__fcnC)
        self.assertTrue(keyC not in self.__fcnA)
        self.assertEqual("dt", self.__fcnA[keyA])
        self.assertEqual("dt", self.__fcnC[keyC])

        # These two have the same global name, but are clearly different
        expected = {
            "source": EXTERNAL_ARGUMENT,
            "type": "milhoja::Real",
            "extents": [2, 3]
        }
        keyA = "external_fakeA_op1_coeffs"
        arg_spec_A = self.__tf_spec.argument_specification(keyA)
        self.assertEqual(expected, arg_spec_A)

        expected = {
            "source": EXTERNAL_ARGUMENT,
            "type": "milhoja::Real",
            "extents": [3]
        }
        keyC = "external_fakeB_op2_coeffs"
        arg_spec_C = self.__tf_spec.argument_specification(keyC)
        self.assertEqual(expected, arg_spec_C)

        self.assertEqual(arg_spec_A["source"], arg_spec_C["source"])
        self.assertEqual(arg_spec_A["type"], arg_spec_C["type"])
        self.assertNotEqual(arg_spec_A["extents"], arg_spec_C["extents"])
        self.assertTrue(keyA not in self.__fcnC)
        self.assertTrue(keyC not in self.__fcnA)
        self.assertEqual("coefficients", self.__fcnA[keyA])
        self.assertEqual("coeffs", self.__fcnC[keyC])

    def testScratchArguments(self):
        expected = {"scratch_fakeA_op1_dt", "scratch_fakeB_op2_dt",
                    "scratch_fakeA_op1_same", "scratch_fakeB_op2_same"}
        scratch = self.__tf_spec.scratch_arguments
        self.assertEqual(expected, scratch)

        # These appear to be the same, but we cannot know that at this level,
        # so they are two variables.
        expected = {
            "source": SCRATCH_ARGUMENT,
            "type": "milhoja::Real",
            "extents": "(4, 8, 16, 1)",
            "lbound": "(-5, tile_lo)"
        }
        keyA = "scratch_fakeA_op1_same"
        keyC = "scratch_fakeB_op2_same"
        arg_spec_A = self.__tf_spec.argument_specification(keyA)
        arg_spec_C = self.__tf_spec.argument_specification(keyC)
        self.assertEqual(expected, arg_spec_A)
        self.assertEqual(arg_spec_A, arg_spec_C)
        self.assertTrue(keyA not in self.__fcnC)
        self.assertTrue(keyC not in self.__fcnA)
        self.assertEqual("scratch4D", self.__fcnA[keyA])
        self.assertEqual("scratch4D", self.__fcnC[keyC])

        # These two have the same global name, but are clearly different.  Note
        # that these have the same global name as a scratch.  These variable
        # names would collide if if weren't for prefixing the variable names
        # with external/scratch.
        expected = {
            "source": SCRATCH_ARGUMENT,
            "type": "milhoja::Real",
            "extents": "(8, 16, 1, 5, 20)",
            "lbound": "(-2, tile_lo, -1)"
        }
        keyA = "scratch_fakeA_op1_dt"
        arg_spec_A = self.__tf_spec.argument_specification(keyA)
        self.assertEqual(expected, arg_spec_A)

        expected = {
            "source": SCRATCH_ARGUMENT,
            "type": "milhoja::Real",
            "extents": "(8, 16, 1, 5, 20)",
            "lbound": "(2, tile_lo, -1)"
        }
        keyC = "scratch_fakeB_op2_dt"
        arg_spec_C = self.__tf_spec.argument_specification(keyC)
        self.assertEqual(expected, arg_spec_C)

        self.assertEqual(arg_spec_A["source"], arg_spec_C["source"])
        self.assertEqual(arg_spec_A["type"], arg_spec_C["type"])
        self.assertEqual(arg_spec_A["extents"], arg_spec_C["extents"])
        self.assertNotEqual(arg_spec_A["lbound"], arg_spec_C["lbound"])
        self.assertTrue(keyA not in self.__fcnC)
        self.assertTrue(keyC not in self.__fcnA)
        self.assertEqual("scratch5D", self.__fcnA[keyA])
        self.assertEqual("scratch5D", self.__fcnC[keyC])
