"""
Automatic unit testing of the TaskFunction class.
"""

import unittest

from pathlib import Path

import milhoja

_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_RUNTIME_PATH = _DATA_PATH.joinpath("runtime")
_SEDOV_PATH = _DATA_PATH.joinpath("Sedov")


class TestTaskFunction(unittest.TestCase):
    def setUp(self):
        # ----- RUNTIME/CPU TEST
        fname = _RUNTIME_PATH.joinpath("cpu_tf_ic.json")
        self.__rt_ic = milhoja.TaskFunction.from_milhoja_json(fname)

        fname = _RUNTIME_PATH.joinpath("cpu_tf_dens.json")
        self.__rt_dens = milhoja.TaskFunction.from_milhoja_json(fname)

        fname = _RUNTIME_PATH.joinpath("cpu_tf_ener.json")
        self.__rt_ener = milhoja.TaskFunction.from_milhoja_json(fname)

        fname = _RUNTIME_PATH.joinpath("cpu_tf_fused.json")
        self.__rt_fused = milhoja.TaskFunction.from_milhoja_json(fname)

        fname = _RUNTIME_PATH.joinpath("cpu_tf_analysis.json")
        self.__rt_analysis = milhoja.TaskFunction.from_milhoja_json(fname)

        # ----- SEDOV/3D/CPU TEST
        fname = _SEDOV_PATH.joinpath("cpu_tf_ic_3D.json")
        self.__sedov_ic = milhoja.TaskFunction.from_milhoja_json(fname)

        fname = _SEDOV_PATH.joinpath("cpu_tf_hydro_3D.json")
        self.__sedov_hy = milhoja.TaskFunction.from_milhoja_json(fname)

        fname = _SEDOV_PATH.joinpath("cpu_tf_IQ_3D.json")
        self.__sedov_IQ = milhoja.TaskFunction.from_milhoja_json(fname)

    def testConstructorDummyArguments(self):
        tests_all = [self.__rt_ic,
                     self.__rt_dens, self.__rt_ener, self.__rt_fused,
                     self.__rt_analysis,
                     self.__sedov_ic,
                     self.__sedov_IQ]
        for test in tests_all:
            result = test.constructor_dummy_arguments
            self.assertFalse(result)

        tests_all = [
            self.__sedov_hy
        ]
        expected = [("dt", "milhoja::Real")]
        for test in tests_all:
            result = test.constructor_dummy_arguments
            self.assertEqual(expected, result)

    def testTileMetadataArguments(self):
        # ------ OFFICIAL TILE METADATA KEYS
        GID = "tile_gridIndex"
        LEVEL = "tile_level"
        DELTAS = "tile_deltas"
        LO = "tile_lo"
        HI = "tile_hi"
        LBOUND = "tile_lbound"
        UBOUND = "tile_ubound"
        COORDS = "tile_coordinates"
        AREAS = "tile_faceAreas"
        VOLUMES = "tile_cellVolumes"

        # ------ RUNTIME/CPU
        tests_all = [
            self.__rt_ic
        ]
        expected = {
            LBOUND: ["tile_lbound"], UBOUND: ["tile_ubound"],
            COORDS: ["tile_xCenters", "tile_yCenters"]
        }
        for test in tests_all:
            result = test.tile_metadata_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__rt_dens, self.__rt_ener, self.__rt_fused
        ]
        expected = {
            LO: ["tile_lo"], HI: ["tile_hi"], DELTAS: ["tile_deltas"]
        }
        for test in tests_all:
            result = test.tile_metadata_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__rt_analysis
        ]
        expected = {
            GID: ["tile_gridIndex"],
            LO: ["tile_lo"], HI: ["tile_hi"],
            COORDS: ["tile_xCenters", "tile_yCenters"]
        }
        for test in tests_all:
            result = test.tile_metadata_arguments
            self.assertEqual(expected, result)

        # ------ SEDOV/3D/CPU
        tests_all = [
            self.__sedov_ic
        ]
        expected = {
            LEVEL: ["tile_level"],
            LBOUND: ["tile_lbound"], UBOUND: ["tile_ubound"],
            DELTAS: ["tile_deltas"],
            COORDS: ["tile_xCenters", "tile_yCenters", "tile_zCenters"]
        }
        for test in tests_all:
            result = test.tile_metadata_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__sedov_hy
        ]
        expected = {
            DELTAS: ["tile_deltas"],
            LO: ["tile_lo"], HI: ["tile_hi"]
        }
        for test in tests_all:
            result = test.tile_metadata_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__sedov_IQ
        ]
        expected = {
            LO: ["tile_lo"], HI: ["tile_hi"],
            VOLUMES: ["tile_cellVolumes"]
        }
        for test in tests_all:
            result = test.tile_metadata_arguments
            self.assertEqual(expected, result)

    def testExternalArguments(self):
        tests_all = [
            self.__rt_ic,
            self.__rt_dens, self.__rt_ener, self.__rt_fused,
            self.__rt_analysis,
            self.__sedov_ic,
            self.__sedov_IQ
        ]
        expected = set()
        for test in tests_all:
            result = test.external_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__sedov_hy
        ]
        expected = set(["dt"])
        for test in tests_all:
            result = test.external_arguments
            self.assertEqual(expected, result)

    def testScratchArguments(self):
        tests_all = [
            self.__rt_ic, self.__rt_analysis,
            self.__sedov_ic, self.__sedov_IQ
        ]
        expected = set()
        for test in tests_all:
            result = test.scratch_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__rt_dens, self.__rt_ener, self.__rt_fused
        ]
        expected = set(["base_op1_scratch"])
        for test in tests_all:
            result = test.scratch_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__sedov_hy
        ]
        expected = set(["hydro_op1_auxc"])
        for test in tests_all:
            result = test.scratch_arguments
            self.assertEqual(expected, result)

    def testTileInArguments(self):
        tests_all = [
            self.__rt_ic,
            self.__rt_dens, self.__rt_ener, self.__rt_fused,
            self.__sedov_ic,
            self.__sedov_hy
        ]
        expected = set()
        for test in tests_all:
            result = test.tile_in_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__rt_analysis,
            self.__sedov_IQ
        ]
        expected = set(["CC_1"])
        for test in tests_all:
            result = test.tile_in_arguments
            self.assertEqual(expected, result)

    def testTileInOutArguments(self):
        tests_all = [
            self.__rt_ic,
            self.__rt_analysis,
            self.__sedov_ic,
            self.__sedov_IQ
        ]
        expected = set()
        for test in tests_all:
            result = test.tile_in_out_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__rt_dens, self.__rt_ener, self.__rt_fused,
            self.__sedov_hy
        ]
        expected = set(["CC_1"])
        for test in tests_all:
            result = test.tile_in_out_arguments
            self.assertEqual(expected, result)

    def testTileOutArguments(self):
        tests_all = [
            self.__rt_dens, self.__rt_ener, self.__rt_fused,
            self.__rt_analysis,
            self.__sedov_IQ
        ]
        expected = set()
        for test in tests_all:
            result = test.tile_out_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__rt_ic,
            self.__sedov_ic
        ]
        expected = set(["CC_1"])
        for test in tests_all:
            result = test.tile_out_arguments
            self.assertEqual(expected, result)

        tests_all = [
            self.__sedov_hy
        ]
        expected = set(["FLX_1", "FLY_1", "FLZ_1"])
        for test in tests_all:
            result = test.tile_out_arguments
            self.assertEqual(expected, result)
