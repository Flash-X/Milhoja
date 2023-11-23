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
    BasicLogger,
    TaskFunctionAssembler
)

_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_SEDOV_PATH = _DATA_PATH.joinpath("Sedov")


class TestTaskFunctionAssembler_FlashX(unittest.TestCase):
    def setUp(self):
        self.__dst = Path.cwd().joinpath("delete_me")
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.makedirs(self.__dst)

        self.__logger = BasicLogger(LOG_LEVEL_NONE)

    def testSedov2D_CpuOnly(self):
        # ----- HARDCODED
        NDIM = 2
        NXB = 8
        NYB = 8
        NZB = 1
        NGUARD = 1

        GRID_SPEC = {
            "dimension":   NDIM,
            "nxb":         NXB,
            "nyb":         NYB,
            "nzb":         NZB,
            "nguardcells": NGUARD
        }

        GRAPH = [
            "Hydro_computeSoundSpeedHll_block_cpu",
            "Hydro_computeFluxesHll_X_block_cpu",
            "Hydro_computeFluxesHll_Y_block_cpu",
            "Hydro_updateSolutionHll_block_cpu",
            "Eos_wrapped"
        ]

        GRID_JSON = self.__dst.joinpath("grid.json")
        GROUP_JSON = _SEDOV_PATH.joinpath("Hydro_op1_FlashX.json")
        GROUP_JSON_XD = self.__dst.joinpath(f"Hydro_op1_FlashX_{NDIM}D.json")

        # ----- DUMP BOILDER PLATE SPECS TO FILE
        self.assertFalse(GRID_JSON.exists())
        with open(GRID_JSON, "w") as fptr:
            json.dump(GRID_SPEC, fptr)
        self.assertTrue(GRID_JSON.is_file())

        # ----- UPDATE GROUP SPEC FOR DIMENSION & DUMP TO FILE
        with open(GROUP_JSON, "r") as fptr:
            group_spec = json.load(fptr)
        spec = group_spec["scratch"]

        spec["_auxC"]["extents"] = f"({NXB+2*NGUARD}, {NYB+2*NGUARD}, 1)"
        spec["_auxC"]["lbound"] = "(tile_lo) - (1, 1, 0)"

        spec["_flX"]["extents"] = f"({NXB+1}, {NYB}, 1, 5)"
        spec["_flY"]["extents"] = f"({NXB}, {NYB+1}, 1, 5)"
        for each in ["_flX", "_flY"]:
            spec[each]["lbound"] = "(tile_lo, 1)"

        spec["_flZ"]["extents"] = "(1, 1, 1, 1)"
        spec["_flZ"]["lbound"] = "(1, 1, 1, 1)"

        self.assertFalse(GROUP_JSON_XD.exists())
        with open(GROUP_JSON_XD, "w") as fptr:
            json.dump(group_spec, fptr)
        self.assertTrue(GROUP_JSON_XD.is_file())

        self.__testSedovCpuOnly(GRAPH, GRID_JSON, GROUP_JSON_XD)

    def testSedov3D_CpuOnly(self):
        # ----- HARDCODED
        NDIM = 3
        NXB = 16
        NYB = 16
        NZB = 16
        NGUARD = 1

        GRID_SPEC = {
            "dimension":   NDIM,
            "nxb":         NXB,
            "nyb":         NYB,
            "nzb":         NZB,
            "nguardcells": NGUARD
        }

        GRAPH = [
            "Hydro_computeSoundSpeedHll_block_cpu",
            [
                "Hydro_computeFluxesHll_X_block_cpu",
                "Hydro_computeFluxesHll_Y_block_cpu",
                "Hydro_computeFluxesHll_Z_block_cpu"
            ],
            "Hydro_updateSolutionHll_block_cpu",
            "Eos_wrapped"
        ]

        GRID_JSON = self.__dst.joinpath("grid.json")
        GROUP_JSON = _SEDOV_PATH.joinpath("Hydro_op1_FlashX.json")
        GROUP_JSON_XD = self.__dst.joinpath(f"Hydro_op1_FlashX_{NDIM}D.json")

        # ----- DUMP BOILDER PLATE SPECS TO FILE
        self.assertFalse(GRID_JSON.exists())
        with open(GRID_JSON, "w") as fptr:
            json.dump(GRID_SPEC, fptr)
        self.assertTrue(GRID_JSON.is_file())

        # ----- UPDATE GROUP SPEC FOR DIMENSION & DUMP TO FILE
        with open(GROUP_JSON, "r") as fptr:
            group_spec = json.load(fptr)

        spec = group_spec["scratch"]

        extents = "({}, {}, {})".format(NXB + 2*NGUARD,
                                        NYB + 2*NGUARD,
                                        NZB + 2*NGUARD)
        spec["_auxC"]["extents"] = extents
        spec["_auxC"]["lbound"] = "(tile_lo) - (1, 1, 1)"

        spec["_flX"]["extents"] = f"({NXB+1}, {NYB}, {NZB}, 5)"
        spec["_flY"]["extents"] = f"({NXB}, {NYB+1}, {NZB}, 5)"
        spec["_flZ"]["extents"] = f"({NXB}, {NYB}, {NZB+1}, 5)"
        for each in ["_flX", "_flY", "_flZ"]:
            spec[each]["lbound"] = "(tile_lo, 1)"

        self.assertFalse(GROUP_JSON_XD.exists())
        with open(GROUP_JSON_XD, "w") as fptr:
            json.dump(group_spec, fptr)
        self.assertTrue(GROUP_JSON_XD.is_file())

        self.__testSedovCpuOnly(GRAPH, GRID_JSON, GROUP_JSON_XD)

    def __testSedovCpuOnly(self, graph, grid_json, group_json):
        # ----- HARDCODED
        NO_OVERWRITE = False

        TF_NAME = "cpu_tf_hydro"

        PARTIAL_TF_SPEC = {
            "task_function": {
                "language": "Fortran",
                "processor": "CPU",
                "cpp_header": f"{TF_NAME}_Cpp2C.h",
                "cpp_source": f"{TF_NAME}_Cpp2C.cxx",
                "c2f_source": f"{TF_NAME}_C2F.F90",
                "fortran_source": f"{TF_NAME}_mod.F90"
            },
            "data_item": {
                "type": "TileWrapper",
                "byte_alignment": -1,
                "header": f"Tile_{TF_NAME}.h",
                "source": f"Tile_{TF_NAME}.cxx"
            }
        }

        PARTIAL_TF_JSON = self.__dst.joinpath(f"{TF_NAME}_partial.json")
        TF_JSON = self.__dst.joinpath(f"{TF_NAME}.json")

        # ----- DUMP BOILER PLATE PARTIAL SPEC TO FILE
        self.assertFalse(PARTIAL_TF_JSON.exists())
        with open(PARTIAL_TF_JSON, "w") as fptr:
            json.dump(PARTIAL_TF_SPEC, fptr)
        self.assertTrue(PARTIAL_TF_JSON.is_file())

        # ----- DETERMINE TF SPEC & DUMP TO FILE
        assembler = TaskFunctionAssembler.from_milhoja_json(
                        TF_NAME,
                        graph, [group_json], grid_json,
                        self.__logger
                    )
        self.assertFalse(TF_JSON.exists())
        assembler.to_milhoja_json(TF_JSON, PARTIAL_TF_JSON, NO_OVERWRITE)
        self.assertTrue(TF_JSON.is_file())

        # TODO: Load result and check against benchmark
