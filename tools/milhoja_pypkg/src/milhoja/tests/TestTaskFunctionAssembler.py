"""
Automatic unit testing of TaskFunctionAssembler
"""

import os
import json
import unittest

from pathlib import Path

import milhoja

_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_SEDOV_PATH = _DATA_PATH.joinpath("Sedov")


class TestTaskFunctionAssembler(unittest.TestCase):
    def setUp(self):
        self.__json_filename = Path.cwd().joinpath("delete_me.json")
        if self.__json_filename.exists():
            os.remove(self.__json_filename)

        # ----- DEFINE SEDOV TEST ASSEMBLER
        # An application needs to construct each task function by grouping a
        # collection of subroutines to be included in the TF in the form of a
        # "graph".  For the moment, our graphs are highly constrained.
        #
        # Imagine that the applications recipe system has identified this
        # internal call graph for our TF under test.
        tf_call_graph = [
            "Hydro_computeSoundSpeedHll_gpu_oacc",
            [
                "Hydro_computeFluxesHll_X_gpu_oacc",
                "Hydro_computeFluxesHll_Y_gpu_oacc",
                "Hydro_computeFluxesHll_Z_gpu_oacc"
            ],
            "Hydro_updateSolutionHll_gpu_oacc"
        ]

        bridge_json = _SEDOV_PATH.joinpath("Hydro_op1.json")

        # The application would then gather together the specifications for
        # each subroutine to be called internally within the TF.  We imagine
        # that the developers of the subroutines encode each subroutine's
        # specification within Milhoja-JSON format files.
        subroutine_jsons_all = {}
        for node in tf_call_graph:
            if isinstance(node, str):
                subroutines_all = [node]
            else:
                subroutines_all = node.copy()

            for subroutine in subroutines_all:
                subroutine_jsons_all[subroutine] = \
                    _SEDOV_PATH.joinpath(f"{subroutine}.json")

        self.__Sedov = milhoja.TaskFunctionAssembler.from_milhoja_json(
            "gpu_tf_hydro", tf_call_graph, subroutine_jsons_all, bridge_json
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
        expected = {"auxC", "flX", "flY", "flZ"}
        self.assertEqual(expected, self.__Sedov.scratch_arguments)

    def testExternalArguments(self):
        expected = {"dt"}
        self.assertEqual(expected, self.__Sedov.external_arguments)

    def testToMilhojaJson(self):
        OVERWRITE = False

        with open(_SEDOV_PATH.joinpath("gpu_tf_hydro_3D.json"), "r") as fptr:
            expected = json.load(fptr)

        self.assertFalse(self.__json_filename.exists())
        self.__Sedov.to_milhoja_json(self.__json_filename, OVERWRITE)
        self.assertTrue(self.__json_filename.is_file())
        self.maxDiff = None

        with open(self.__json_filename, "r") as fptr:
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

        # ----- CLEAN-UP
        # Clean-up manually here rather than in tearDown so file still exists
        # for inspection if failure detected
        os.remove(self.__json_filename)
