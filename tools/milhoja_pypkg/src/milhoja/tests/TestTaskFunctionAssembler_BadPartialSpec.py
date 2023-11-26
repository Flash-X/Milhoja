"""
Automatic unit testing of TaskFunctionAssembler
"""

import os
import copy
import json
import shutil
import unittest

from pathlib import Path

from milhoja import (
    LOG_LEVEL_NONE,
    BasicLogger,
    TaskFunctionAssembler
)
from milhoja.tests import (
    NOT_STR_LIST, NOT_INT_LIST,
    generate_sedov_gpu_tf_specs
)


_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_SEDOV_PATH = _DATA_PATH.joinpath("Sedov")


class TestTaskFunctionAssembler_BadPartialSpec(unittest.TestCase):
    def setUp(self):
        self.__dst = Path.cwd().joinpath("delete_me")
        if self.__dst.exists():
            shutil.rmtree(self.__dst)
        os.makedirs(self.__dst)

        GRID_SPEC = {
            "dimension": 3,
            "nxb": 16,
            "nyb": 16,
            "nzb": 16,
            "nguardcells": 1
        }
        GRID_JSON = self.__dst.joinpath("grid.json")
        with open(GRID_JSON, "w") as fptr:
            json.dump(GRID_SPEC, fptr)

        logger = BasicLogger(LOG_LEVEL_NONE)

        gpu_spec_fname = self.__dst.joinpath("gpu_tf_hydro_3D.json")
        self.assertFalse(gpu_spec_fname.exists())
        block_size = [
            GRID_SPEC["nxb"],
            GRID_SPEC["nyb"],
            GRID_SPEC["nzb"]
        ]
        filename = generate_sedov_gpu_tf_specs(
                     GRID_SPEC["dimension"], block_size,
                     _SEDOV_PATH, self.__dst, False, logger
                   )
        self.assertEqual(gpu_spec_fname, filename)
        self.assertTrue(gpu_spec_fname.is_file())

        with open(gpu_spec_fname, "r") as fptr:
            tf_spec = json.load(fptr)
        call_graph = tf_spec["task_function"]["subroutine_call_graph"]

        group_json = self.__dst.joinpath("Hydro_op1_Fortran_3D.json")
        self.__Sedov = TaskFunctionAssembler.from_milhoja_json(
            "gpu_tf_hydro", call_graph, [group_json], GRID_JSON, logger
        )

        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")
        self.__partial = {
            "task_function": {
                "language":       "Fortran",
                "processor":      "CPU",
                "cpp_header":     "cpu_tf_test_Cpp2C.h",
                "cpp_source":     "cpu_tf_test_Cpp2C.cxx",
                "c2f_source":     "cpu_tf_test_C2F.F90",
                "fortran_source": "cpu_tf_test_mod.F90"
            },
            "data_item": {
                "type":           "DataPacket",
                "byte_alignment": 16,
                "header":         "DataPacket_cpu_tf_test.h",
                "source":         "DataPacket_cpu_tf_test.cxx"
            }
        }
        self.assertFalse(TF_PARTIAL_JSON.exists())
        with open(TF_PARTIAL_JSON, "w") as fptr:
            json.dump(self.__partial, fptr)
        self.assertTrue(TF_PARTIAL_JSON.is_file())

        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        self.assertFalse(FILENAME.exists())
        self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)
        self.assertTrue(FILENAME.is_file())
        os.remove(FILENAME)
        os.remove(TF_PARTIAL_JSON)

    def testKeys(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        for each in ["task_function", "data_item"]:
            bad = copy.deepcopy(self.__partial)
            del bad[each]
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

        bad = copy.deepcopy(self.__partial)
        bad["fail"] = {}
        self.assertEqual(3, len(bad))
        with open(TF_PARTIAL_JSON, "w") as fptr:
            json.dump(bad, fptr)
        with self.assertRaises(ValueError):
            self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testTaskFunctionKeys(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        expected = {"language", "processor",
                    "cpp_header", "cpp_source",
                    "c2f_source", "fortran_source"}
        for each in expected:
            bad = copy.deepcopy(self.__partial)
            del bad["task_function"][each]
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

        bad = copy.deepcopy(self.__partial)
        bad["task_function"]["fail"] = {}
        self.assertEqual(7, len(bad["task_function"]))
        with open(TF_PARTIAL_JSON, "w") as fptr:
            json.dump(bad, fptr)
        with self.assertRaises(ValueError):
            self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testDataItemKeys(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        expected = {"type", "byte_alignment", "header", "source"}
        for each in expected:
            bad = copy.deepcopy(self.__partial)
            del bad["data_item"][each]
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

        bad = copy.deepcopy(self.__partial)
        bad["data_item"]["fail"] = {}
        self.assertEqual(5, len(bad["data_item"]))
        with open(TF_PARTIAL_JSON, "w") as fptr:
            json.dump(bad, fptr)
        with self.assertRaises(ValueError):
            self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testLanguage(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        # Can't write sets to JSON
        for bad in NOT_STR_LIST:
            if not isinstance(bad, set):
                bad_spec = copy.deepcopy(self.__partial)
                bad_spec["task_function"]["language"] = bad
                with open(TF_PARTIAL_JSON, "w") as fptr:
                    json.dump(bad_spec, fptr)
                with self.assertRaises(TypeError):
                    self.__Sedov.to_milhoja_json(
                        FILENAME, TF_PARTIAL_JSON, False
                    )

        for bad in ["fail", "Frtran", "Cpp", ""]:
            bad_spec = copy.deepcopy(self.__partial)
            bad_spec["task_function"]["language"] = bad
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad_spec, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testProcessor(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        # Can't write sets to JSON
        for bad in NOT_STR_LIST:
            if not isinstance(bad, set):
                bad_spec = copy.deepcopy(self.__partial)
                bad_spec["task_function"]["processor"] = bad
                with open(TF_PARTIAL_JSON, "w") as fptr:
                    json.dump(bad_spec, fptr)
                with self.assertRaises(TypeError):
                    self.__Sedov.to_milhoja_json(
                        FILENAME, TF_PARTIAL_JSON, False
                    )

        for bad in ["fail", "cpUu", "gp", ""]:
            bad_spec = copy.deepcopy(self.__partial)
            bad_spec["task_function"]["processor"] = bad
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad_spec, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testCppFilenames(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        # Can't write sets to JSON
        for key in ["cpp_header", "cpp_source"]:
            for bad in NOT_STR_LIST:
                if not isinstance(bad, set):
                    bad_spec = copy.deepcopy(self.__partial)
                    bad_spec["task_function"][key] = bad
                    with open(TF_PARTIAL_JSON, "w") as fptr:
                        json.dump(bad_spec, fptr)
                    with self.assertRaises(TypeError):
                        self.__Sedov.to_milhoja_json(
                            FILENAME, TF_PARTIAL_JSON, False
                        )

            bad_spec = copy.deepcopy(self.__partial)
            bad_spec["task_function"][key] = ""
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad_spec, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testFortranSources(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        language = self.__partial["task_function"]["language"]
        self.assertEqual("fortran", language.lower())

        # Can't write sets to JSON
        for key in ["c2f_source", "fortran_source"]:
            for bad in NOT_STR_LIST:
                if not isinstance(bad, set):
                    bad_spec = copy.deepcopy(self.__partial)
                    bad_spec["task_function"][key] = bad
                    with open(TF_PARTIAL_JSON, "w") as fptr:
                        json.dump(bad_spec, fptr)
                    with self.assertRaises(TypeError):
                        self.__Sedov.to_milhoja_json(
                            FILENAME, TF_PARTIAL_JSON, False
                        )

            bad_spec = copy.deepcopy(self.__partial)
            bad_spec["task_function"][key] = ""
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad_spec, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testDataItemType(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        # Can't write sets to JSON
        for bad in NOT_STR_LIST:
            if not isinstance(bad, set):
                bad_spec = copy.deepcopy(self.__partial)
                bad_spec["data_item"]["type"] = bad
                with open(TF_PARTIAL_JSON, "w") as fptr:
                    json.dump(bad_spec, fptr)
                with self.assertRaises(TypeError):
                    self.__Sedov.to_milhoja_json(
                        FILENAME, TF_PARTIAL_JSON, False
                    )

        for bad in ["fail", "Tile", "Packet", ""]:
            bad_spec = copy.deepcopy(self.__partial)
            bad_spec["data_item"]["type"] = bad
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad_spec, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testByteAlignment(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        item_type = self.__partial["data_item"]["type"]
        self.assertEqual("datapacket", item_type.lower())

        # Can't write sets to JSON
        for bad in NOT_INT_LIST:
            bad_spec = copy.deepcopy(self.__partial)
            if not isinstance(bad, set):
                bad_spec["data_item"]["byte_alignment"] = bad
                with open(TF_PARTIAL_JSON, "w") as fptr:
                    json.dump(bad_spec, fptr)
                with self.assertRaises(TypeError):
                    self.__Sedov.to_milhoja_json(
                        FILENAME, TF_PARTIAL_JSON, False
                    )

        for bad in range(-5, 1):
            bad_spec["data_item"]["byte_alignment"] = bad
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad_spec, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)

    def testDataItemFilenames(self):
        FILENAME = self.__dst.joinpath("cpu_tf_test.json")
        TF_PARTIAL_JSON = self.__dst.joinpath("cpu_tf_test_partial.json")

        # Can't write sets to JSON
        for key in ["header", "source"]:
            for bad in NOT_STR_LIST:
                if not isinstance(bad, set):
                    bad_spec = copy.deepcopy(self.__partial)
                    bad_spec["data_item"][key] = bad
                    with open(TF_PARTIAL_JSON, "w") as fptr:
                        json.dump(bad_spec, fptr)
                    with self.assertRaises(TypeError):
                        self.__Sedov.to_milhoja_json(
                            FILENAME, TF_PARTIAL_JSON, False
                        )

            bad_spec = copy.deepcopy(self.__partial)
            bad_spec["data_item"][key] = ""
            with open(TF_PARTIAL_JSON, "w") as fptr:
                json.dump(bad_spec, fptr)
            with self.assertRaises(ValueError):
                self.__Sedov.to_milhoja_json(FILENAME, TF_PARTIAL_JSON, False)
