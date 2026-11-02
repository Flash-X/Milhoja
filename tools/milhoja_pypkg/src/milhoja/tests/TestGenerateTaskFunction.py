"""Tests for task-function generator dispatch."""

import copy
import json
import tempfile
import unittest

from pathlib import Path

import milhoja


_FILE_PATH = Path(__file__).resolve().parent
_GPU_SPEC = _FILE_PATH.joinpath("data", "FlashX", "gpu_tf_hydro_2D.json")


class TestGenerateTaskFunction(unittest.TestCase):
    def testCpuDataPacketGeneration(self):
        with open(_GPU_SPEC, "r") as fptr:
            specification = copy.deepcopy(json.load(fptr))

        task_function = specification["task_function"]
        task_function["processor"] = "CPU"
        task_function["computation_offloading"] = ""

        tf_spec = milhoja.TaskFunction(_GPU_SPEC, specification)
        logger = milhoja.BasicLogger(milhoja.LOG_LEVEL_NONE)

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            milhoja.generate_task_function(
                tf_spec, destination, False, 4, logger
            )

            outputs = tf_spec.output_filenames
            generated = [
                outputs[milhoja.TaskFunction.FORTRAN_TF_KEY]["source"],
                outputs[milhoja.TaskFunction.C2F_KEY]["source"],
                outputs[milhoja.TaskFunction.CPP_TF_KEY]["source"],
                outputs[milhoja.TaskFunction.DATA_ITEM_KEY]["module"]
            ]
            for filename in generated:
                self.assertTrue(destination.joinpath(filename).is_file())

            fortran_source = destination.joinpath(generated[0]).read_text()
            self.assertNotIn("use openacc", fortran_source.lower())
            self.assertNotIn("!$acc", fortran_source.lower())
            self.assertNotIn("milhoja::", fortran_source)
            self.assertIn("real(MILHOJA_REAL)", fortran_source)
            self.assertIn("do n = 1, nTiles_d", fortran_source)

            c2f_source = destination.joinpath(generated[1]).read_text()
            self.assertIn("MILHOJA_HOSTMEM_RUNTIME_BACKEND", c2f_source)
            self.assertNotIn("MILHOJA_OPENACC_OFFLOADING", c2f_source)
            self.assertNotIn("use openacc", c2f_source.lower())
            self.assertIn("real(MILHOJA_REAL)", c2f_source)

            cpp_source = destination.joinpath(generated[2]).read_text()
            self.assertIn("MILHOJA_HOSTMEM_RUNTIME_BACKEND", cpp_source)
            self.assertNotIn("MILHOJA_OPENACC_OFFLOADING", cpp_source)
            self.assertNotIn("asynchronousQueue()", cpp_source)

    def testVariableFirstC2FShape(self):
        with open(_GPU_SPEC, "r") as fptr:
            specification = copy.deepcopy(json.load(fptr))

        specification["task_function"]["processor"] = "CPU"
        specification["task_function"]["computation_offloading"] = ""
        specification["data_item"]["data_layout"] = "variable-first"
        tf_spec = milhoja.TaskFunction(_GPU_SPEC, specification)

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            milhoja.generate_task_function(
                tf_spec, destination, False, 4,
                milhoja.BasicLogger(milhoja.LOG_LEVEL_NONE)
            )
            c2f_name = tf_spec.output_filenames[
                milhoja.TaskFunction.C2F_KEY
            ]["source"]
            c2f_source = destination.joinpath(c2f_name).read_text()

        self.assertIn(
            "shape=[9, 16 + 2 * 1 * MILHOJA_K1D, "
            "16 + 2 * 1 * MILHOJA_K2D, "
            "1 + 2 * 1 * MILHOJA_K3D, F_nTiles_h]",
            c2f_source
        )


if __name__ == "__main__":
    unittest.main()
