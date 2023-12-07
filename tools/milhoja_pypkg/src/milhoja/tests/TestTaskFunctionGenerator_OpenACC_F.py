"""
Unit test of TaskFunctionGenerator_OpenACC_F class.
"""

from pathlib import Path

import milhoja.tests

_FILE_PATH = Path(__file__).resolve().parent
_TEST_PATH = _FILE_PATH.joinpath("data")


def _create_generator(json_filename):
    INDENT = 4

    logger = milhoja.BasicLogger(milhoja.LOG_LEVEL_NONE)
    tf_spec = milhoja.TaskFunction.from_milhoja_json(json_filename)

    return milhoja.TaskFunctionGenerator_OpenACC_F(tf_spec, INDENT, logger)


class TestTaskFunctionGenerator_OpenACC_F(milhoja.tests.TestCodeGenerators):
    def testGeneration(self):
        path = _TEST_PATH.joinpath("Sedov")

        # Only first node in call graph has concurrent kernel launches
        test_1 = {"json": path.joinpath("gpu_tf_test_3D.json"),
                  "header": None,
                  "header_dim_dependent": False,
                  "source": path.joinpath("REF_gpu_tf_test_3D.F90"),
                  "source_dim_dependent": False}
        # Only final node in call graph has concurrent kernel launches
        test_2 = {"json": path.joinpath("gpu_tf_test2_3D.json"),
                  "header": None,
                  "header_dim_dependent": False,
                  "source": path.joinpath("REF_gpu_tf_test2_3D.F90"),
                  "source_dim_dependent": False}

        self.run_tests([test_1, test_2], [3], _create_generator)

    def testSedovGeneration(self):
        path = _TEST_PATH.joinpath("Sedov")

        hydro_2D = {"json": path.joinpath("gpu_tf_hydro_2DF.json"),
                    "header": None,
                    "header_dim_dependent": False,
                    "source": path.joinpath("REF_gpu_tf_hydro_2D.F90"),
                    "source_dim_dependent": False}
        hydroFC_2D = {"json": path.joinpath("gpu_tf_hydroFC_2D.json"),
                      "header": None,
                      "header_dim_dependent": False,
                      "source": path.joinpath("REF_gpu_tf_hydroFC_2D.F90"),
                      "source_dim_dependent": False}
        # Only middle node has concurrent kernel launch
        hydro_3D = {"json": path.joinpath("gpu_tf_hydro_3DF.json"),
                    "header": None,
                    "header_dim_dependent": False,
                    "source": path.joinpath("REF_gpu_tf_hydro_3D.F90"),
                    "source_dim_dependent": False}

        self.run_tests([hydro_3D], [3], _create_generator)
        self.run_tests([hydro_2D, hydroFC_2D], [2], _create_generator)
