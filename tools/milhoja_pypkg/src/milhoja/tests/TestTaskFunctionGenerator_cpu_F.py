"""
Unit test of TaskFunctionGenerator_cpu_F class.
"""

from pathlib import Path

import milhoja.tests

_FILE_PATH = Path(__file__).resolve().parent
_TEST_PATH = _FILE_PATH.joinpath("data")


def _create_generator(json_filename):
    INDENT = 4

    logger = milhoja.BasicLogger(milhoja.LOG_LEVEL_NONE)
    tf_spec = milhoja.TaskFunction.from_milhoja_json(json_filename)

    return milhoja.TaskFunctionGenerator_cpu_F(tf_spec, INDENT, logger)


class TestTaskFunctionGenerator_cpu_F(milhoja.tests.TestCodeGenerators):
    def testFlashXGeneration(self):
        fx_path = _TEST_PATH.joinpath("FlashX")
        spark_path = _TEST_PATH.joinpath("Spark")

        # hydro_2D = {"json": fx_path.joinpath("cpu_tf_hydro_2D.json"),
        #             "header": None,
        #             "header_dim_dependent": False,
        #             "source": fx_path.joinpath("REF_cpu_tf_hydro_2D.F90"),
        #             "source_dim_dependent": False}
        # hydroFC_2D = {"json": fx_path.joinpath("cpu_tf_hydroFC_2D.json"),
        #               "header": None,
        #               "header_dim_dependent": False,
        #               "source": fx_path.joinpath("REF_cpu_tf_hydroFC_2D.F90"),
        #               "source_dim_dependent": False}
        # Only middle node has concurrent kernel launch
        hydro_3D = {
            "json": fx_path.joinpath("REF_cpu_tf_hydro_3D.json"),
            "header": None,
            "header_dim_dependent": False,
            "source": fx_path.joinpath("REF_cpu_tf_hydro_3D.F90"),
            "source_dim_dependent": False
        }
        spark_2D = {
            "json": spark_path.joinpath("cpu_taskfn_0.json"),
            "header": None,
            "header_dim_dependent": False,
            "source": spark_path.joinpath("REF_cpu_taskfn_0_mod.F90"),
            "source_dim_dependent": False
        }

        self.run_tests([hydro_3D], [3], _create_generator)
        self.run_tests([spark_2D], [2], _create_generator)
        # self.run_tests([hydro_2D, hydroFC_2D], [2], _create_generator)
