"""
Unit test of TaskFunctionGeneratorCpp2C_cpu_F class.
"""

from pathlib import Path

import milhoja.tests

_FILE_PATH = Path(__file__).resolve().parent
_FLASHX_PATH = _FILE_PATH.joinpath("data", "FlashX")
_SPARK_PATH = _FILE_PATH.joinpath("data", "Spark")


def _create_generator(json_filename):
    INDENT = 4

    logger = milhoja.BasicLogger(milhoja.LOG_LEVEL_NONE)
    tf_spec = milhoja.TaskFunction.from_milhoja_json(json_filename)

    return milhoja.TaskFunctionCpp2CGenerator_cpu_F(tf_spec, INDENT, logger)


class TestTaskFunctionCpp2CGenerator_cpu_F(milhoja.tests.TestCodeGenerators):
    def testSedovGeneration(self):
        hydro_2D = {
            "json": _FLASHX_PATH.joinpath("REF_cpu_tf_hydro_2D.json"),
            "header": None,
            "header_dim_dependent": False,
            "source": _FLASHX_PATH.joinpath("REF_cpu_tf_hydro_2D_Cpp2C.cxx"),
            "source_dim_dependent": False
        }
        hydro_3D = {
            "json": _FLASHX_PATH.joinpath("REF_cpu_tf_hydro_3D.json"),
            "header": None,
            "header_dim_dependent": False,
            "source": _FLASHX_PATH.joinpath("REF_cpu_tf_hydro_3D_Cpp2C.cxx"),
            "source_dim_dependent": False
        }
        spark_2D = {
            "json": _SPARK_PATH.joinpath("REF__tf_spec_cpu_taskfn_0.json"),
            "header": None,
            "header_dim_dependent": False,
            "source": _SPARK_PATH.joinpath("REF_cpu_taskfn_0.cxx"),
            "source_dim_dependent": False
        }

        self.run_tests([hydro_3D], [3], _create_generator)
        self.run_tests([hydro_2D], [2], _create_generator)
        self.run_tests([spark_2D], [2], _create_generator)
