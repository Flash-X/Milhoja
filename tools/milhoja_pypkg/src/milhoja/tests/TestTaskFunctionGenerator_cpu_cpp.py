"""
Unit test of TaskFunctionGenerator_cpu_cpp class.
"""

from pathlib import Path

import milhoja.tests

_FILE_PATH = Path(__file__).resolve().parent
_TEST_PATH = _FILE_PATH.joinpath("data")


def _create_generator(json_filename):
    INDENT = 4
    tf_spec = milhoja.TaskFunction.from_milhoja_json(json_filename)
    return milhoja.TaskFunctionGenerator_cpu_cpp(
                tf_spec, milhoja.LOG_LEVEL_NONE, INDENT
           )


class TestTaskFunctionGenerator_cpu_cpp(milhoja.tests.TestCodeGenerators):
    def testRuntimeGeneration(self):
        path = _TEST_PATH.joinpath("runtime")

        ic = {"json": path.joinpath("cpu_tf_ic.json"),
              "header": path.joinpath("REF_cpu_tf_ic.h"),
              "header_dim_dependent": False,
              "source": path.joinpath("REF_cpu_tf_ic.cpp"),
              "source_dim_dependent": False}
        dens = {"json": path.joinpath("cpu_tf_dens.json"),
                "header": path.joinpath("REF_cpu_tf_dens.h"),
                "header_dim_dependent": False,
                "source": path.joinpath("REF_cpu_tf_dens.cpp"),
                "source_dim_dependent": False}
        ener = {"json": path.joinpath("cpu_tf_ener.json"),
                "header": path.joinpath("REF_cpu_tf_ener.h"),
                "header_dim_dependent": False,
                "source": path.joinpath("REF_cpu_tf_ener.cpp"),
                "source_dim_dependent": False}
        fused = {"json": path.joinpath("cpu_tf_fused.json"),
                 "header": path.joinpath("REF_cpu_tf_fused.h"),
                 "header_dim_dependent": False,
                 "source": path.joinpath("REF_cpu_tf_fused.cpp"),
                 "source_dim_dependent": False}
        analysis = {"json": path.joinpath("cpu_tf_analysis.json"),
                    "header": path.joinpath("REF_cpu_tf_analysis.h"),
                    "header_dim_dependent": False,
                    "source": path.joinpath("REF_cpu_tf_analysis.cpp"),
                    "source_dim_dependent": False}

        tests_all = [ic, dens, ener, fused, analysis]
        self.run_tests(tests_all, [2], _create_generator)

    def testSedovGeneration(self):
        path = _TEST_PATH.joinpath("Sedov")

        ic = {"json": path.joinpath("cpu_tf_ic_3D.json"),
              "header": path.joinpath("REF_cpu_tf_ic.h"),
              "header_dim_dependent": False,
              "source": path.joinpath("REF_cpu_tf_ic.cpp"),
              "source_dim_dependent": False}
        hydro = {"json": path.joinpath("cpu_tf_hydro_3D.json"),
                 "header": path.joinpath("REF_cpu_tf_hydro.h"),
                 "header_dim_dependent": False,
                 "source": path.joinpath("REF_cpu_tf_hydro.cpp"),
                 "source_dim_dependent": False}
        IQ = {"json": path.joinpath("cpu_tf_IQ_3D.json"),
              "header": path.joinpath("REF_cpu_tf_IQ.h"),
              "header_dim_dependent": False,
              "source": path.joinpath("REF_cpu_tf_IQ.cpp"),
              "source_dim_dependent": False}

        self.run_tests([ic, hydro, IQ], [1, 2, 3], _create_generator)

    def testString(self):
        path = _TEST_PATH.joinpath("Sedov")
        json_fname = path.joinpath("cpu_tf_IQ_3D.json").resolve()

        generator = _create_generator(json_fname)

        msg = str(generator)
        self.assertTrue(msg.strip() != "")
