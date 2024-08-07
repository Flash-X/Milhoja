"""
Unit test of TileWrapperGenerator_cpp class.
"""

from pathlib import Path

import milhoja.tests

_FILE_PATH = Path(__file__).resolve().parent
_TEST_PATH = _FILE_PATH.joinpath("data")


def _create_generator(json_filename):
    INDENT = 4

    logger = milhoja.BasicLogger(milhoja.LOG_LEVEL_NONE)
    tf_spec = milhoja.TaskFunction.from_milhoja_json(json_filename)

    return milhoja.TileWrapperGenerator(tf_spec, INDENT, logger)


class TestTileWrapperGenerator_cpp(milhoja.tests.TestCodeGenerators):
    def testRuntimeGeneration(self):
        path = _TEST_PATH.joinpath("runtime")

        ic = {"json": path.joinpath("REF_cpu_tf_ic.json"),
              "header": path.joinpath("REF_Tile_cpu_tf_ic.h"),
              "header_dim_dependent": False,
              "source": path.joinpath("REF_Tile_cpu_tf_ic.cpp"),
              "source_dim_dependent": False}
        dens = {"json": path.joinpath("REF_cpu_tf_dens.json"),
                "header": path.joinpath("REF_Tile_cpu_tf_dens.h"),
                "header_dim_dependent": False,
                "source": path.joinpath("REF_Tile_cpu_tf_dens.cpp"),
                "source_dim_dependent": False}
        ener = {"json": path.joinpath("REF_cpu_tf_ener.json"),
                "header": path.joinpath("REF_Tile_cpu_tf_ener.h"),
                "header_dim_dependent": False,
                "source": path.joinpath("REF_Tile_cpu_tf_ener.cpp"),
                "source_dim_dependent": False}
        fused = {"json": path.joinpath("REF_cpu_tf_fused.json"),
                 "header": path.joinpath("REF_Tile_cpu_tf_fused.h"),
                 "header_dim_dependent": False,
                 "source": path.joinpath("REF_Tile_cpu_tf_fused.cpp"),
                 "source_dim_dependent": False}
        analysis = {"json": path.joinpath("REF_cpu_tf_analysis.json"),
                    "header": path.joinpath("REF_Tile_cpu_tf_analysis.h"),
                    "header_dim_dependent": False,
                    "source": path.joinpath("REF_Tile_cpu_tf_analysis.cpp"),
                    "source_dim_dependent": False}

        tests_all = [ic, dens, ener, fused, analysis]
        self.run_tests(tests_all, [2], _create_generator)

    def testSedovGeneration(self):
        path = _TEST_PATH.joinpath("Sedov")

        ic = {"json": path.joinpath("REF_cpu_tf_ic_{}D.json"),
              "header": path.joinpath("REF_Tile_cpu_tf_ic.h"),
              "header_dim_dependent": False,
              "source": path.joinpath("REF_Tile_cpu_tf_ic.cpp"),
              "source_dim_dependent": False}
        hydro = {"json": path.joinpath("REF_cpu_tf_hydro_{}D.json"),
                 "header": path.joinpath("REF_Tile_cpu_tf_hydro_{}D.h"),
                 "header_dim_dependent": True,
                 "source": path.joinpath("REF_Tile_cpu_tf_hydro.cpp"),
                 "source_dim_dependent": False}
        IQ = {"json": path.joinpath("REF_cpu_tf_IQ_{}D.json"),
              "header": path.joinpath("REF_Tile_cpu_tf_IQ_{}D.h"),
              "header_dim_dependent": True,
              "source": path.joinpath("REF_Tile_cpu_tf_IQ.cpp"),
              "source_dim_dependent": False}

        self.run_tests([ic, hydro, IQ], [1, 2, 3], _create_generator)

    def testSparkGeneration(self):
        path = _TEST_PATH.joinpath("Spark")

        spark_2D = {
            "json": path.joinpath("REF__tf_spec_cpu_taskfn_0.json"),
            "header": path.joinpath("REF_TileWrapper_cpu_taskfn_0.h"),
            "header_dim_dependent": False,
            "source": path.joinpath("REF_TileWrapper_cpu_taskfn_0.cxx"),
            "source_dim_dependent": False
        }

        self.run_tests([spark_2D], [2], _create_generator)

    def testString(self):
        path = _TEST_PATH.joinpath("Sedov")
        json_fname = path.joinpath("REF_cpu_tf_IQ_3D.json").resolve()

        generator = _create_generator(json_fname)

        msg = str(generator)
        self.assertTrue(msg.strip() != "")
