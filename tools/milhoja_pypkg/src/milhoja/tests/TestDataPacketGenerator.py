import os
import json
import milhoja.tests
import glob

from pathlib import Path
from milhoja import LOG_LEVEL_NONE
from milhoja.DataPacketGenerator import DataPacketGenerator
from milhoja.TaskFunction import TaskFunction
from milhoja.BasicLogger import BasicLogger

_FILE_PATH = Path(__file__).resolve().parent
# temporary
_TEST_PATH = _FILE_PATH.joinpath("data")


class TestDataPacketGenerator(milhoja.tests.TestCodeGenerators):
    """
    Unit test of DataPacketGenerator class.

    ..todo::
        * move lbound parser tests into package test.
        * create tests for extents parser.
    """
    # keys for test dictionaries.
    JSON = "json"
    HEADER = "header"
    HDD = "header_dim_dependent"
    SOURCE = "source",
    SDD = "source_dim_dependent"
    SIZES = "sizes"
    FOLDER = "folder"

    def setUp(self):
        # load task function spec
        # TODO: Once this is in milhoja package change path
        self._runtime = [
            {
                self.JSON: _TEST_PATH.joinpath(
                    "runtime",
                    "gpu_tf_dens.json"
                ),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                # temp use summit sizes
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            },
            {
                self.JSON: _TEST_PATH.joinpath(
                    "runtime",
                    "gpu_tf_fused_actions.json"
                ),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            },
            {
                self.JSON: _TEST_PATH.joinpath(
                    "runtime",
                    "gpu_tf_fused_kernels.json"
                ),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            }
        ]

        self._sedov = [
            {
                self.JSON: _TEST_PATH.joinpath(
                    "Sedov",
                    "gpu_tf_hydro_2D.json"
                ),
                self.FOLDER: "Sedov",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            },
            {
                self.JSON: _TEST_PATH.joinpath(
                    "Sedov", 
                    "gpu_tf_hydro_3D.json"
                ),
                self.FOLDER: "Sedov",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            }
        ]

    def tearDown(self):
        pass

    def check_generated_files(self, generated, correct):
        """
        Checks the generated file by comparing it to correct
        with no whitespace.

        TODO: Find a better way to remove whitespaces and compare.
        """
        generated_string = generated.read().replace(' ', '') \
                                    .replace('\n', '').replace('\t', '')
        correct_string = correct.read().replace(' ', '') \
                                .replace('\n', '').replace('\t', '')

        self.assertTrue(
            len(generated_string) == len(correct_string),
            f"Generated length: {len(generated_string)}, "
            f"correct length: {len(correct_string)}"
        )
        self.assertTrue(
            generated_string == correct_string,
            f"Comparison between {generated.name}"
            f"and {correct.name} returned false."
        )

    def testPacketGeneration(self):
        # Tests all files in the test data folder.
        for test_set in [self._runtime, self._sedov]:
            for test in test_set:
                json_path = test[self.JSON]
                sizes = test[self.SIZES]
                self.assertTrue(isinstance(sizes, dict))
                tf_spec = TaskFunction.from_milhoja_json(json_path)
                # use default logging value for now
                logger = BasicLogger(LOG_LEVEL_NONE)

                generator = DataPacketGenerator(
                    tf_spec, 4, logger, sizes, './'
                )
                generator.generate_header_code(overwrite=True)
                generator.generate_source_code(overwrite=True)

                generated_name_cpp = generator.source_filename
                correct_name_cpp = Path(
                    _TEST_PATH,
                    test[self.FOLDER],
                    os.path.basename(generator.source_filename)
                )

                # logger.log(
                #     "TestDataPacketGenerator",
                #     f"Testing {generated_name_cpp}",
                #     LOG_LEVEL_MAX
                # )

                # check c++ source code
                with open(generated_name_cpp, 'r') as generated_cpp:
                    with open(correct_name_cpp, 'r') as correct:
                        # Test generated files.
                        self.check_generated_files(generated_cpp, correct)

                generated_name_h = generator.header_filename
                correct_name_h = Path(
                    _TEST_PATH,
                    test[self.FOLDER],
                    os.path.basename(generator.header_filename)
                )

                # logger.log(
                #     "TestDataPacketGenerator",
                #     f"Testing {generated_name_h}",
                #     LOG_LEVEL_MAX
                # )

                # check c++ headers
                with open(generated_name_h, 'r') as generated_h:
                    with open(correct_name_h, 'r') as correct:
                        self.check_generated_files(generated_h, correct)

                # ..todo::
                #   * Generator should generate TaskFunction call

                # ..todo::
                #       * currently the cpp2c layer is only generated when
                #         using a fortran task function there should be
                #         another "cpp2c layer" that's just for cpp task
                #         functions.
                generated_cpp2c = None
                generated_c2f = None
                if generator.language == "fortran":
                    generated_cpp2c = generator.cpp2c_file
                    correct_cpp2c = Path(
                        _TEST_PATH,
                        test[self.FOLDER],
                        os.path.basename(generator.cpp2c_file)
                    )
                    # logger.log(
                    #     "TestDataPacketGenerator",
                    #     f"Testing {generated_cpp2c}",
                    #     LOG_LEVEL_MAX
                    # )
                    with open(generated_cpp2c, 'r') as generated:
                        with open(correct_cpp2c, 'r') as correct:
                            self.check_generated_files(generated, correct)

                    generated_c2f = generator.c2f_file
                    correct_c2f = Path(
                        _TEST_PATH,
                        test[self.FOLDER],
                        os.path.basename(generator.c2f_file)
                    )
                    # logger.log(
                    #     "TestDataPacketGenerator",
                    #     f"Testing {generated_c2f}", LOG_LEVEL_MAX)
                    with open(generated_c2f, 'r') as generated:
                        with open(correct_c2f, 'r') as correct:
                            self.check_generated_files(generated, correct)

                # clean up generated files if test passes.
                try:
                    os.remove(generated_name_cpp)
                    os.remove(generated_name_h)
                    if generated_cpp2c:
                        os.remove(generated_cpp2c)
                    if generated_c2f:
                        os.remove(generated_c2f)
                    # be careful when cleaning up here
                    for file in glob.glob(
                        str(Path(generator._destination, "cg-tpl.*.cpp"))
                    ):
                        os.remove(file)
                except FileNotFoundError:
                    print("Could not find files. Continue.")
