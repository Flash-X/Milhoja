#!/usr/bin/env python

"""
Unit test of DataPacketGenerator class.
"""

import os
import json

from pathlib import Path

import milhoja.tests
import glob

from pathlib import Path
from DataPacketGenerator import DataPacketGenerator
from milhoja import LOG_LEVEL_MAX
from milhoja import LOG_LEVEL_BASIC
from milhoja import TaskFunction
from milhoja import BasicLogger

_FILE_PATH = Path(__file__).resolve().parent
# temporary
_TEST_PATH = _FILE_PATH.joinpath("CppTestData")

class TestDataPacketGenerator(milhoja.tests.TestCodeGenerators):
    JSON = "json"
    HEADER = "header"
    HDD = "header_dim_dependent"
    SOURCE = "source",
    SDD = "source_dim_dependent"
    SIZES = "sizes"
 
    def setUp(self):
        # load task function spec
        # TODO: Once this is in milhoja package change path
        self._runtime = [
            {
                self.JSON: _TEST_PATH.joinpath("gpu_tf_dens.json"),
                self.HEADER: _TEST_PATH.joinpath("DataPacket_gpu_tf_dens.h"),
                self.HDD: False,
                self.SOURCE: _TEST_PATH.joinpath("DataPacket_gpu_tf_dens.cpp"),
                self.SDD: False,
                self.SIZES: _FILE_PATH.joinpath("sample_jsons", "summit_sizes.json") # temp use summit sizes
            },
            {
                self.JSON: _TEST_PATH.joinpath("gpu_tf_fused_actions.json"),
                self.HEADER: _TEST_PATH.joinpath("DataPacket_gpu_tf_fused_actions.h"),
                self.HDD: False,
                self.SOURCE: _TEST_PATH.joinpath("DataPacket_gpu_tf_fused_actions.cpp"),
                self.SDD: False,
                self.SIZES: _FILE_PATH.joinpath("sample_jsons", "summit_sizes.json")
            },
            {
                self.JSON: _TEST_PATH.joinpath("gpu_tf_fused_kernels.json"),
                self.HEADER: _TEST_PATH.joinpath("DataPacket_gpu_tf_fused_kernels.h"),
                self.HDD: False,
                self.SOURCE: _TEST_PATH.joinpath("DataPacket_gpu_tf_fused_kernels.cpp"),
                self.SDD: False,
                self.SIZES: _FILE_PATH.joinpath("sample_jsons", "summit_sizes.json")
            }
        ]

        self._sedov = [
            {
                self.JSON: _TEST_PATH.joinpath("gpu_tf_hydro_2D.json"),
                self.HEADER: _TEST_PATH.joinpath(""),
                self.HDD: False,
                self.SOURCE: _TEST_PATH.joinpath(""),
                self.SDD: False,
                self.SIZES: _FILE_PATH.joinpath("sample_jsons", "summit_sizes.json") # temp use summit sizes
            },
            {
                self.JSON: _TEST_PATH.joinpath("gpu_tf_hydro_3D.json"),
                self.HEADER: _TEST_PATH.joinpath(""),
                self.HDD: False,
                self.SOURCE: _TEST_PATH.joinpath(""),
                self.SDD: False,
                self.SIZES: _FILE_PATH.joinpath("sample_jsons", "summit_sizes.json")
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
        generated_string = generated.read().replace(' ', '').replace('\n', '').replace('\t', '')
        correct_string = correct.read().replace(' ', '').replace('\n', '').replace('\t', '')

        self.assertTrue(
            len(generated_string) == len(correct_string),
            f"Generated length: {len(generated_string)}, correct length: {len(correct_string)}"
        )
        self.assertTrue(
            generated_string == correct_string,
            f"Comparison between {generated.name} and {correct.name} returned false."
        )
 
    def testPacketGeneration(self):
        """
        Tests all files in the test data folder.
        """
        for test_set in [self._runtime, self._sedov]:
            for test in test_set:
                print(f"""---------------------{test[self.JSON]}---------------------""")

                json_path = test[self.JSON]
                sizes = test[self.SIZES]
                with open(str(sizes), 'r') as sizes_json:
                    sizes = json.load(sizes_json)
                self.assertTrue(isinstance(sizes, dict))
                tf_spec = TaskFunction.from_milhoja_json(json_path)
                # use default logging value for now
                logger = BasicLogger(LOG_LEVEL_MAX)
               
                generator = DataPacketGenerator(tf_spec, 4, logger, sizes, "./templates", './')
                generator.generate_header_code(overwrite=True)
                generator.generate_source_code(overwrite=True)

                generated_name_cpp = generator.source_filename
                correct_name_cpp = Path(_TEST_PATH, os.path.basename(generator.source_filename))

                logger.log("TestDataPacketGenerator", f"Testing {generated_name_cpp}", LOG_LEVEL_MAX)
                
                # check c++ source code
                with open(generated_name_cpp, 'r') as generated_cpp, \
                open(correct_name_cpp, 'r') as correct:
                    # Test generated files.
                    self.check_generated_files(generated_cpp, correct)
                    
                generated_name_h = generator.header_filename
                correct_name_h = Path(_TEST_PATH, os.path.basename(generator.header_filename))

                logger.log("TestDataPacketGenerator", f"Testing {generated_name_h}", LOG_LEVEL_MAX)

                # check c++ headers
                with open(generated_name_h, 'r') as generated_h, \
                open(correct_name_h, 'r') as correct:
                    self.check_generated_files(generated_h, correct)

                # TOOD: Generator should generate TaskFunction call
                print("----- Success")

                # ..todo::
                #       * currently the cpp2c layer is only generated when using a fortran task function
                #         there should be another "cpp2c layer" that's just for cpp task functions.
                generated_cpp2c = None
                generated_c2f = None
                if generator.language == "fortran":
                    generated_cpp2c = generator.cpp2c_file
                    correct_cpp2c = Path(_TEST_PATH, os.path.basename(generator.cpp2c_file))
                    logger.log("TestDataPacketGenerator", f"Testing {generated_cpp2c}", LOG_LEVEL_MAX)
                    with open(generated_cpp2c, 'r') as generated, \
                    open(correct_cpp2c, 'r') as correct:
                        self.check_generated_files(generated, correct)

                    generated_c2f = generator.c2f_file
                    correct_c2f = Path(_TEST_PATH, os.path.basename(generator.c2f_file))
                    logger.log("TestDataPacketGenerator", f"Testing {generated_c2f}", LOG_LEVEL_MAX)
                    with open(generated_c2f, 'r') as generated, \
                    open(correct_c2f, 'r') as correct:
                        self.check_generated_files(generated, correct)

                # clean up generated files if test passes.
                try:
                    os.remove(generated_name_cpp)
                    os.remove(generated_name_h)
                    if generated_cpp2c:
                        os.remove(generated_cpp2c)
                    if generated_c2f:
                        os.remove(generated_c2f)
                    for file in glob.glob(str(Path(generator._destination, "cg-tpl.*.cpp"))): # be careful when cleaning up here
                        os.remove(file)
                except: 
                    print("Could not find files. Continue.")
