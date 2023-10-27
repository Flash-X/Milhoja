#!/usr/bin/env python

"""
Unit test of DataPacketGenerator class.
"""

import os
import json

from pathlib import Path

import milhoja.tests
import glob
import difflib
import subprocess
import sys

from pathlib import Path
from DataPacketGenerator import DataPacketGenerator
from packet_generation_utility import Language
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
 
    def setUp(self) -> None:
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
 
 
    def testCpp(self):
        """
        Tests all files in CppTestData.
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

                generator.generate_header_code()
                generator.generate_source_code()

                generated_name_cpp = generator.source_filename
                correct_name_cpp = f'CppTestData/{generator.source_filename}'
                
                # check c++ source code
                with open(generated_name_cpp, 'r') as generated_cpp, \
                open(correct_name_cpp, 'r') as correct:
                    # Test generated files.
                    self.check_generated_files(generated_cpp, correct)
                    
                generated_name_h = generator.header_filename
                correct_name_h = f'CppTestData/{generator.header_filename}'

                # check c++ headers
                with open(generated_name_h, 'r') as generated_h, \
                open(correct_name_h, 'r') as correct:
                    self.check_generated_files(generated_h, correct)

                # TOOD: Generator should generate TaskFunction call
                print("----- Success")

                # clean up generated files if test passes.
                try:
                    os.remove(generated_name_cpp)
                    os.remove(generated_name_h)
                    for file in glob.glob("cg-tpl.*.cpp"):
                        os.remove(file)
                except: 
                    print("Could not find files. Continue.")

    # TODO: Fortran testing with dpg interface
    def testFortran(self):
        """
        Tests all files in FortranTestData.
        """
        # fortran_jsons = glob.glob("FortranTestData/*.json")
        # self.namespace.language = Language.fortran
        # for file in fortran_jsons:
        #     # generate fortran packet and test.
        #     self.namespace.JSON = file
        #     self.log_beginning()
        #     generator = DataPacketGenerator.from_json(self.namespace)

        #     generator.generate_header_code()
        #     generator.generate_source_code()

        #     # check C++ source code when building for fortran
        #     with open(generator.source_filename, 'r') as generated_cpp, \
        #     open(f'FortranTestData/{generator.source_filename}', 'r') as correct:
        #         # Test generated files.
        #         self.check_generated_files(generated_cpp, correct)
            
        #     # check C++ header when building for fortran.
        #     with open(generator.header_filename, 'r') as generated_h, \
        #     open(f'FortranTestData/{generator.header_filename}', 'r') as correct:
        #         self.check_generated_files(generated_h, correct)
 
        #     # TODO: Should cpp2c layers be named something specific? 
        #     #       Let developer choose the name?
        #     # Check C++ to C layer
        #     with open(generator.cpp2c_filename, 'r') as generated, \
        #     open(f'FortranTestData/{generator.cpp2c_filename}', 'r') as correct:
        #         self.check_generated_files(generated, correct)

        #     # Check C to F layer.
        #     with open(generator.c2f_filename, 'r') as generated, \
        #     open(f'FortranTestData/{generator.c2f_filename}', 'r') as correct:
        #         self.check_generated_files(generated, correct)

    def test_updated_json(self):
        ...

