#!/usr/bin/env python

"""
Unit test of WesleyGenerator class.
"""

import os

from pathlib import Path

import milhoja.tests
import glob
import difflib
import subprocess
import sys

from DataPacketGenerator import DataPacketGenerator
from packet_generation_utility import Language
from argparse import Namespace

_FILE_PATH = Path(__file__).resolve().parent
_TEST_PATH = _FILE_PATH.joinpath("data")

class TestDataPacketGenerator(milhoja.tests.TestCodeGenerators):

    def setUp(self) -> None:
        self.namespace = Namespace(
            language=Language.fortran,
            JSON='',
            sizes='sample_jsons/summit_sizes.json'
        )

        self.code_repo = os.getenv("MILHOJA_CODE_REPO", "")
        if not self.code_repo:
            print("Missing $MILHOJA_CODE_REPO enviornment variable. Abort.")
            exit(-1)

    def tearDown(self):
        ...

    def log_beginning(self) -> None:
        log = (
            f"""
------------------------------------ TEST LOG ------------------------------------
            JSON: {self.namespace.JSON}
            Lang: {self.namespace.language}
            Size: {self.namespace.sizes}
----------------------------------------------------------------------------------
            """
        )
        print(log)

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
        cpp_jsons = glob.glob("CppTestData/*.json")

        self.namespace.language = Language.cpp
        for file in cpp_jsons:
            # generate data packet
            self.namespace.JSON = file
            self.log_beginning()
            generator = DataPacketGenerator.from_json(self.namespace)
            name = generator.name

            generator.generate_header_code()
            generator.generate_source_code()

            generated_name_cpp = generator.source_filename
            correct_name_cpp = f'CppTestData/{name}.cpp'
            
            # check c++ source code
            with open(generated_name_cpp, 'r') as generated_cpp, \
            open(correct_name_cpp, 'r') as correct:
                # Test generated files.
                self.check_generated_files(generated_cpp, correct)
                
            generated_name_h = generator.header_filename
            correct_name_h = f'CppTestData/{name}.h'

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
                for file in glob.glob("cg-tpl.*"):
                    os.remove(file)
            except: 
                print("Could not find files. Continue.")
 

    def testFortran(self):
        """
        Tests all files in FortranTestData.
        """
        fortran_jsons = glob.glob("FortranTestData/*.json")
        self.namespace.language = Language.fortran
        for file in fortran_jsons:
            # generate fortran packet and test.
            self.namespace.JSON = file
            self.log_beginning()
            generator = DataPacketGenerator.from_json(self.namespace)
            name = generator.name

            file_names = generator.generate_packet()
            generated_name_cpp = file_names["source"]
            correct_name_cpp = f'FortranTestData/cgkit.{name}.cpp'

            # check C++ source code when building for fortran
            with open(generated_name_cpp, 'r') as generated_cpp, \
            open(correct_name_cpp, 'r') as correct:
                # Test generated files.
                self.check_generated_files(generated_cpp, correct)
            
            # check C++ header when building for fortran.
            with open(f'{name}.h', 'r') as generated_h, \
            open(f'FortranTestData/cgkit.{name}.h', 'r') as correct:
                self.check_generated_files(generated_h, correct)
 
            # TODO: Should cpp2c layers be named something specific? 
            #       Let developer choose the name?
            # Check C++ to C layer
            with open(f'cgkit.cpp2c.cxx', 'r') as generated, \
            open(f'FortranTestData/cpp2c.{name}.cxx', 'r') as correct:
                self.check_generated_files(generated, correct)

            # Check C to F layer.
            with open(f'c2f.F90', 'r') as generated, \
            open(f'FortranTestData/c2f.{name}.F90', 'r') as correct:
                self.check_generated_files(generated, correct)


