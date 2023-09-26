import unittest
import sys
import os
import glob
import difflib
import subprocess
import generate_packet
from packet_generation_utility import Language
from argparse import Namespace

class TestPacketGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.namespace = Namespace(
            language=Language.fortran,
            JSON='',
            sizes='sample_jsons/summit_sizes.json'
        )
        self._CPP2C = "cgkit.cpp2c.cxx"
        self._C2F = "c2f.F90"

        # keys in this dictionary contain a file located within TestData.
        # values are directories where the comparison data is located. 
        # The name of the file to compare to should always be the same name as the 
        # file that the json outputs.
        self.json_list = glob.glob("TestData/*.json")
        print(self.json_list)        
 
        self.code_repo = os.getenv("MILHOJA_CODE_REPO", "")
        if not self.code_repo:
            print("Missing $MILHOJA_CODE_REPO enviornment variable. Abort.")
            exit(-1)
    
    def tearDown(self) -> None:
        pass

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

    def test_all_packet_generation(self):
        for file in self.json_list:
            # generate data packet
            self.namespace.JSON = file
            self.namespace.language = Language.cpp
            self.log_beginning()

            # generate c++ versions first
            generate_packet.generate_packet(self.namespace)
            name = os.path.basename(self.namespace.JSON).replace(".json", "")

            generated_name_cpp = f'cgkit.{name}.cpp'
            correct_name_cpp = f'TestData/cgkit.{name}.cpp'
            
            # check c++ source code
            with open(generated_name_cpp, 'r') as generated_cpp, \
            open(correct_name_cpp, 'r') as correct:
                # Test generated files.
                self.check_generated_files(generated_cpp, correct)
                
            generated_name_h = f'cgkit.{name}.h'
            correct_name_h = f'TestData/cgkit.{name}.h'

            # check c++ headers
            with open(generated_name_h, 'r') as generated_h, \
            open(correct_name_h, 'r') as correct:
                self.check_generated_files(generated_h, correct)

            # generate fortran packet and test.
    #         self.namespace.JSON = 'TestData/DataPacket_Hydro_gpu_3.json'
    #         self.namespace.language = Language.fortran
    #         log = (
    #             f"""
    # ------------------------------------ TEST LOG ------------------------------------
    #             JSON: {self.namespace.JSON}\n
    #             Lang: {self.namespace.language}\n
    #             Size: {self.namespace.sizes}
    #             """
    #         )

    #         generate_packet.generate_packet(self.namespace)
    #         name = os.path.basename(self.namespace.JSON).replace(".json", "")

            # check C++ source code when building for fortran
            # with open(f'cgkit.{name}.cpp', 'r') as generated_cpp, \
            # open(f'{os.getenv("MILHOJA_CODE_REPO")}/test/Sedov/gpu/variant3/cgkit.DataPacket_Hydro_gpu_3.cpp', 'r') as correct:
            #     # Test generated files.
            #     test1 = generated_cpp.read().replace(' ', '').replace('\n', '').replace('\t', '')
            #     test2 = correct.read().replace(' ', '').replace('\n', '').replace('\t', '')
            #     self.assertTrue(len(test1) == len(test2))
            #     self.assertTrue(test1 == test2)

            # # check C++ header when building for fortran.
            # with open(f'cgkit.{name}.h', 'r') as generated_h, \
            # open(f'{os.getenv("MILHOJA_CODE_REPO")}/test/Sedov/gpu/variant3/cgkit.DataPacket_Hydro_gpu_3.h', 'r') as correct:
            #     test1 = generated_h.read().replace(' ', '').replace('\n', '').replace('\t', '')
            #     test2 = correct.read().replace(' ', '').replace('\n', '').replace('\t', '')
            #     self.assertTrue(len(test1) == len(test2))
            #     self.assertTrue(test1 == test2)

            # TODO: Check c2f and cpp2c layers. 
            print("----- Success")

            # clean up generated files if test passes.
            try:
                os.remove(generated_name_cpp)
                os.remove(generated_name_h)
                for file in glob.glob("cg-tpl.*"):
                    os.remove(file)
                os.remove(r"c2f.F90")
                os.remove(r"cgkit.cpp2c.cxx")
            except: 
                print("Could not find files. Continue.")
    

    def check_generated_files(self, generated, correct):
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
    

if __name__ == "__main__":
    unittest.main()
