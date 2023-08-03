#!/usr/bin/env python

"""
Unit test of TileWrapperGenerator class.
"""

import os
import unittest

from pathlib import Path

import milhoja

_FILE_PATH = Path(__file__).resolve().parent
_TEST_PATH = _FILE_PATH.joinpath("TestData")

class TestTileWrapperGenerator(unittest.TestCase):
    def setUp(self):
        # Pick names that pair appropriately with the files generated in
        # TestCppTaskFunctionGenerator.
        self.__hdr_fname = Path("Tile_delete_me.h").resolve()
        self.__src_fname = Path("Tile_delete_me.cpp").resolve()
        if self.__hdr_fname.exists():
            os.remove(self.__hdr_fname)
        if self.__src_fname.exists():
            os.remove(self.__src_fname)

    def tearDown(self):
        if self.__hdr_fname.exists():
            os.remove(self.__hdr_fname)
        if self.__src_fname.exists():
            os.remove(self.__src_fname)

    def __load_code(self, filename):
        with open(filename, "r") as fptr:
            lines = fptr.readlines()
        
        cleaned = []
        for each in lines:
            clean = [e for e in each.strip().split() if e != ""]
            if clean:
                cleaned.append(clean)

        return cleaned

    def _testConstructionError(self):
        GOOD_FNAME  = _TEST_PATH.joinpath("cpu_tf00_3D.json").resolve()
        GOOD_INDENT_MIN     = 0
        GOOD_INDENT_NOMINAL = 4

        BAD_FNAME = _TEST_PATH.joinpath("AFileWithThisNameMustNotExist.jpeg1234")
        self.assertFalse(BAD_FNAME.exists())
        BAD_INDENT = -1

        # Confirm correct construction first
        # Indent optional
        milhoja.TileWrapperGenerator(GOOD_FNAME)
        milhoja.TileWrapperGenerator(GOOD_FNAME, GOOD_INDENT_MIN)
        milhoja.TileWrapperGenerator(GOOD_FNAME, GOOD_INDENT_NOMINAL)

        # Confirm non-existent file
        try:
            milhoja.TileWrapperGenerator(BAD_FNAME)
            self.assertTrue(False)
        except:
            pass

        # TODO: Add in calls with bad JSON files?

        # Confirm bad indent
        try:
            milhoja.TileWrapperGenerator(GOOD_FNAME, BAD_INDENT)
            self.assertTrue(False)
        except:
            pass

    def testGeneration(self):
        # TODO: We need to test many different JSONs.  Examples
        # - Try the 2D
        # - No scratch
        # - Multiple scratch
        # - No external
        # - Multiple external
        json_fname    = _TEST_PATH.joinpath("cpu_tf00_3D.json").resolve()
        ref_hdr_fname = _TEST_PATH.joinpath("Tile_cpu_tf00_3D.h").resolve()
        ref_src_fname = _TEST_PATH.joinpath("Tile_cpu_tf00_3D.cpp").resolve()

        log_level = milhoja.CodeGenerationLogger.NO_LOGGING_LEVEL
        logger = milhoja.CodeGenerationLogger("TileWrapper Generator", log_level)

        generator = milhoja.TileWrapperGenerator.from_json(json_fname,
                                                           self.__hdr_fname,
                                                           self.__src_fname, 
                                                           logger)

        self.assertTrue(not self.__hdr_fname.exists())
        self.assertTrue(not self.__src_fname.exists())
        generator.generate_header_code()
        self.assertTrue(self.__hdr_fname.is_file())
        generator.generate_source_code()
        self.assertTrue(self.__src_fname.is_file())

        #####----- CHECK HEADER AGAINST BASELINE
        ref       = self.__load_code(ref_hdr_fname)
        generated = self.__load_code(self.__hdr_fname)

        self.assertEqual(len(ref), len(generated))
        for gen_line, ref_line in zip(generated, ref):
            self.assertEqual(gen_line, ref_line)

        #####----- CHECK SOURCE AGAINST BASELINE
        ref       = self.__load_code(ref_src_fname)
        generated = self.__load_code(self.__src_fname)

        self.assertEqual(len(ref), len(generated))
        for gen_line, ref_line in zip(generated, ref):
            self.assertEqual(gen_line, ref_line)

    def testString(self):
        json_fname = _TEST_PATH.joinpath("cpu_tf00_3D.json").resolve()

        log_level = milhoja.CodeGenerationLogger.NO_LOGGING_LEVEL
        logger = milhoja.CodeGenerationLogger("TileWrapper Generator", log_level)

        generator = milhoja.TileWrapperGenerator.from_json(json_fname,
                                                           self.__hdr_fname,
                                                           self.__src_fname,
                                                           logger)
        msg = str(generator)
        self.assertTrue(msg.strip() != "")

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTileWrapperGenerator))

    return suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())

