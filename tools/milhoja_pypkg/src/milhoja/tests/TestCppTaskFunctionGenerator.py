"""
Unit test of CppTaskFunctionGenerator class.
"""

import os
import json
import unittest

from pathlib import Path

import milhoja

_FILE_PATH = Path(__file__).resolve().parent
_TEST_PATH = _FILE_PATH.joinpath("data")


class TestCppTaskFunctionGenerator(unittest.TestCase):
    def setUp(self):
        self.__hdr_fname = Path("delete_me.h").resolve()
        self.__src_fname = Path("delete_me.cpp").resolve()
        if self.__hdr_fname.exists():
            os.remove(self.__hdr_fname)
        if self.__src_fname.exists():
            os.remove(self.__src_fname)

    def tearDown(self):
        if self.__hdr_fname.exists():
            os.remove(self.__hdr_fname)
        if self.__src_fname.exists():
            os.remove(self.__src_fname)

    def __set_extents(self, extents_str, dim):
        self.assertTrue(dim in [1, 2, 3])
        if dim == 3:
            return extents_str

        tmp = extents_str.strip()
        self.assertTrue(tmp.startswith("("))
        self.assertTrue(tmp.endswith(")"))
        tmp = tmp.lstrip("(").rstrip(")")

        tmp = tmp.split(",")
        self.assertEqual(4, len(tmp))

        extents = [int(e) for e in tmp]
    
        if dim <= 2:
            extents[2] = 1
        if dim == 1:
            extents[1] = 1

        return "(" + ", ".join([str(e) for e in extents]) + ")"

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
        GOOD_FNAME = _TEST_PATH.joinpath("cpu_tf00_3D.json").resolve()
        GOOD_INDENT_MIN = 0
        GOOD_INDENT_NOMINAL = 4

        BAD_FNAME = _TEST_PATH.joinpath(
                        "AFileWithThisNameMustNotExist.jpeg1234"
                    )
        self.assertFalse(BAD_FNAME.exists())
        BAD_INDENT = -1

        # Confirm correct construction first
        # Indent optional
        milhoja.CppTaskFunctionGenerator(GOOD_FNAME)
        milhoja.CppTaskFunctionGenerator(GOOD_FNAME, GOOD_INDENT_MIN)
        milhoja.CppTaskFunctionGenerator(GOOD_FNAME, GOOD_INDENT_NOMINAL)

        # Confirm non-existent file
        with self.assertRaises(Exception):
            milhoja.CppTaskFunctionGenerator(BAD_FNAME)

        # TODO: Add in calls with bad JSON files?

        # Confirm bad indent
        with self.assertRaises(Exception):
            milhoja.CppTaskFunctionGenerator(GOOD_FNAME, BAD_INDENT)

    def _testGeneration(self):
        # TODO: We need to test many different JSONs.  Examples
        # - Try the 2D
        # - No scratch
        # - Multiple scratch
        # - No external
        # - Multiple external
        json_fname = _TEST_PATH.joinpath("cpu_tf00_3D.json").resolve()
        ref_hdr_fname = _TEST_PATH.joinpath("cpu_tf00_3D.h").resolve()
        ref_src_fname = _TEST_PATH.joinpath("cpu_tf00_3D.cpp").resolve()

        log_level = milhoja.CodeGenerationLogger.NO_LOGGING_LEVEL
        logger = milhoja.CodeGenerationLogger(
                    "C++ Task Function Generator",
                    log_level
                 )

        generator = milhoja.CppTaskFunctionGenerator.from_json(
                        json_fname,
                        self.__hdr_fname,
                        self.__src_fname,
                        logger
                    )
        self.assertTrue(not self.__hdr_fname.exists())
        self.assertTrue(not self.__src_fname.exists())

        # ----- CHECK HEADER AGAINST BASELINE
        generator.generate_header_code()
        self.assertTrue(self.__hdr_fname.is_file())

        ref = self.__load_code(ref_hdr_fname)
        generated = self.__load_code(self.__hdr_fname)

        self.assertEqual(len(ref), len(generated))
        for gen_line, ref_line in zip(generated, ref):
            self.assertEqual(gen_line, ref_line)

        # ----- CHECK SOURCE AGAINST BASELINE
        generator.generate_source_code()
        self.assertTrue(self.__src_fname.is_file())

        ref = self.__load_code(ref_src_fname)
        generated = self.__load_code(self.__src_fname)

        self.assertEqual(len(ref), len(generated))
        for gen_line, ref_line in zip(generated, ref):
            self.assertEqual(gen_line, ref_line)

    def testSedovGeneration(self):
        path = _TEST_PATH.joinpath("Sedov")

        log_level = milhoja.CodeGenerationLogger.NO_LOGGING_LEVEL
        logger = milhoja.CodeGenerationLogger(
                    "C++ Task Function Generator",
                    log_level
                 )

        for task_function in ["IQ"]:
            json_fname_3D = path.joinpath(f"cpu_tf_{task_function}_3D.json").resolve()
            ref_hdr_fname = path.joinpath(f"REF_cpu_tf_{task_function}.h").resolve()
            ref_src_fname = path.joinpath(f"REF_cpu_tf_{task_function}.cpp").resolve()
            json_fname_XD = Path(f"cpu_{task_function}.json")
            new_hdr_fname = Path(f"cpu_tf_IQ.h")
            new_src_fname = Path(f"cpu_tf_IQ.cpp")

            with open(json_fname_3D, "r") as fptr:
                json_3D = json.load(fptr)
            specs = json_3D["argument_specifications"]["CC_1"]
            extents_in_3D = specs["extents_in"]
            extents_out_3D = specs["extents_out"]

            for dim in [1, 2, 3]:
                extents_in_XD = self.__set_extents(extents_in_3D, dim)
                extents_out_XD = self.__set_extents(extents_out_3D, dim)

                json_XD = json_3D.copy()
                json_3D["argument_specifications"]["CC_1"]["extents_in"] \
                    = extents_in_XD
                json_3D["argument_specifications"]["CC_1"]["extents_out"] \
                    = extents_out_XD

                with open(json_fname_XD, "w") as fptr:
                    json.dump(json_XD, fptr)
                self.assertTrue(json_fname_XD.is_file())

                generator = milhoja.CppTaskFunctionGenerator.from_json(
                                json_fname_XD,
                                new_hdr_fname,
                                new_src_fname,
                                logger
                            )
                self.assertTrue(not new_hdr_fname.exists())
                self.assertTrue(not new_src_fname.exists())

                # ----- CHECK HEADER AGAINST BASELINE
                generator.generate_header_code()
                self.assertTrue(new_hdr_fname.is_file())

                ref = self.__load_code(ref_hdr_fname)
                generated = self.__load_code(new_hdr_fname)

                self.assertEqual(len(ref), len(generated))
                for gen_line, ref_line in zip(generated, ref):
                    self.assertEqual(gen_line, ref_line)

                # ----- CHECK SOURCE AGAINST BASELINE
                generator.generate_source_code()
                self.assertTrue(new_src_fname.is_file())

                ref = self.__load_code(ref_src_fname)
                generated = self.__load_code(new_src_fname)

                self.assertEqual(len(ref), len(generated))
                for gen_line, ref_line in zip(generated, ref):
                    self.assertEqual(gen_line, ref_line)

                # ----- CLEAN-UP YA LAZY SLOB!
                os.remove(str(json_fname_XD))
                os.remove(str(new_hdr_fname))
                os.remove(str(new_src_fname))

    def testString(self):
        json_fname = _TEST_PATH.joinpath("cpu_tf00_3D.json").resolve()

        log_level = milhoja.CodeGenerationLogger.NO_LOGGING_LEVEL
        logger = milhoja.CodeGenerationLogger(
                    "C++ Task Function Generator",
                    log_level
                 )

        generator = milhoja.CppTaskFunctionGenerator.from_json(
                        json_fname,
                        self.__hdr_fname,
                        self.__src_fname,
                        logger
                    )
        msg = str(generator)
        self.assertTrue(msg.strip() != "")
