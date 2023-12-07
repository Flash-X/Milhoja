"""
Base class for concrete test cases that test code generators derived from
BaseCodeGenerator.

Since potentially many classes will be derived from this class, this test case
class should *not* include any actual test methods.  Methods in the class
can, however, use the unittest self.assert*() methods.
"""

import os
import json
import shutil
import unittest

from pathlib import Path


class TestCodeGenerators(unittest.TestCase):
    def __load_code(self, filename):
        #
        # Loads the given file, splits each line by words, and strips off all
        # white space.
        #
        # Returns a list of lines in the file with blank lines removed.  Each
        # line is itself a list of the words in that line.
        #

        with open(filename, "r") as fptr:
            lines = fptr.readlines()

        cleaned = []
        for each in lines:
            clean = [e for e in each.strip().split() if e != ""]
            if clean:
                cleaned.append(clean)

        return cleaned

    def set_extents(self, extents_str, dim):
        #
        # Given a string that expresses the extents for an array for a 3D
        # version of its test, return the extents for the test in the given
        # dimension.
        #

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

    def __dimensionalize(self, json_fname_3D, json_fname_XD, dim):
        #
        # Given a file containing the specification of a task function for a 3D
        # test problem, write the specification to a new file for the version
        # of the problem associated with the given dimension.
        #

        with open(json_fname_3D, "r") as fptr:
            json_3D = json.load(fptr)

        json_XD = json_3D.copy()

        json_XD["grid"]["dimension"] = dim
        if dim <= 2:
            json_XD["grid"]["nzb"] = 1
        if dim == 1:
            json_XD["grid"]["nyb"] = 1

        with open(json_fname_XD, "w") as fptr:
            json.dump(json_XD, fptr)

    def run_tests(self, tests_all, dims_all, create_generator):
        #
        # For each test in the Cartesian product of tests_all x dims_all,
        #   * Create a new dimension-specific version of the task function,
        #   * Create a generator object,
        #   * Generate the header and source files,
        #   * Confirm that both files are identical to the given reference
        #     files except for white space and blank lines.
        #

        dst = Path.cwd()

        for test in tests_all:
            json_fname_3D = test["json"]
            hdr_depends_on_dim = test["header_dim_dependent"]
            src_depends_on_dim = test["source_dim_dependent"]

            json_fname_XD = dst.joinpath("tmp.json")

            for dim in dims_all:
                self.assertTrue(not json_fname_XD.exists())
                if hdr_depends_on_dim or src_depends_on_dim:
                    self.__dimensionalize(json_fname_3D, json_fname_XD, dim)
                else:
                    shutil.copy(json_fname_3D, json_fname_XD)
                self.assertTrue(json_fname_XD.is_file())

                generator = create_generator(json_fname_XD)

                # ----- CHECK HEADER AGAINST BASELINE
                if generator.header_filename is not None:
                    header_filename = dst.joinpath(generator.header_filename)
                    self.assertTrue(not header_filename.exists())

                    ref_hdr_fname = test["header"]
                    if hdr_depends_on_dim:
                        ref_hdr_fname = Path(str(ref_hdr_fname).format(dim))

                    generator.generate_header_code(dst, True)
                    self.assertTrue(header_filename.is_file())

                    ref = self.__load_code(ref_hdr_fname)
                    generated = self.__load_code(header_filename)

                    self.assertEqual(len(ref), len(generated))
                    for gen_line, ref_line in zip(generated, ref):
                        self.assertEqual(gen_line, ref_line)

                    # Clean-up
                    os.remove(str(header_filename))

                # ----- CHECK SOURCE AGAINST BASELINE
                source_filename = dst.joinpath(generator.source_filename)
                self.assertTrue(not source_filename.exists())

                ref_src_fname = test["source"]
                if src_depends_on_dim:
                    ref_src_fname = Path(str(ref_src_fname).format(dim))

                generator.generate_source_code(dst, True)
                self.assertTrue(source_filename.is_file())

                print(ref_src_fname)
                print(source_filename)
                ref = self.__load_code(ref_src_fname)
                generated = self.__load_code(source_filename)

                self.assertEqual(len(ref), len(generated))
                for gen_line, ref_line in zip(generated, ref):
                    self.assertEqual(gen_line, ref_line)

                # Clean-up
                os.remove(str(json_fname_XD))
                os.remove(str(source_filename))
