"""
Base class for concrete test cases that test code generators derived from
BaseCodeGenerator.

Since potentially many classes will be derived from this class, this test case
class should *not* include any actual test methods.  Methods in the class
can, however, use the unittest self.assert*() methods.
"""

import os
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
            hdr_depends_on_dim = test["header_dim_dependent"]
            src_depends_on_dim = test["source_dim_dependent"]

            for dim in dims_all:
                json_fname = Path(str(test["json"]).format(dim))
                self.assertTrue(json_fname.is_file())

                generator = create_generator(json_fname)

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

                # ----- CLEAN-UP YA LAZY SLOB!
                os.remove(str(header_filename))
                os.remove(str(source_filename))
