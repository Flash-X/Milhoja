#!/usr/bin/env python

"""

Run the script with -h to obtain more information regarding the script.

"""

import sys
import unittest
import argparse
from pathlib import Path

_FILE_PATH = Path(__file__).resolve().parent
_TEST_PATH = _FILE_PATH.joinpath("tests")
sys.path.append(str(_TEST_PATH))

from TestDataPacketGenerator import TestDataPacketGenerator

# ----- HARDCODED VALUES

_TESTS_ALL = [TestDataPacketGenerator]

def main(tests_all):

    # ----- SPECIFY COMMAND LINE USAGE
    DESCRIPTION =   ""
    SIZES_HELP = "The path to the sizes file associated with the built milhoja library."
    VERBOSE_HELP = f"Verbosity level of logging"
    
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # allow no sizes for now...
    parser.add_argument(
        "--sizes", "-s",
        type=str,
        help=SIZES_HELP, default=""
    )

    parser.add_argument(
        "--verbose", "-v",
        type=int, choices=[0, 1, 2], default=1,
        help=VERBOSE_HELP
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()
    verbosity_level = args.verbose

    suite = unittest.TestSuite()
    for test in tests_all:
        suite.addTest(unittest.makeSuite(test))
    result = unittest.TextTestRunner(verbosity=verbosity_level).run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    if not main(_TESTS_ALL):
        sys.exit(1)
    sys.exit(0)

