#!/usr/bin/env python

"""
Run the script with -h to obtain more information regarding the script.
"""

import argparse

import milhoja


def main():
    # ----- HARDCODED VALUES
    # Exit codes so that this can be used in CI build server
    FAILURE = 1
    SUCCESS = 0

    DEFAULT_VERBOSITY = 1

    # ----- SPECIFY COMMAND LINE USAGE
    DESCRIPTION = "Return status of milhoja python package full testing " \
                  + "as exit code for use with CI\n"
    VERBOSE_HELP = "Verbosity level of logging"
    parser = argparse.ArgumentParser(
                description=DESCRIPTION,
                formatter_class=argparse.RawTextHelpFormatter
             )
    parser.add_argument(
        "--verbose", "-v",
        type=int, choices=[0, 1, 2], default=DEFAULT_VERBOSITY,
        help=VERBOSE_HELP
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()
    verbosity_level = args.verbose

    # ----- RUN FULL TEST SUITE
    return SUCCESS if milhoja.test(verbosity_level) else FAILURE


if __name__ == "__main__":
    exit(main())
