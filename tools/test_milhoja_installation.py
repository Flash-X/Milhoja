#!/usr/bin/env python

"""
Run the script with -h to obtain more information regarding the script.
"""

import sys
import argparse

import milhoja

def main():
    # ----- SPECIFY COMMAND LINE USAGE
    DESCRIPTION =   "Return status of milhoja python package full testing " \
                  + "as exit code for use with CI\n"
    VERBOSE_HELP = f"Verbosity level of logging"
    parser = argparse.ArgumentParser(
                description=DESCRIPTION,
                formatter_class=argparse.RawTextHelpFormatter
             )
    parser.add_argument(
        "--verbose", "-v",
        type=int, choices=[0, 1, 2], default=1,
        help=VERBOSE_HELP
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()
    verbosity_level = args.verbose

    # ----- RUN FULL TEST SUITE
    if not milhoja.test(verbosity_level):
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()

