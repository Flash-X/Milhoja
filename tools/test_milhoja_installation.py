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
    parser = argparse.ArgumentParser(
                description=DESCRIPTION,
                formatter_class=argparse.RawTextHelpFormatter
             )
    args = parser.parse_args()

    # ----- RUN FULL TEST SUITE
    if not milhoja.test():
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()

