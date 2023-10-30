#!/usr/bin/env python

import argparse
import traceback

from pathlib import Path

import milhoja.tests


def main():
    # ----- HARDCODED VALUES
    CLONE_PATH = Path(__file__).resolve().parents[3]

    # Location of Milhoja-JSON files
    CG_PATH = CLONE_PATH.joinpath("test", "Base", "code_generation")

    # Exit codes so that this can be used in CI build server
    FAILURE = 1
    SUCCESS = 0

    INDENT = 4
    LOG_TAG = "Milhoja Test"
    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC

    # ----- PROGRAM USAGE INFO
    DESCRIPTION = "Milhoja's test build system calls this to generate code"
    DESTINATION_HELP = "Pre-existing folder to write files to"
    DIM_HELP = "Dimension of test problem"
    NXB_HELP = "N cells in each block along x-axis"
    NYB_HELP = "N cells in each block along y-axis"
    NZB_HELP = "N cells in each block along z-axis"
    MAKEFILE_HELP = "Filename with path of Makefile to generate"
    OVERWRITE_HELP = "Original files overwritten if given"
    VERBOSE_HELP = "Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
                description=DESCRIPTION, formatter_class=formatter
             )
    parser.add_argument("destination", nargs=1, help=DESTINATION_HELP)
    parser.add_argument("makefile", nargs=1, help=MAKEFILE_HELP)
    parser.add_argument(
        "dimension", nargs=1, type=int, choices=[1, 2, 3],
        help=DIM_HELP
    )
    parser.add_argument("nxb", nargs=1, type=int, help=NXB_HELP)
    parser.add_argument("nyb", nargs=1, type=int, help=NYB_HELP)
    parser.add_argument("nzb", nargs=1, type=int, help=NZB_HELP)
    parser.add_argument(
        "--overwrite", action='store_true', required=False,
        help=OVERWRITE_HELP
    )
    parser.add_argument(
        "--verbose", "-v", type=int,
        choices=milhoja.LOG_LEVELS, default=DEFAULT_LOG_LEVEL,
        help=VERBOSE_HELP
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    destination = Path(args.destination[0]).resolve()
    makefile = Path(args.makefile[0]).resolve()
    dimension = args.dimension[0]
    nxb = args.nxb[0]
    nyb = args.nyb[0]
    nzb = args.nzb[0]
    overwrite = args.overwrite
    logger = milhoja.BasicLogger(args.verbose)

    assert dimension == 2
    assert nxb == 8
    assert nyb == 16
    assert nzb == 1

    # ----- ABORT WITH MESSAGE & COMMUNICATE FAILURE
    def log_and_abort(error_msg):
        logger.error(LOG_TAG, error_msg)
        exit(FAILURE)

    try:
        tf_spec_jsons = milhoja.tests.generate_runtime_cpu_tf_specs(
                            CG_PATH, destination,
                            overwrite, logger
                        )
        milhoja.tests.generate_runtime_cpu_code(
                            tf_spec_jsons, destination, overwrite,
                            INDENT, makefile, logger
        )
    except Exception as error:
        error_msg = str(error)
        if logger.level >= milhoja.LOG_LEVEL_BASIC_DEBUG:
            error_msg += f"\n{traceback.format_exc()}"
        log_and_abort(error_msg)

    return SUCCESS


if __name__ == "__main__":
    exit(main())
