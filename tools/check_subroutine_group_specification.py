#!/usr/bin/env python

"""
Run the script with -h to obtain more information regarding the script.
"""

import argparse
import traceback

from pathlib import Path

import milhoja


def main():
    # ----- HARDCODED VALUES
    # Exit codes so that this can be used in CI build server
    FAILURE = 1
    SUCCESS = 0

    LOG_TAG = "Milhoja Tools"
    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC_DEBUG

    # ANSI Terminal Colors
    # Rather than use green/red, I have been told that blue/red is better
    # for people with color blindness
    SUCCESS_COLOR = '\033[0;94;1m'  # Bright Blue/bold
    NO_COLOR = '\033[0m'            # No Color/Not bold

    # ----- PROGRAM USAGE INFO
    DESCRIPTION = "Check the given subroutine group specification for errors"
    FILENAME_HELP = "Subroutine group specification file to check"
    FROM_HELP = "Format of specification file"
    VERBOSE_HELP = "Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=formatter)
    parser.add_argument("file", nargs=1, help=FILENAME_HELP)
    parser.add_argument(
        "from_format", nargs=1, type=str.lower,
        choices=[e.lower() for e in milhoja.TASK_FUNCTION_FORMATS],
        help=FROM_HELP
    )
    parser.add_argument("--verbose", "-v", type=int,
                        choices=milhoja.LOG_LEVELS, help=VERBOSE_HELP,
                        default=DEFAULT_LOG_LEVEL)

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    group_spec = Path(args.file[0]).resolve()
    from_format = args.from_format[0]
    logger = milhoja.BasicLogger(args.verbose)

    def log_and_abort(error_msg):
        print()
        logger.error(LOG_TAG, error_msg)
        print()
        exit(FAILURE)

    if not group_spec.is_file():
        log_and_abort(f"{group_spec} does not exist or is not a file")

    # ----- CHECK SPECIFICATION
    try:
        if from_format.lower() == milhoja.MILHOJA_JSON_FORMAT.lower():
            milhoja.SubroutineGroup.from_milhoja_json(group_spec, logger)
        else:
            # This should never happen because argparse should error first
            error_msg = f"Unknown from specification format {from_format}"
            log_and_abort(error_msg)
    except Exception as error:
        error_msg = str(error)
        if logger.level >= (milhoja.LOG_LEVEL_BASIC_DEBUG + 1):
            # Don't show traceback by default
            error_msg += f"\n{traceback.format_exc()}"
        log_and_abort(error_msg)

    msg = f"{SUCCESS_COLOR}Specification Valid{NO_COLOR}"
    logger.log(LOG_TAG, "", milhoja.LOG_LEVEL_BASIC)
    logger.log(LOG_TAG, msg, milhoja.LOG_LEVEL_BASIC)
    logger.log(LOG_TAG, "", milhoja.LOG_LEVEL_BASIC)

    return SUCCESS


if __name__ == "__main__":
    exit(main())
