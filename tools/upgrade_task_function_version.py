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
    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC

    # ----- PROGRAM USAGE INFO
    DESCRIPTION = "Upgrade the task function specification to current version"
    FILENAME_HELP = "Task function specification file to upgrade"
    FORMAT_HELP = "Task function specification format"
    OVERWRITE_HELP = "Original file is overwritten if given"
    VERBOSE_HELP = "Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=formatter
    )
    parser.add_argument("file", nargs=1, help=FILENAME_HELP)
    parser.add_argument(
        "format", nargs=1,
        type=str.lower,
        choices=[e.lower() for e in milhoja.TASK_FUNCTION_FORMATS],
        help=FORMAT_HELP
    )
    parser.add_argument(
        "--overwrite",
        action='store_true', required=False,
        help=OVERWRITE_HELP
    )
    parser.add_argument(
        "--verbose", "-v",
        type=int, choices=milhoja.LOG_LEVELS,
        help=VERBOSE_HELP,
        default=DEFAULT_LOG_LEVEL
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    filename = Path(args.file[0]).resolve()
    fmt = args.format
    overwrite = args.overwrite
    logger = milhoja.BasicLogger(args.verbose)

    def log_and_abort(error_msg):
        logger.error(LOG_TAG, error_msg)
        exit(FAILURE)

    # ----- GET TO UPGRADIN'
    log_and_abort("This has never been tested")

    if overwrite:
        filename_dest = filename
    else:
        filename_dest = Path(str(filename) + "_new")
        if filename_dest.exists():
            log_and_abort(f"Upgraded file already exists ({filename_dest})")

    try:
        if fmt.lower() == milhoja.MILHOJA_JSON_FORMAT.lower():
            tf_spec = milhoja.TaskFunction.from_milhoja_json(filename)
            _, version = tf_spec.specification_format
            if version.lower() == milhoja.CURRENT_MILHOJA_JSON_VERSION.lower():
                log_and_abort("File already at current version")
            else:
                tf_spec.to_milhoja_json(filename_dest)
        else:
            # This should never happen because argparse should error first
            error_msg = f"Unknown specification format {fmt}"
            log_and_abort(error_msg)
    except Exception as error:
        error_msg = str(error)
        if logger.level >= milhoja.LOG_LEVEL_BASIC_DEBUG:
            error_msg += f"\n{traceback.format_exc()}"
        log_and_abort(error_msg)

    return SUCCESS


if __name__ == "__main__":
    exit(main())
