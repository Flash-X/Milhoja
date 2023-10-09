#!/usr/bin/env python3
import generate_helpers_tpl
import packet_source_tree_cgkit as ctree
import argparse
import os
import json
import packet_generation_utility as consts
import json_sections as sections
import c2f_generator
import cpp2c_generator
from typing import TextIO
from argparse import RawTextHelpFormatter

# TODO: DataPacket generator should have an option to clean up any files it
#       generatoes outside of the CPP2C layer, C2F layer, and data packet
#       header and source files.

_LANGUAGE_DESCRIPTION = (
"[mandatory] Generate a packet to work with a task function using this language."
)

_APPLICATION_DESCRIPTION = (
"""=== DataPacket Generator ===

This tool is responsible for generating DataPacket subclasses for use with an
associated task function. To use this tool, run a command similar to:

    generate_packet.py -l fortran -s sample_jsons/sizes.json sample_jsons/DataPacket_Hydro_gpu_3.json

This will output a set of generated files that contain the code for DataPacket
based on the contents of the json file. Note that the generator does not clean
up any files that it generates.

The DataPacket generator overwrites any files that have the same name as its
outputs, so please take care to move or rename any important files that have
been generated by this program so they are not overwritten by accident.
"""
)


# TODO: Replace exception handling with CI build server friendly error handling and logging scheme.
class _NoLanguageException(BaseException):
    """Raised when no language is provided when generating a data packet."""
    pass


class _NotAJSONException(BaseException):
    """Raised when a value that is passed in to generate the data packet is not a JSON."""
    pass


class _EmptyFileException(BaseException):
    """Raised when the file passed in to generate the JSON is empty."""
    pass


class _MissingSizesException(BaseException):
    pass


class _NoTaskFunctionNameException(BaseException):
    pass


def _load_json(file: TextIO, args) -> dict:
    """
    Loads the json file into a dict and adds any necessary information to it.

    :param TextIO file: The file pointer containing the data of the DataPacket json.
    :param args: Command-line argument namespace.
    :return: The loaded json dictionary.
    :rtype: dict[Unknown, Unknown]
    """
    data = json.load(file)
    data[sections.FILE_NAME] = file.name.replace(".json", "")
    data[sections.NAME] = os.path.basename(file.name).replace(".json", "")
    data[sections.LANG] = args.language
    data[sections.OUTER] = f'cg-tpl.{data[sections.NAME]}_outer.cpp'
    data[sections.HELPERS] = f'cg-tpl.{data[sections.NAME]}_helpers.cpp'

    # Sizes file should always be provided. Generating a data packet subclass without sorting
    # the pointers based on size is prone to memory alignment errors, and requires extra padding
    # on every variable.
    with open(args.sizes, 'r') as sizes:
        data[sections.SIZES] = json.load(sizes)
    return data


def generate_packet(args):
    """
    Loads the arguments and JSON, then generates the data packet files.
    Also generates the cpp2c and c2f layers if necessary.
    """
    if args.language is None:
        raise _NoLanguageException("You must provide the language of the paired task function!")
    if not args.JSON.endswith('.json'):
        raise _NotAJSONException("File does not have a .json extension.")
    if os.path.getsize(args.JSON) < 5:
        raise _EmptyFileException("File is empty or too small.")
    if not args.sizes or not args.sizes.endswith('.json'):
        raise _MissingSizesException("No sizes json file provided.")

    with open(args.JSON, "r") as file:
        # load data.
        data = _load_json(file, args)
        # check if the task args list matches the items in the JSON
        consts.check_json_validity(data)

        # insert nTiles into data after checking json.
        if sections.GENERAL not in data:
            data[sections.GENERAL] = {}
        nTiles_type = 'int' if args.language == consts.Language.fortran else 'std::size_t'
        data[sections.GENERAL]['nTiles'] = nTiles_type

        # generate helper templates
        generate_helpers_tpl.generate_helper_template(data)
        # assemble data packet
        ctree.generate_packet_code(data)

        # generate cpp2c and c2f layers here.
        if args.language == consts.Language.fortran:
            if not data.get(sections.TASK_FUNCTION_NAME, ""):
                raise _NoTaskFunctionNameException(f"Missing {sections.TASK_FUNCTION_NAME}.")
            c2f_generator.generate_c2f(data)
            cpp2c_generator.generate_cpp2c(data)


def parse_configuration():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description=_APPLICATION_DESCRIPTION)
    parser.add_argument(
        "JSON", help="[mandatory] The JSON file to generate from."
    )
    parser.add_argument(
        '--language', '-l',
        type=consts.Language, choices=list(consts.Language),
        help=_LANGUAGE_DESCRIPTION
    )
    parser.add_argument(
        "--sizes", "-s", help="[mandatory] Path to data type size information."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_configuration()
    generate_packet(args)