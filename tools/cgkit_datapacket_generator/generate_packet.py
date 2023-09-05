#!/usr/bin/env python3
import generate_helpers_tpl
import packet_source_tree_cgkit as ctree
import argparse
import os
import json
import packet_generation_utility as consts
import json_sections as sections
import warnings
import c2f_generator
import cpp2c_generator
from typing import TextIO

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
    data[sections.SIZES] = None
    
    # if sizes file is provided it is inserted into the dictionary. TODO: Should sizes file always be provided? Probably (See requirement 2)
    if args.sizes:
        with open(args.sizes, 'r') as sizes:
            try:
                data[sections.SIZES] = json.load(sizes)
            except Exception:
                warnings.warn("Sizes file could not be loaded. Continuing...")
    return data

def main():
    """
    Loads the arguments and JSON, then generates the data packet files.
    Also generates the cpp2c and c2f layers if necessary.
    """
    parser = argparse.ArgumentParser(description="Generate packet code files for use in Flash-X simulations.")
    parser.add_argument("JSON", help="The JSON file to generate from.")
    parser.add_argument('--language', '-l', type=consts.Language, choices=list(consts.Language), help="Generate a packet to work with this language.")
    parser.add_argument("--sizes", "-s", help="Path to data type size information.")
    args = parser.parse_args()

    if args.language is None:
        raise _NoLanguageException("You must provide a language!")
    if not args.JSON.endswith('.json'):
        raise _NotAJSONException("File does not have a .json extension.")
    if os.path.getsize(args.JSON) < 5:
        raise _EmptyFileException("File is empty or too small.")

    with open(args.JSON, "r") as file:
        data = _load_json(file, args) # load data.
        consts.check_json_validity(data) # check if the task args list matches the items in the JSON
        generate_helpers_tpl.generate_helper_template(data) # generate helper templates
        ctree.main(data) # assemble data packet

        # generate cpp2c and c2f layers here.
        if args.language == consts.Language.fortran:
            c2f_generator.main(data)
            cpp2c_generator.main(data)

if __name__ == "__main__":
    main()
