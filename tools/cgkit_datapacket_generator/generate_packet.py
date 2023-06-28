#!/usr/bin/env/ python
import generate_helpers_tpl
import run
import argparse
import os
import json
import utility as consts
import warnings

class NoLanguageException(BaseException):
    pass

class NotAJSONException(BaseException):
    pass

class EmptyFileException(BaseException):
    pass

def load_json(file, args) -> dict:
    """Loads the json file into a dict and adds any necessary information to the dict."""
    data = json.load(file)
    data["file_name"] = file.name.replace(".json", "")
    data["name"] = os.path.basename(file.name).replace(".json", "")
    data["language"] = args.language
    data["outer"] = f'cg-tpl.{data["name"]}_outer.cpp'
    data["helpers"] = f'cg-tpl.{data["name"]}_helpers.cpp'
    data["sizes"] = None
    
    #if sizes file is provided it is inserted into the dictionary.
    if args.sizes:
        with open(args.sizes, 'r') as sizes:
            try:
                data["sizes"] = json.load(sizes)
            except Exception:
                warnings.warn("Sizes file could not be loaded. Continuing...")

    return data

def main():
    parser = argparse.ArgumentParser(description="Generate packet code files for use in Flash-X simulations.")
    parser.add_argument("JSON", help="The JSON file to generate from.")
    parser.add_argument('--language', '-l', type=consts.Language, choices=list(consts.Language), help="Generate a packet to work with this language.")
    parser.add_argument("--sizes", "-s", help="Path to data type size information.")
    args = parser.parse_args()

    if args.language is None:
        raise NoLanguageException("You must provide a language!")
    if not args.JSON.endswith('.json'):
        raise NotAJSONException()
    if os.path.getsize(args.JSON) < 5:
        raise EmptyFileException()

    with open(args.JSON, "r") as file:
        data = load_json(file, args) # load data.
        consts.check_json_validity(data) # check if the task args list matches the items in the JSON
        generate_helpers_tpl.generate_helper_template(data) # generate helper templates
        run.main(data) # assemble data packet

        # generate cpp2c and c2f layers here.
        if args.language == consts.Language.fortran:
            import c2f_generator
            import cpp2c_generator
            # cpp2c_generator.main(data)
            c2f_generator.main(data)

if __name__ == "__main__":
    main()
