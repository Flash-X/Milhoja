#!/usr/bin/env python3

import os
import argparse
import json

from argparse import RawTextHelpFormatter
from milhoja import LOG_LEVEL_MAX
from milhoja import TaskFunction
from milhoja import BasicLogger
from DataPacketGenerator_cpp import DataPacketGenerator_cpp

def parse_configuration():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "JSON", help="[mandatory] The JSON file to generate from."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    code_path = os.environ['MILHOJA_CODE_REPO']
    args = parse_configuration()
    sizes = f"{code_path}/tools/datapacket_generator/sample_jsons/summit_sizes.json"
    with open(sizes, 'r') as sizes_json:
        sizes = json.load(sizes_json)

    assert isinstance(sizes, dict)
    tf_spec = TaskFunction.from_milhoja_json(f"{code_path}/{args.JSON}")
    # use default logging value for now
    logger = BasicLogger(LOG_LEVEL_MAX)
    generator = DataPacketGenerator_cpp(tf_spec, 4, logger, sizes, f"{code_path}/tools/datapacket_generator/templates", './')
    generator.generate_header_code(True)
    generator.generate_source_code(True)
