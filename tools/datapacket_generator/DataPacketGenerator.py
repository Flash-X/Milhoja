#!/usr/bin/env python

import json
import json_sections as sections
import packet_generation_utility as utility
import generate_packet
import generate_helpers_tpl
import packet_source_tree_cgkit as datapacket_cgkit

from pathlib import Path
from milhoja import BaseCodeGenerator
from milhoja import CodeGenerationLogger
from milhoja import TaskFunction
from milhoja import LOG_LEVEL_BASIC

class DataPacketGenerator(BaseCodeGenerator):
    """
    TODO: Convert all jsons to new format from Jared.
    DataPacketGenerator interface uses original json format for now.
    """
    # TODO: This should take a json instead of a namespace or both.
    @classmethod
    def from_json(cls, args, log_level=LOG_LEVEL_BASIC, indent=4):
        instance = cls("", "", "", log_level, indent)
        data_json_file = Path(args.JSON).resolve()
        if not data_json_file.is_file():
            raise ValueError(f'{data_json_file} does not exist or is not a file.')
        
        instance.json = None
        with open(args.JSON, 'r') as json_file:
            instance.json = generate_packet._load_json(json_file, args)

        # insert nTiles into data after checking json.
        if sections.GENERAL not in instance.json:
            instance.json[sections.GENERAL] = {}
        nTiles_type = 'int' if args.language == utility.Language.fortran else 'std::size_t'

        instance.json[sections.GENERAL]['nTiles'] = nTiles_type
        return instance
        
    def __init__(
        self,
        tf_spec,
        header_filename,
        source_filename,
        log_level,
        indent
    ):
        ...
        self.json = {}
        #uper().__init__()

    @property
    def name(self):
        return self.json.get(sections.NAME, "")
    
    def generate_packet(self):
        """Calls all necessary functions to generate a packet."""
        self.generate_templates()
        header = self.generate_header_code()
        source = self.generate_source_code()
        cpp2c = self.generate_cpp2c()
        c2f = self.generate_c2f()
        return {
            "header": header,
            "source": source,
            "cpp2c": cpp2c,
            "c2f": c2f
        }

    def generate_templates(self, overwrite=True):
        """Generates templates for use by cgkit."""
        generate_helpers_tpl.generate_helper_template(self.json)

    def generate_header_code(self, overwrite=True) -> str:
        """
        Generate C++ header
        
        :rtype: str
        :returns: Returns the name of the output.
        """
        # TODO: Replace with new json format
        output_name = f"cgkit.{self.json[sections.NAME]}.h"
        datapacket_cgkit.generate_file(self.json, 'cg-tpl.datapacket_header.cpp', output_name)
        return output_name

    def generate_source_code(self, overwrite=True) -> str:
        """
        Generate C++ source code
        
        :rtype: str
        :returns: Returns the name of the output.
        """
        output_name = f"cgkit.{self.json[sections.NAME]}.cpp"
        datapacket_cgkit.generate_file(self.json, 'cg-tpl.datapacket.cpp', output_name)
        return output_name

    # use language to determine which function to call.
    def generate_cpp2c(self, overwrite=True):
        ...

        def generate_cpp2c_cpp(self):
            ...

        def generate_cpp2c_f(self):
            ...

    def generate_c2f(self, overwrite=True):
        ...