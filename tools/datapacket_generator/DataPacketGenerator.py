#!/usr/bin/env python
import json
import json_sections as sections
import packet_generation_utility as utility
import generate_packet
import generate_helpers_tpl
import packet_source_tree_cgkit as datapacket_cgkit
import cpp2c_generator
import c2f_generator

from pathlib import Path
from milhoja import AbcCodeGenerator
from milhoja import LOG_LEVEL_BASIC
from milhoja import LOG_LEVEL_BASIC_DEBUG
from milhoja import LOG_LEVEL_MAX


class DataPacketGenerator(AbcCodeGenerator):
    """
    TODO: Convert all jsons to new format from Jared.
    DataPacketGenerator interface uses datapacket json format for now.

    This class serves as a wrapper for all of the packet generation scripts.
    This will eventually be built into the primary means of generating data
    packets instead of calling generate_packet.py.
    """

    # TODO: This should take in a new json with format brought in by tile
    #       wrapper
    @classmethod
    def from_json(cls, args, log_level=LOG_LEVEL_BASIC, indent=4):
        data_json_file = Path(args.JSON).resolve()
        if not data_json_file.is_file():
            raise ValueError(
                f'{data_json_file} does not exist or is not a file.'
            )

        json = None
        with open(args.JSON, 'r') as json_file:
            json = generate_packet._load_json(json_file, args)
        instance = cls(
            "",
            f'{json[sections.NAME]}.h',
            f'{json[sections.NAME]}.cpp',
            log_level,
            indent
        )
        instance.json = json

        # insert nTiles into data after checking json.
        if sections.GENERAL not in instance.json:
            instance.json[sections.GENERAL] = {}
        nTiles_type = 'std::size_t'
        if args.language == utility.Language.fortran:
            nTiles_type = 'int'
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
        self.json = {}
        self.__TOOL_NAME = self.__class__.__name__
        super().__init__(
            tf_spec,
            header_filename,
            source_filename,
            self.__TOOL_NAME,
            log_level,
            indent
        )
        self._helper_tpl = None
        self._outer_tpl = None
        self.__cpp2c_name = f'{header_filename.replace(".h", "").replace(".json", "")}.cpp2c.cxx'
        self.__c2f_name = f'{header_filename.replace(".h", "").replace(".json", "")}.c2f.F90'

    @property
    def name(self):
        return self.json.get(sections.NAME, "")

    def generate_packet(self):
        """Calls all necessary functions to generate a packet."""
        self._log("Generating headers...", LOG_LEVEL_BASIC)
        self.generate_header_code()
        self._log("Generating source...", LOG_LEVEL_BASIC)
        self.generate_source_code()
        self._log("Generating layers...", LOG_LEVEL_BASIC)
        self.generate_cpp2c()
        self.generate_c2f()
        self._log("Generation complete.", LOG_LEVEL_BASIC)

    # TODO: This does not work if the templates need to be overwritten or there
    #  is a new version of the code generator.
    def check_generate_template(self, overwrite=True):
        """Generates templates for use by cgkit."""
        if self._helper_tpl and self._outer_tpl:
            self._log(
                "Templates already created, skipping...",
                LOG_LEVEL_BASIC_DEBUG
            )

        self._log("Checking for generated template...", LOG_LEVEL_BASIC_DEBUG)
        self._helper_tpl = Path(f"{self.json[sections.NAME]}_helpers.cpp")\
            .resolve()
        self._outer_tpl = Path(f"{self.json[sections.NAME]}_outer.cpp")\
            .resolve()

        generate_helpers_tpl.generate_helper_template(self.json)

    def generate_header_code(self, overwrite=True):
        """
        Generate C++ header
        """
        # TODO: Replace with new json format
        self.check_generate_template()
        datapacket_cgkit.generate_file(
            self.json,
            'cg-tpl.datapacket_header.cpp',
            self.header_filename
        )

    def generate_source_code(self, overwrite=True):
        """
        Generate C++ source code. Also generates the
        interoperability layers if necessary.
        """
        self.check_generate_template()
        datapacket_cgkit.generate_file(
            self.json,
            'cg-tpl.datapacket.cpp',
            self.source_filename
        )
        self.generate_cpp2c()
        self.generate_c2f()

    # use language to determine which function to call.
    def generate_cpp2c(self, overwrite=True):
        """
        Generates translation layers based on the language
        of the TaskFunction.
            fortran - Generates c2f and cpp2c layers.
            cpp - Generates a C++ task function that calls
        """

        def generate_cpp2c_cpp():
            ...

        def generate_cpp2c_f():
            self._log("Generating cpp2c for fortran", LOG_LEVEL_BASIC_DEBUG)
            self._log(json.dumps(self.json, indent=4, default=str), LOG_LEVEL_MAX)
            cpp2c_generator.generate_cpp2c(self.json)

        lang = self.json[sections.LANG]
        if lang == utility.Language.fortran:
            generate_cpp2c_f()
        elif lang == utility.Language.cpp:
            generate_cpp2c_cpp()

    def generate_c2f(self, overwrite=True):
        if self.json[sections.LANG] == utility.Language.fortran:
            c2f_generator.generate_c2f(self.json)

    @property
    def packet_outer_tpl(self):
        return self._helper_tpl

    @property
    def packet_helper_tpl(self):
        return self._helper_tpl

    @property
    def cpp2c_filename(self):
        return self.__cpp2c_name

    @property
    def c2f_filename(self):
        return self.__c2f_name
