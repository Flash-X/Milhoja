#!/usr/bin/env python
import json
import json_sections as sections
import packet_generation_utility as utility
import generate_helpers_tpl
import packet_source_tree_cgkit as datapacket_cgkit
import cpp2c_generator
import c2f_generator

from collections import OrderedDict
from pathlib import Path
from milhoja import AbcCodeGenerator
from milhoja import LOG_LEVEL_BASIC
from milhoja import LOG_LEVEL_BASIC_DEBUG
from milhoja import LOG_LEVEL_MAX
from milhoja import TaskFunction
from milhoja import BasicLogger

class DataPacketGenerator(AbcCodeGenerator):
    """
    TODO: Convert all jsons to new format from Jared.
    DataPacketGenerator interface uses datapacket json format for now.

    TODO: User should need to pass in a destination path.

    This class serves as a wrapper for all of the packet generation scripts.
    This will eventually be built into the primary means of generating data
    packets instead of calling generate_packet.py.
    """
    TILE_VARIABLE_MAPPING = {
        'levels': 'unsigned int',
        'gridIndex': 'int',
        'tileIndex': 'int',
        'tile_deltas': 'RealVect',
        'tile_lo': "IntVect",
        'tile_hi': "IntVect",
        'tile_loGC': "IntVect",
        'tile_hiGC': "IntVect"
    }

    FARRAY_MAPPING = {
        "int": "IntVect",
        "real": "RealVect"
    }

    F_HOST_EQUIVALENT = {
        'RealVect': 'real',
        'IntVect': 'int'
    }

    CPP_EQUIVALENT = {
        "real": "RealVect",
        "int": "IntVect",
        "logical": "bool"
    }

    SOURCE_DATATYPE = {
        "tile_lo": "IntVect",
        "tile_hi": "IntVect",
        "tile_loGC": "IntVect",
        "tile_hiGC": "IntVect",
        "levels": "unsigned int",
        "grid_data": "real",
        'levels': 'unsigned int',

    }

    def __init__(
        self,
        tf_spec: TaskFunction,
        indent: int,
        logger: BasicLogger,
        sizes: dict
    ):
        if not isinstance(tf_spec, TaskFunction):
            raise TypeError("TF Specification was not derived from task function.")

        self.json = {}
        self._TOOL_NAME = self.__class__.__name__
        self._sizes = sizes

        outputs = tf_spec.output_filenames
        header = outputs[TaskFunction.DATA_ITEM_KEY]["header"]
        source = outputs[TaskFunction.DATA_ITEM_KEY]["source"]
        # TODO: Include layer source files
        cpp2c_source = ""
        c2f_source = ""

        super().__init__(
            tf_spec,
            header,
            source,
            indent,
            self._TOOL_NAME,
            logger
        )

        self._log("Loaded tf spec", LOG_LEVEL_BASIC_DEBUG)

        self._helper_tpl = f"cg-tpl.outer_{tf_spec.data_item_class_name}.cpp"
        self._outer_tpl = f"cg-tpl.outer_{tf_spec.data_item_class_name}.cpp"
        self.__cpp2c_name = ""
        self.__c2f_name = ""

    @property
    def language(self):
        return 
    
    @property
    def class_name(self):
        return self._tf_spec.data_item_class_name
    
    @property
    def ppd_name(self):
        return f'{self.class_name.upper()}_UNIQUE_IFNDEF_H_'

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
    def outer_template(self):
        return self._helper_tpl

    @property
    def helper_template(self):
        return self._helper_tpl

    @property
    def cpp2c_filename(self):
        return self.__cpp2c_name

    @property
    def c2f_filename(self):
        return self.__c2f_name
    
    @property
    def n_extra_streams(self) -> int:
        # for now data packet generator will return the number of 
        # extra streams.
        return self._tf_spec.n_streams-1
    
    @property
    def byte_alignment(self) -> int:
        return self._tf_spec.data_item_byte_alignment

    @property
    def external_args(self) -> OrderedDict:
        lang = self._tf_spec.language.lower()
        args = self._tf_spec.external_arguments
        external = {
            'nTiles': {
                'source': 'internal', # ?
                'name': 'nTiles',
                'type': 'int' if lang == "fortran" else 'std::size_t'
            }
        }
        
        # insert arguments into separate dict for use later
        for item in args:
            external[item] = self._tf_spec.argument_specification(item)

        sort_func = lambda key_and_type: self._sizes.get(key_and_type[1]['type'], 0)
        return self._sort_dict(external.items(), sort_func)

    @property
    def tile_metadata_args(self) -> OrderedDict:
        lang = self._tf_spec.language.lower()
        sort_func = None

        if lang == 'c++':
            sort_func = lambda kv_pair: self._sizes.get(
                self.TILE_VARIABLE_MAPPING[kv_pair[1]['source']], 0
            )
        elif lang == 'fortran':
            sort_func = lambda x: self._sizes.get(
                self.F_HOST_EQUIVALENT[self.TILE_VARIABLE_MAPPING[x[1]['source']]], 0
            )

        args = self._tf_spec.tile_metadata_arguments
        for key in args:
            args[key] = self._tf_spec.argument_specification(key)
        return self._sort_dict(args.items(), sort_func)

    @property
    def tile_in_args(self):
        sort_func = lambda x: self._sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0)
        args = self._tf_spec.tile_in_arguments
        arg_dictionary = {}
        for arg in args:
            arg_dictionary[arg] = self._tf_spec.argument_specification(arg)
        return self._sort_dict(arg_dictionary.items(), sort_func)

    @property
    def tile_in_out_args(self):
        sort_func = lambda x: self._sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0)
        args = self._tf_spec.tile_in_out_arguments
        arg_dictionary = {}
        for arg in args:
            arg_dictionary[arg] = self._tf_spec.argument_specification(arg)
        return self._sort_dict(arg_dictionary.items(), sort_func)

    @property
    def tile_out_args(self):
        sort_func = lambda x: self._sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0)
        args = self._tf_spec.tile_out_arguments
        arg_dictionary = {}
        for arg in args:
            arg_dictionary[arg] = self._tf_spec.argument_specification(arg)
        return self._sort_dict(arg_dictionary.items(), sort_func)

    @property
    def scratch_args(self):
        sort_func = lambda x: self._sizes.get(x[1]["type"], 0)
        args = self._tf_spec.scratch_arguments
        arg_dictionary = {}
        for arg in args:
            arg_dictionary[arg] = self._tf_spec.argument_specification(arg)
        return self._sort_dict(arg_dictionary.items(), sort_func)
    
    @property
    def block_extents(self):
        return self._tf_spec.block_interior_shape
    
    @property
    def nguard(self):
        return self._tf_spec.n_guardcells

    def _sort_dict(self, arguments, sort_key) -> OrderedDict:
        """
        Sorts a given dictionary using the sort key.
        
        :param dict section: The dictionary to sort.
        :param func sort_key: The function to sort with.
        """
        dict_items = [ (k,v) for k,v in arguments ]
        return OrderedDict(sorted(dict_items, key=sort_key, reverse=True))

    def abort(self, msg: str):
        # print message and exit
        ...