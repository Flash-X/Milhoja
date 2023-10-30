#!/usr/bin/env python
import json
import json_sections as sections
import packet_generation_utility as utility
import generate_helpers_tpl
import cpp2c_generator
import c2f_generator
import cgkit.ctree.srctree as srctree
import os
import re
import pathlib

from cgkit.ctree.srctree import SourceTree
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
        TaskFunction.TILE_LO: "IntVect",
        TaskFunction.TILE_HI: "IntVect",
        "tile_loGC": "IntVect",
        "tile_hiGC": "IntVect",
        TaskFunction.TILE_DELTAS: "RealVect",
        TaskFunction.TILE_LEVEL: "unsigned int",
        "grid_data": "real",
        TaskFunction.TILE_FACE_AREAS: "real",
        TaskFunction.TILE_COORDINATES: "real",
        TaskFunction.TILE_GRID_INDEX: "int",
        TaskFunction.TILE_CELL_VOLUMES: "real"
    }

    FORTRAN_EQUIVALENT = {
        "IntVect": "int",
        "RealVect": "real",
        "bool": "logical"
    }

    def __init__(
        self,
        tf_spec: TaskFunction,
        indent: int,
        logger: BasicLogger,
        sizes: dict,
        templates_path: str,
        files_destination: str
    ):
        if not isinstance(tf_spec, TaskFunction):
            raise TypeError("TF Specification was not derived from task function.")

        self.json = {}
        self._TOOL_NAME = self.__class__.__name__
        self._sizes = sizes
        self._templates_path = templates_path
        self._destination = files_destination

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

        self._header_tpl = Path(templates_path, "cg-tpl.datapacket_header.cpp").resolve()
        self._source_tpl = Path(templates_path, "cg-tpl.datapacket.cpp").resolve()
        self.__cpp2c_name = None
        self.__c2f_name = None

        # I need to generate templates immediately.
        self._helper_tpl = Path(self._destination, f"cg-tpl.helper_{self._tf_spec.data_item_class_name}.cpp").resolve()
        self._outer_tpl = Path(self._destination, f"cg-tpl.outer_{self._tf_spec.data_item_class_name}.cpp").resolve()
        generate_helpers_tpl.generate_helper_template(self, True)

    @property
    def language(self):
        return self._tf_spec.language
    
    @property
    def class_name(self):
        return self._tf_spec.data_item_class_name
    
    @property
    def ppd_name(self):
        return f'{self.class_name.upper()}_UNIQUE_IFNDEF_H_'

    def generate_header_code(self, overwrite=True):
        """
        Generate C++ header
        """
        self.generate_packet_file(
            self.header_filename,
            {
                'codePath': pathlib.Path.cwd(),
                'indentSpace': ' '*4,
                'verbose': False,
                'verbosePre': '/* ',
                'verbosePost': ' */',
            },
            [self._outer_tpl, self._header_tpl, self._helper_tpl]
        )

    def generate_source_code(self, overwrite=True):
        """
        Generate C++ source code. Also generates the
        interoperability layers if necessary.
        """
        self.generate_packet_file(
            self.source_filename,
            {
                'codePath': pathlib.Path.cwd(),
                'indentSpace': ' '*4,
                'verbose': False,
                'verbosePre': '/* ',
                'verbosePost': ' */',
            },
            [self._outer_tpl, self._source_tpl, self._helper_tpl]
        )
        # self.generate_cpp2c()
        # self.generate_c2f()

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
            cpp2c_generator.generate_cpp2c(self.json)

        lang = self.language.lower()
        if lang == "fortran":
            generate_cpp2c_f()
        elif lang == "c++":
            generate_cpp2c_cpp()

    def generate_c2f(self, overwrite=True):
        if self.language.lower() == "fortran":
            c2f_generator.generate_c2f(self.json)

    @property
    def outer_template(self):
        return self._outer_tpl

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
                'type': 'int' if lang == "fortran" else 'std::size_t',
                'extents': []
            }
        }
        
        # insert arguments into separate dict for use later
        for item in args:
            external[item] = self._tf_spec.argument_specification(item)

        sort_func = lambda key_and_type: self._sizes.get(key_and_type[1]['type'], 0)
        return self._sort_dict(external.items(), sort_func, True)

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
            args[key]['type'] = self.SOURCE_DATATYPE[args[key]["source"]]
            if lang == "fortran":
                args[key]['type'] = self.FORTRAN_EQUIVALENT[args[key]['type']]
        return self._sort_dict(args.items(), sort_func, True)
    
    def __adjust_tile_data(self, args: dict) -> dict:
        arg_dictionary = {}
        block_extents = self.block_extents
        nguard = self.n_guardcells
        for arg in args:
            arg_dictionary[arg] = self._tf_spec.argument_specification(arg)
            x = '({0}) + 1' if arg_dictionary[arg]['structure_index'][0].lower() == 'fluxx' else '{0}'
            y = '({0}) + 1' if arg_dictionary[arg]['structure_index'][0].lower() == 'fluxy' else '{0}'
            z = '({0}) + 1' if arg_dictionary[arg]['structure_index'][0].lower() == 'fluxz' else '{0}'
            arg_dictionary[arg]['extents'] = [
                x.format(f'{block_extents[0]} + 2 * {nguard} * MILHOJA_K1D'),
                y.format(f'{block_extents[1]} + 2 * {nguard} * MILHOJA_K2D'),
                z.format(f'{block_extents[2]} + 2 * {nguard} * MILHOJA_K3D')
            ]

            # adjust masking.
            if 'variables_in' in arg_dictionary[arg]:
                mask = arg_dictionary[arg]['variables_in']
                arg_dictionary[arg]['variables_in'] = [mask[0]-1, mask[1]-1]
            if 'variables_out' in arg_dictionary[arg]:
                mask = arg_dictionary[arg]['variables_out']
                arg_dictionary[arg]['variables_out'] = [mask[0]-1, mask[1]-1]

            arg_dictionary[arg]['type'] = self.SOURCE_DATATYPE[arg_dictionary[arg]['source']]
        return arg_dictionary

    @property
    def tile_in_args(self) -> OrderedDict:
        """
        Gets all tile_in arguments and formats them for the data packet generator.

        :return: OrderedDict containing all tile_in arguments.
        """
        sort_func = lambda x: self._sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0)
        args = self._tf_spec.tile_in_arguments
        arg_dictionary = self.__adjust_tile_data(args)
        return self._sort_dict(arg_dictionary.items(), sort_func, True)

    @property
    def tile_in_out_args(self) -> OrderedDict:
        """
        Gets all tile_in_out arguments and formats them for the data packet generator.

        :return: OrderedDict containing all tile_in_out arguments.
        """
        sort_func = lambda x: self._sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0)
        args = self._tf_spec.tile_in_out_arguments
        arg_dictionary = self.__adjust_tile_data(args)
        return self._sort_dict(arg_dictionary.items(), sort_func, True)

    @property
    def tile_out_args(self) -> OrderedDict:
        """
        Gets all tile_out arguments and formats them for the data packet generator.

        :return: OrderedDict containing all tile_out arguments.
        """
        sort_func = lambda x: self._sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0)
        args = self._tf_spec.tile_out_arguments
        arg_dictionary = self.__adjust_tile_data(args)
        return self._sort_dict(arg_dictionary.items(), sort_func, True)

    @property
    def scratch_args(self) -> OrderedDict:
        """
        Gets all scratch arguments and formats them for the data packet generator.

        :return: OrderedDict containing all scratch arguments.
        """
        sort_func = lambda x: (self._sizes.get(x[1]["type"], 0), x[0])
        args = self._tf_spec.scratch_arguments
        arg_dictionary = OrderedDict()
        for arg in args:
            arg_dictionary[arg] = self._tf_spec.argument_specification(arg)
            arg_dictionary[arg]['extents'] = self.__parse_extents(arg_dictionary[arg]['extents'])
            arg_dictionary[arg]['lbound'] = self.__parse_lbound(arg_dictionary[arg]['lbound'], 'scratch')
        return self._sort_dict(arg_dictionary.items(), sort_func, False)
    
    @property
    def block_extents(self):
        return self._tf_spec.block_interior_shape
    
    @property
    def n_guardcells(self):
        return self._tf_spec.n_guardcells
    
    @property
    def dimension(self):
        return self._tf_spec.grid_dimension

    def _sort_dict(self, arguments, sort_key, reverse) -> OrderedDict:
        """
        Sorts a given dictionary using the sort key.
        
        :param dict section: The dictionary to sort.
        :param func sort_key: The function to sort with.
        """
        dict_items = [ (k,v) for k,v in arguments ]
        return OrderedDict(sorted(dict_items, key=sort_key, reverse=reverse))

    def generate_packet_file(self, output: str,  sourcetree_opts: dict, linked_templates: list):
        
        def construct_source_tree(stree: SourceTree, templates: list):
            assert len(templates) > 0
            stree.initTree(templates[0])
            stree.pushLink(srctree.search_links(stree.getTree()))

            # load and link each template into source tree.
            for idx,link in enumerate(templates[1:]):
                tree_link = srctree.load(link)
                pathInfo = stree.link(tree_link, linkPath=srctree.LINK_PATH_FROM_STACK)
                if pathInfo:
                    stree.pushLink(srctree.search_links(tree_link))
                else:
                    raise RuntimeError(f'Linking layer {idx} ({link}) unsuccessful!')


        """Generates a source file given a template and output name"""
        stree = SourceTree(**sourcetree_opts, debug=False)
        construct_source_tree(stree, linked_templates)
        lines = stree.parse()
        if os.path.isfile(output):
            # use logger here but for now just print a warning.
            print(f"Warning: {output} already exists. Overwriting.")
        with open(output, 'w') as new_file:
            lines = re.sub(r'#if 0.*?#endif\n\n', '', lines, flags=re.DOTALL)
            new_file.write(lines)

    def __parse_lbound(self, lbound: str, data_source: str):
        """
        Parses an lbound string for use within the generator.
        
        :param str lbound: The lbound string to parse.
        :param str data_source: The source of the data. Eg: scratch or grid data. 
        """
        starting_index = "1"
        # data source is either grid or scratch for tile arrays.
        if data_source == "grid_data":
            lbound_info = lbound.split(',')
            # We control lbound format for grid data structures, 
            # so the length of this lbound should always be 2.
            assert len(lbound_info) == 2
            # get low index
            low = lbound_info[0]
            low = low.strip().replace(')', '').replace('(', '')
            starting_index = lbound_info[-1]
            starting_index = starting_index.strip().replace('(', '').replace(')', '')
            return [low, starting_index]
        elif data_source == "scratch":
            lbound = lbound[1:-1].replace("tile_", '') # remove outer parens, and tile_ from bounds
            # Since tile_*** can be anywhere in scratch data we use SO solution for using negative lookahead 
            # to find tile data.
            lookahead = r',\s*(?![^()]*\))'
            matches = re.split(lookahead, lbound)
            # Can't assume lbound split is a specific size since we don't have control over
            # structures of scratch data.
            for idx,item in enumerate(matches):
                match_intvects = r'\((?:[0-9]+[, ]*)*\)' # use this to match any int vects with only numbers
                unlabeled_intvects = re.findall(match_intvects, item)
                for vect in unlabeled_intvects:
                    matches[idx] = item.replace(vect, f"IntVect{{ LIST_NDIM{vect} }}")
            return matches
        # data source was not valid.
        return ['']
    
    def __parse_extents(self, extents: str) -> list:
        """Parses an extents string."""
        extents = extents.replace('(', '').replace(')', '')
        return [ item.strip() for item in extents.split(',') ]

    def abort(self, msg: str):
        # print message and exit
        self._error(msg)
        exit(-1)
