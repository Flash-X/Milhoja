#!/usr/bin/env python
import cgkit.ctree.srctree as srctree
import re
import pathlib

from TemplateUtility import TemplateUtility
from FortranTemplateUtility import FortranTemplateUtility
from CppTemplateUtility import CppTemplateUtility
from DataPacketMemberVars import DataPacketMemberVars
from collections import defaultdict
from abc import abstractmethod
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

    CPP_EQUIVALENT = {
        "real": "RealVect",
        "int": "IntVect",
        "logical": "bool"
    }

    F_HOST_EQUIVALENT = {
        'RealVect': 'real',
        'IntVect': 'int'
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

    __DEFAULT_SOURCE_TREE_OPTS = {
        'codePath': pathlib.Path.cwd(),
        'indentSpace': ' '*4,
        'verbose': False,
        'verbosePre': '/* ',
        'verbosePost': ' */',
    }

    _SOURCE_TILE_DATA_MAPPING = {
        "CENTER": "tileDesc_h->dataPtr()",
        "FLUXX": "&tileDesc_h->fluxData(milhoja::Axis::I)",
        "FLUXY": "&tileDesc_h->fluxData(milhoja::Axis::J)",
        "FLUXZ": "&tileDesc_h->fluxData(milhoja::Axis::K)"
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

        self._size_connectors = defaultdict(str)
        self._connectors = defaultdict(list) 
        self._params = defaultdict(str)

        outputs = tf_spec.output_filenames
        header = outputs[TaskFunction.DATA_ITEM_KEY]["header"]
        source = outputs[TaskFunction.DATA_ITEM_KEY]["source"]

        super().__init__(
            tf_spec,
            header,
            source,
            indent,
            self._TOOL_NAME,
            logger
        )

        self._log("Initialized", LOG_LEVEL_BASIC)
        self._cpp2c_source = outputs[TaskFunction.CPP_TF_KEY]["source"]

        if self._tf_spec.language.lower() == "c++":
            self.template_utility = CppTemplateUtility
        elif self._tf_spec.language.lower() == "fortran":
            self.tempalte_utility = FortranTemplateUtility
        else:
            self.abort("No template utility for specifed language")

        self._header_tpl = Path(templates_path, "cg-tpl.datapacket_header.cpp").resolve()
        self._source_tpl = Path(templates_path, "cg-tpl.datapacket.cpp").resolve()
        self._cpp2c_tpl = Path(templates_path, "cg-tpl.cpp2c.cpp").resolve()
        self._cpp2c_extra_streams_tpl = Path(templates_path, "cg-tpl.cpp2c_no_extra_queue.cpp")
        if self.n_extra_streams > 0:
            self._cpp2c_extra_streams_tpl = Path(templates_path, "cg-tpl.cpp2c_extra_queue.cpp")
        self._cpp2c_extra_streams_tpl = self._cpp2c_extra_streams_tpl.resolve()

        # I need to generate templates immediately, since both the header and source file for the data packet share 
        # outer and helper template files with each other.
        # This avoid generating the files twice and annoying logic to check if the templates have been generated. 
        # (ex: )
        self._log("Creating templates", LOG_LEVEL_BASIC_DEBUG)
        self._helper_tpl = Path(self._destination, f"cg-tpl.helper_{self._tf_spec.data_item_class_name}.cpp").resolve()
        self._outer_tpl = Path(self._destination, f"cg-tpl.outer_{self._tf_spec.data_item_class_name}.cpp").resolve()
        self.generate_templates()

        # generate cpp2c helpers & outer templates
        self._cpp2c_helper_tpl = Path(self._destination, f"cg-tpl.helper_{self._tf_spec.data_item_class_name}_cpp2c.cpp").resolve()
        self._cpp2c_outer_tpl = Path(self._destination, f"cg-tpl.outer_{self._tf_spec.data_item_class_name}_cpp2c.cpp").resolve()
        # Note: c2f layer does not use cgkit so no templates.
        self._log("Loaded", LOG_LEVEL_MAX)

    # Not really necessary to use constant string keys for this function
    # since param keys do not get used outside of this function.
    def _set_default_params(self):
        """
        Sets the default parameters for cgkit.
        
        :param dict data: The dict containing the data packet JSON data.
        :param dict params: The dict containing all parameters for the outer template.
        """
        # should be guaranteed to be integers by TaskFunction class.
        self._params['align_size'] = str(self.byte_alignment)
        self._params['n_extra_streams'] = str(self.n_extra_streams)
        self._params['class_name'] = self._tf_spec.data_item_class_name
        self._params['ndef_name'] = self.ppd_name
        self._params['header_name'] = self.header_filename

    def generate_templates(self):
        # set defaults for the connectors. 
        self._connectors[self.template_utility._CON_ARGS] = []
        self._connectors[self.template_utility._SET_MEMBERS] = []
        self._connectors[self.template_utility._SIZE_DET] = []
        self._set_default_params()
        
        """Generates the helper template with the provided JSON data."""
        with open(self._helper_tpl, 'w') as template:
            # # SETUP FOR CONSTRUCTOR  
            external = self.external_args
            metadata = self.tile_metadata_args
            tile_in = self.tile_in_args
            tile_in_out = self.tile_in_out_args
            tile_out = self.tile_out_args
            scratch = self.scratch_args

            self.template_utility.iterate_externals(
                self._connectors, self._size_connectors, external
            )

            # # SETUP FOR TILE_METADATA
            num_arrays = len(scratch) + len(tile_in) + \
                        len(tile_in_out) + len(tile_out)
            
            self.template_utility.iterate_tilemetadata(self._connectors, self._size_connectors, metadata, num_arrays)
            self.template_utility.iterate_tile_in(self._connectors, self._size_connectors, tile_in)
            self.template_utility.iterate_tile_in_out(self._connectors, self._size_connectors, tile_in_out)
            self.template_utility.iterate_tile_out(self._connectors, self._size_connectors, tile_out)
            self.template_utility.iterate_tile_scratch(self._connectors, self._size_connectors, scratch)

            tile_data = [ tile_in, tile_in_out, tile_out, scratch ]
            self.template_utility.insert_farray_information(
                self._connectors, tile_data
            )

            # Write to all files.
            self.template_utility.write_size_connectors(self._size_connectors, template)
            self.template_utility.generate_extra_streams_information(self._connectors, self.n_extra_streams)
            self.template_utility.write_connectors(self._connectors, template)

        with open(self._outer_tpl, 'w') as outer:
            outer.writelines(
            [
                '/* _connector:datapacket_outer */\n',
                '/* _link:datapacket */\n'
            ] + 
            ['\n'.join(f'/* _param:{item} = {self._params[item]} */' for item in self._params)]
        )

    def generate_header_code(self, overwrite):
        """
        Generate C++ header
        """
        self.generate_packet_file(
            self.header_filename,
            self.__DEFAULT_SOURCE_TREE_OPTS,
            [self._outer_tpl, self._header_tpl, self._helper_tpl],
            overwrite
        )

    def generate_source_code(self, overwrite):
        """
        Generate C++ source code.
        """
        self.generate_packet_file(
            self.source_filename,
            self.__DEFAULT_SOURCE_TREE_OPTS,
            [self._outer_tpl, self._source_tpl, self._helper_tpl],
            overwrite
        )

    def generate_packet_file(self, output: str,  sourcetree_opts: dict, linked_templates: list, overwrite: bool):
        """
        Generates a data packet file for creating the entire packet.

        :param str output: The output name for the file. 
        :param dict sourcetree_opts: The dictionary containing the sourcetree parameters.
        :param list linked_templates: All templates to be linked. The first in the list is the initial template.
        """
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

        outfile = Path(self._destination, output)
        """Generates a source file given a template and output name"""
        stree = SourceTree(**sourcetree_opts, debug=False)
        construct_source_tree(stree, linked_templates)
        lines = stree.parse()
        
        if outfile.is_file() and overwrite:
            self.warn(f"{str(outfile)} already exists. Overwriting.")
        elif outfile.is_file() and not overwrite:
            self.abort(f"{str(outfile)} is a file. Abort")

        with open(outfile, 'w') as new_file:
            lines = re.sub(r'#if 0.*?#endif\n\n', '', lines, flags=re.DOTALL)
            new_file.write(lines)

    @property
    def name(self):
        return self._tf_spec.name
    
    @property
    def dummy_arguments(self):
        return self._tf_spec.dummy_arguments

    @property
    def packet_class_name(self):
        return self._tf_spec.data_item_class_name
    
    @property
    def ppd_name(self):
        return f'{self.packet_class_name.upper()}_UNIQUE_IFNDEF_H_'

    @property
    def cpp2c_file(self):
        return self._tf_spec["cpp_source"]
    
    @property
    def cpp2c_outer_template(self) -> Path:
        return self._cpp2c_outer_tpl
    
    @property
    def cpp2c_helper_template(self) -> Path:
        return self._cpp2c_helper_tpl
    
    @property
    def cpp2c_streams_template(self) -> Path:
        return self._cpp2c_extra_streams_tpl
    
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
            arg_dictionary[arg]['lbound'] = self.__parse_lbound(arg_dictionary[arg]['lbound'])
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

    def __parse_lbound(self, lbound: str) -> list:
        """
        Parses an lbound string for use within the generator.
        ..todo::
            * This lbound parser only allows simple lbounds. The current format does not allow
            nested arithmetic expressions or more than 2 intvects being combined.
        
        :param str lbound: The lbound string to parse.
        :param str data_source: The source of the data. Eg: scratch or grid data. 
        """
        lbound = lbound.replace("tile_", '').replace(' ', '') # remove tile_ prefix from keywords
        lbound_parts = []
        regexr = r'\(([^\)]+)\)'
        matches = re.findall(regexr, lbound)
        stitch = ''
        
        # find stitching arithmetic
        # ..todo::
        #    * allow math expressions inside intvect constructors?
        if len(matches) > 1:
            assert len(matches) == 2 # only allow simple math for now.
            symbols = re.findall(r'[\+\-\/\*]', lbound)
            assert len(symbols) == 1 # for now
            stitch = symbols[0]

        for m in matches:
            m = m.split(',')
            assert len(m) > 0
            if not m[0].isnumeric(): # we have a keyword
                ncomp = m[1] if len(m) > 1 else None
                lbound_parts.append((f'({m[0]})', ncomp))
            elif all([ value.isnumeric() for value in m ]):
                init_vect = ['1','1','1']
                ncomp = None
                for idx,value in enumerate(init_vect):
                    init_vect[idx] = str(m[idx])
                if len(m) > len(init_vect):
                    assert len(m) == 4 # there should never be a case where its greater than 4
                    ncomp = m[-1]
                lbound_parts.append((f'IntVect{{LIST_NDIM({",".join(init_vect)})}}', ncomp))

        results = []
        for item in lbound_parts:
            if item[0]:
                if len(results) == 0:
                    results.append(item[0])
                else:
                    results[0] = results[0] + stitch + item[0]

                if item[1]:
                    if len(results) == 1:
                        results.append(item[1])
                    else:
                        results[1] = results[1] + stitch + item[1]
        return results
    
    def __parse_extents(self, extents: str) -> list:
        """Parses an extents string."""
        extents = extents.replace('(', '').replace(')', '')
        return [ item.strip() for item in extents.split(',') ]
    
    def warn(self, msg: str):
        self._warn(msg)

    def abort(self, msg: str):
        # print message and exit
        self._error(msg)
        exit(-1)
