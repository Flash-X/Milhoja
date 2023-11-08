import cgkit.ctree.srctree as srctree
import re
import pathlib
import functools
import pkgutil # use pkgutil for backwards compatability older python 3s.

from copy import deepcopy
from collections import defaultdict
from cgkit.ctree.srctree import SourceTree
from collections import OrderedDict
from pathlib import Path

from .parse_helpers import parse_extents
from .parse_helpers import parse_lbound
from .Cpp2CLayerGenerator import Cpp2CLayerGenerator
from .C2FortranLayerGenerator import C2FortranLayerGenerator
from .TemplateUtility import TemplateUtility
from .FortranTemplateUtility import FortranTemplateUtility
from .CppTemplateUtility import CppTemplateUtility
from .AbcCodeGenerator import AbcCodeGenerator
from .TaskFunction import TaskFunction
from .BasicLogger import BasicLogger
from .constants import (
    LOG_LEVEL_BASIC,
    LOG_LEVEL_BASIC_DEBUG,
    LOG_LEVEL_MAX
)


class DataPacketGenerator(AbcCodeGenerator):
    """
    This class serves as a wrapper for all of the packet generation scripts.
    This will eventually be built into the primary means of generating data
    packets instead of calling generate_packet.py.

    ..todo::
        * check if lru_caching is necessary on properties that
          are not tile_scratch.
    """
    CPP_EQUIVALENT = {
        "real": "RealVect",
        "int": "IntVect",
        "logical": "bool"
    }

    F_HOST_EQUIVALENT = {
        'RealVect': 'real',
        'IntVect': 'int'
    }

    TILE_VARIABLE_MAPPING = TemplateUtility.TILE_VARIABLE_MAPPING
    F_HOST_EQUIVALENT = FortranTemplateUtility.F_HOST_EQUIVALENT

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
        files_destination: str
    ):
        if not isinstance(tf_spec, TaskFunction):
            raise TypeError(
                "TF Specification was not derived from task function."
            )

        self._TOOL_NAME = self.__class__.__name__
        self._sizes = sizes
        self._destination = files_destination
        self._indent = indent

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
            self.template_utility = FortranTemplateUtility
        else:
            self.abort("No template utility for specifed language")

        self._templates_path = './templates'
        self._templates_path = ...

        self._cpp2c_extra_streams_tpl = "cg-tpl.cpp2c_no_extra_queue.cpp"
        if self.n_extra_streams > 0:
            self._cpp2c_extra_streams_tpl = "cg-tpl.cpp2c_extra_queue.cpp"

        # I need to generate templates immediately, since both the header
        # and source file for the data packet share
        # outer and helper template files with each other.
        # This avoid generating the files twice and annoying logic to check
        # if the templates have been generated.
        self._log("Creating templates", LOG_LEVEL_BASIC_DEBUG)
        self.generate_templates()

        # Note: c2f layer does not use cgkit so no templates.
        self._log("Loaded", LOG_LEVEL_MAX)

    # Not really necessary to use constant string keys for this function
    # since param keys do not get used outside of this function.
    def _set_default_params(self):
        """
        Sets the default parameters for cgkit.

        :param dict data: The dict containing the data packet JSON data.
        :param dict params: The dict containing all parameters
                            for the outer template.
        """
        # should be guaranteed to be integers by TaskFunction class.
        self._params['align_size'] = str(self.byte_alignment)
        self._params['n_extra_streams'] = str(self.n_extra_streams)
        self._params['class_name'] = self._tf_spec.data_item_class_name
        self._params['ndef_name'] = self.ppd_name
        self._params['header_name'] = super().header_filename

    def generate_templates(self):
        # set defaults for the connectors.
        self._connectors[self.template_utility._CON_ARGS] = []
        self._connectors[self.template_utility._SET_MEMBERS] = []
        self._connectors[self.template_utility._SIZE_DET] = []
        self._set_default_params()

        """Generates the helper template with the provided JSON data."""
        with open(self.helper_template, 'w') as template:
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

            self.template_utility.iterate_tilemetadata(
                self._connectors, self._size_connectors, metadata, num_arrays
            )
            self.template_utility.iterate_tile_in(
                self._connectors, self._size_connectors, tile_in
            )
            self.template_utility.iterate_tile_in_out(
                self._connectors, self._size_connectors, tile_in_out
            )
            self.template_utility.iterate_tile_out(
                self._connectors, self._size_connectors, tile_out
            )
            self.template_utility.iterate_tile_scratch(
                self._connectors, self._size_connectors, scratch
            )

            tile_data = [tile_in, tile_in_out, tile_out, scratch]
            self.template_utility.insert_farray_information(
                self._connectors, tile_data
            )

            # Write to all files.
            self.template_utility.write_size_connectors(
                self._size_connectors, template
            )
            self.template_utility.generate_extra_streams_information(
                self._connectors, self.n_extra_streams
            )
            self.template_utility.write_connectors(self._connectors, template)

        with open(self.outer_template, 'w') as outer:
            outer.writelines(
                [
                    '/* _connector:datapacket_outer */\n',
                    '/* _link:datapacket */\n'
                ] +
                [
                    '\n'.join(
                        f'/* _param:{item} = {self._params[item]} */'
                        for item in self._params
                    )
                ]
            )

    def generate_header_code(self, overwrite):
        """
        Generate C++ header
        """
        self.generate_packet_file(
            self.header_filename,
            self.__DEFAULT_SOURCE_TREE_OPTS,
            [self.outer_template, self.header_template, self.helper_template],
            overwrite
        )

    def generate_source_code(self, overwrite):
        """
        Generate C++ source code.
        """
        self.generate_packet_file(
            self.source_filename,
            self.__DEFAULT_SOURCE_TREE_OPTS,
            [self.outer_template, self.source_template, self.helper_template],
            overwrite
        )

        if self._tf_spec.language.lower() == "fortran":
            # generate cpp2c layer if necessary
            #   -> Cpp task function generator?
            cpp2c_layer = Cpp2CLayerGenerator(
                self._tf_spec, self.cpp2c_outer_template,
                self.cpp2c_helper_template, self._indent, self._logger.level,
                self.n_extra_streams, self.external_args
            )
            cpp2c_layer.generate_source_code()
            self.generate_packet_file(
                self.cpp2c_file,
                self.__DEFAULT_SOURCE_TREE_OPTS,
                [
                    self.cpp2c_outer_template,
                    self.cpp2c_template,
                    self.cpp2c_helper_template,
                    self.cpp2c_streams_template
                ],
                # dev note: ORDER MATTERS HERE!
                # If helpers is put before the base
                # template it will throw an error
                overwrite
            )
            # generate fortran to c layer if necessary
            c2f_layer = C2FortranLayerGenerator(
                self._tf_spec, self.c2f_file, self._indent,
                self._logger.level, self.n_extra_streams,
                self.external_args, self.tile_metadata_args,
                self.tile_in_args, self.tile_in_out_args,
                self.tile_out_args, self.scratch_args
            )
            # c2f layer does not use cgkit so no need
            # to call generate_packet_file
            c2f_layer.generate_source_code(overwrite)

    def generate_packet_file(
        self, output: Path,
        sourcetree_opts: dict,
        linked_templates: list,
        overwrite: bool
    ):
        """
        Generates a data packet file for creating the entire packet.

        :param str output: The output name for the file.
        :param dict sourcetree_opts: The dictionary containing the
                                     sourcetree parameters.
        :param list linked_templates: All templates to be linked.
                                      The first in the list is
                                      the initial template.
        """
        def construct_source_tree(stree: SourceTree, templates: list):
            assert len(templates) > 0
            stree.initTree(templates[0])
            stree.pushLink(srctree.search_links(stree.getTree()))

            # load and link each template into source tree.
            for idx, link in enumerate(templates[1:]):
                tree_link = srctree.load(link)
                pathInfo = stree.link(
                    tree_link, linkPath=srctree.LINK_PATH_FROM_STACK
                )
                if pathInfo:
                    stree.pushLink(srctree.search_links(tree_link))
                else:
                    raise RuntimeError(
                        f'Linking layer {idx} ({link}) unsuccessful!'
                    )

        """Generates a source file given a template and output name"""
        stree = SourceTree(**sourcetree_opts, debug=False)
        construct_source_tree(stree, linked_templates)
        lines = stree.parse()

        if output.is_file() and overwrite:
            self.warn(f"{str(output)} already exists. Overwriting.")
        elif output.is_file() and not overwrite:
            self.abort(f"{str(output)} is a file. Abort")

        with open(output, 'w') as new_file:
            lines = re.sub(r'#if 0.*?#endif\n\n', '', lines, flags=re.DOTALL)
            new_file.write(lines)

    @property
    def name(self):
        """Task function name"""
        return self._tf_spec.name

    @property
    def language(self):
        return self._tf_spec.language.lower()

    @property
    def dummy_arguments(self):
        """Dummy argument list"""
        return self._tf_spec.dummy_arguments

    @property
    def packet_class_name(self):
        """Class name of the data packet"""
        return self._tf_spec.data_item_class_name

    @property
    def ppd_name(self):
        """Preprocessor string for the data packet"""
        return f'{self.packet_class_name.upper()}_UNIQUE_IFNDEF_H_'

    @property
    def helper_template(self) -> Path:
        return Path(
            self._destination,
            f"cg-tpl.helper_{self._tf_spec.data_item_class_name}.cpp"
        ).resolve()

    @property
    def outer_template(self) -> Path:
        return Path(
            self._destination,
            f"cg-tpl.outer_{self._tf_spec.data_item_class_name}.cpp"
        ).resolve()

    @property
    def header_template(self) -> Path:
        return Path(
            self._templates_path,
            "cg-tpl.datapacket_header.cpp"
        ).resolve()

    @property
    def source_template(self) -> Path:
        return Path(self._templates_path, "cg-tpl.datapacket.cpp").resolve()

    @property
    def header_filename(self) -> Path:
        return Path(self._destination, super().header_filename).resolve()

    @property
    def source_filename(self) -> Path:
        return Path(self._destination, super().source_filename).resolve()

    @property
    def cpp2c_file(self) -> Path:
        """The cpp2c output file"""
        return Path(
            self._destination,
            self._tf_spec.output_filenames[TaskFunction.CPP_TF_KEY]['source']
        ).resolve()

    @property
    def cpp2c_outer_template(self) -> Path:
        """Outer template for the cpp2c layer"""
        return Path(
            self._destination,
            f"cg-tpl.outer_{self._tf_spec.data_item_class_name}_cpp2c.cpp"
        ).resolve()

    @property
    def cpp2c_helper_template(self) -> Path:
        """Helper template path for the cpp2c layer"""
        return Path(
            self._destination,
            f"cg-tpl.helper_{self._tf_spec.data_item_class_name}_cpp2c.cpp"
        ).resolve()

    @property
    def cpp2c_streams_template(self) -> Path:
        """Extra streams template for the cpp2c layer"""
        return Path(self._templates_path, self._cpp2c_extra_streams_tpl)

    @property
    def cpp2c_template(self) -> Path:
        return Path(self._templates_path, "cg-tpl.cpp2c.cpp")

    @property
    def c2f_file(self) -> Path:
        return Path(
            self._destination,
            self._tf_spec.output_filenames[TaskFunction.C2F_KEY]["source"]
        )

    @property
    def n_extra_streams(self) -> int:
        # for now data packet generator will return the number of
        # extra streams.
        return self._tf_spec.n_streams-1

    @property
    def byte_alignment(self) -> int:
        return self._tf_spec.data_item_byte_alignment

    # DEV NOTE:
    # we cache these property to avoid generating it twice.
    # we could probably just store these in a variable instead...

    @property
    @functools.lru_cache
    def external_args(self) -> OrderedDict:
        lang = self._tf_spec.language.lower()
        # ..todo::
        #   * do we need deep copy?
        args = deepcopy(self._tf_spec.external_arguments)
        external = {
            'nTiles': {
                # placeholder source name
                'source': 'internal',
                'name': 'nTiles',
                'type': 'int' if lang == "fortran" else 'std::size_t',
                'extents': []
            }
        }

        # insert arguments into separate dict for use later
        for item in args:
            external[item] = self._tf_spec.argument_specification(item)

        return self._sort_dict(
            external.items(),
            lambda key_and_type: self._sizes.get(key_and_type[1]['type'], 0),
            True
        )

    @property
    @functools.lru_cache
    def tile_metadata_args(self) -> OrderedDict:
        lang = self._tf_spec.language.lower()
        sort_func = None

        def cpp_sort(kv_pair):
            return self._sizes.get(
                self.TILE_VARIABLE_MAPPING[kv_pair[1]['source']], 0
            )

        def fortran_sort(x):
            return self._sizes.get(
                self.F_HOST_EQUIVALENT[
                    self.TILE_VARIABLE_MAPPING[x[1]['source']]
                ],
                0
            )

        if lang == 'c++':
            sort_func = cpp_sort
        elif lang == 'fortran':
            sort_func = fortran_sort

        args = deepcopy(self._tf_spec.tile_metadata_arguments)
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
            struct_index = arg_dictionary[arg]['structure_index']
            x = '({0}) + 1' if struct_index[0].lower() == 'fluxx' else '{0}'
            y = '({0}) + 1' if struct_index[0].lower() == 'fluxy' else '{0}'
            z = '({0}) + 1' if struct_index[0].lower() == 'fluxz' else '{0}'
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

            arg_dictionary[arg]['type'] = \
                self.SOURCE_DATATYPE[arg_dictionary[arg]['source']]
        return arg_dictionary

    @property
    @functools.lru_cache
    def tile_in_args(self) -> OrderedDict:
        """
        Gets all tile_in arguments and formats them for the
        data packet generator.

        :return: OrderedDict containing all tile_in arguments.
        """
        sizes = self._sizes
        args = deepcopy(self._tf_spec.tile_in_arguments)
        arg_dictionary = self.__adjust_tile_data(args)
        return self._sort_dict(
            arg_dictionary.items(),
            lambda x: sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0),
            True
        )

    @property
    @functools.lru_cache
    def tile_in_out_args(self) -> OrderedDict:
        """
        Gets all tile_in_out arguments and formats them for the
        data packet generator.

        :return: OrderedDict containing all tile_in_out arguments.
        """
        sizes = self._sizes
        args = deepcopy(self._tf_spec.tile_in_out_arguments)
        arg_dictionary = self.__adjust_tile_data(args)
        return self._sort_dict(
            arg_dictionary.items(),
            lambda x: sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0),
            True
        )

    @property
    @functools.lru_cache
    def tile_out_args(self) -> OrderedDict:
        """
        Gets all tile_out arguments and formats them for the
        data packet generator.

        :return: OrderedDict containing all tile_out arguments.
        """
        sizes = self._sizes
        args = deepcopy(self._tf_spec.tile_out_arguments)
        arg_dictionary = self.__adjust_tile_data(args)
        return self._sort_dict(
            arg_dictionary.items(),
            lambda x: sizes.get(self.SOURCE_DATATYPE[x[1]["source"]], 0),
            True
        )

    # ..todo::
    #    * investigate why not using lru_cache here
    #    * causes an error when the scratch dict is obtained more than once
    @property
    @functools.lru_cache
    def scratch_args(self) -> OrderedDict:
        """
        Gets all scratch arguments and formats them for the
        data packet generator.

        :return: OrderedDict containing all scratch arguments.
        """
        args = deepcopy(self._tf_spec.scratch_arguments)
        arg_dictionary = OrderedDict()
        for arg in args:
            arg_dictionary[arg] = self._tf_spec.argument_specification(arg)
            arg_dictionary[arg]['extents'] = \
                parse_extents(arg_dictionary[arg]['extents'])
            arg_dictionary[arg]['lbound'] = \
                parse_lbound(arg_dictionary[arg]['lbound'])
        return self._sort_dict(
            arg_dictionary.items(),
            lambda x: (self._sizes.get(x[1]["type"], 0), x[0]),
            False
        )

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
        dict_items = [(k, v) for k, v in arguments]
        return OrderedDict(sorted(dict_items, key=sort_key, reverse=reverse))

    def warn(self, msg: str):
        self._warn(msg)

    def abort(self, msg: str):
        # print message and exit
        self._error(msg)
        exit(-1)
