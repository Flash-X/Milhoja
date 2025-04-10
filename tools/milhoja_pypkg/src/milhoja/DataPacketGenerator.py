import functools

from pkg_resources import resource_filename
from copy import deepcopy
from collections import defaultdict, OrderedDict
from pathlib import Path

from .parse_helpers import parse_extents
from .generate_packet_file import generate_packet_file
from .FortranTemplateUtility import FortranTemplateUtility
from .CppTemplateUtility import CppTemplateUtility
from .AbcCodeGenerator import AbcCodeGenerator
from .TaskFunction import TaskFunction
from .BasicLogger import BasicLogger
from .LogicError import LogicError
from . import (
    LOG_LEVEL_MAX, INTERNAL_ARGUMENT, GRID_DATA_EXTENTS, LOG_LEVEL_BASIC,
    LOG_LEVEL_BASIC_DEBUG, VECTOR_ARRAY_EQUIVALENT, SOURCE_DATATYPES,
    F2C_TYPE_MAPPING
)
from .milhoja_pypkg_opts import opts


class DataPacketGenerator(AbcCodeGenerator):
    """
    Responsible for generating code related to DataPackets.

    ..todo::
        * check if lru_caching is necessary
    """

    def __init__(
        self, tf_spec: TaskFunction, indent: int, logger: BasicLogger,
        sizes: dict
    ):
        self._TOOL_NAME = "Milhoja DataPacket"
        self._sizes = deepcopy(sizes)
        self._indent = indent
        self._size_connectors = defaultdict(str)
        self._connectors = defaultdict(list)
        self._params = defaultdict(str)

        outputs = tf_spec.output_filenames
        header = outputs[TaskFunction.DATA_ITEM_KEY]["header"]
        source = outputs[TaskFunction.DATA_ITEM_KEY]["source"]

        super().__init__(
            tf_spec, header, source, indent, self._TOOL_NAME, logger
        )

        self._cpp2c_source = outputs[TaskFunction.CPP_TF_KEY]["source"]

        if self._tf_spec.language.lower() == "c++":
            self.template_utility = CppTemplateUtility(tf_spec)
        elif self._tf_spec.language.lower() == "fortran":
            self.template_utility = FortranTemplateUtility(tf_spec)
        else:
            self.log_and_abort(
                "No template utility for specifed language", LogicError()
            )

        self._outer_template = None
        self._helper_template = None
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

    def generate_templates(self, destination, overwrite):
        """Generates templates from creating the data packet source code."""
        self._connectors[self.template_utility._CON_ARGS] = []
        self._connectors[self.template_utility._SET_MEMBERS] = []
        if opts['nxyzb_args']:
            self._connectors[self.template_utility._CONXYZ_ARGS] = \
                ['\nconst int nxb', 'const int nyb', 'const int nzb']
            self._connectors[self.template_utility._HOSTXYZ_MEMBERS] = \
                ['\nnxb', 'nyb', 'nzb']
            self._connectors[self.template_utility._TILECONST_MEMBERS] = \
                ['const int nxb;', 'const int nyb;', 'const int nzb;']
            self._connectors[self.template_utility._SET_TILECONST] = \
                ['nxb{nxb}', 'nyb{nyb}', 'nzb{nzb}']
        else:
            self._connectors[self.template_utility._CONXYZ_ARGS] = []
            self._connectors[self.template_utility._HOSTXYZ_MEMBERS] = []
            self._connectors[self.template_utility._TILECONST_MEMBERS] = []
            self._connectors[self.template_utility._SET_TILECONST] = []
        self._connectors[self.template_utility._SIZE_DET] = []
        self._connectors[self.template_utility._SET_SIZE_DET] = []
        self._set_default_params()

        destination_path = self.get_destination_path(destination)
        helper_template = \
            destination_path.joinpath(self.helper_template_name).resolve()

        if helper_template.is_file():
            self.warn(f"{str(helper_template)} already exists.")
            if not overwrite:
                self.log_and_abort(
                    f"Overwrite flag is {overwrite}", FileExistsError()
                )

        msg = f"Generating helper template {helper_template}"
        self._log(msg, LOG_LEVEL_BASIC)
        # get all data dicts
        external = self.external_args
        metadata = self.tile_metadata_args
        tile_in = self.tile_in_args
        tile_in_out = self.tile_in_out_args
        tile_out = self.tile_out_args
        scratch = self.scratch_args

        """Generates the helper template with the provided JSON data."""
        with open(helper_template, 'w') as template:
            self.template_utility.iterate_externals(
                self._connectors, self._size_connectors, external,
                self._tf_spec.dummy_arguments
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
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)

        outer_template = \
            destination_path.joinpath(self.outer_template_name).resolve()

        if outer_template.is_file():
            self.warn(f"{str(outer_template)} already exists.")
            if not overwrite:
                self.log_and_abort(
                    f"Overwrite is {overwrite}. Abort.",
                    FileExistsError()
                )

        msg = f"Generating outer template {outer_template}"
        self._log(msg, LOG_LEVEL_BASIC)
        with open(outer_template, 'w') as outer:
            outer.writelines([
                '/* _connector:datapacket_outer */\n',
                '/* _link:datapacket */\n'
            ])
            outer.write(
                '\n'.join(
                    f'/* _param:{item} = {self._params[item]} */'
                    for item in self._params
                )
            )
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)
        # save templates for later use.
        self._outer_template = outer_template
        self._helper_template = helper_template

    def generate_header_code(self, destination, overwrite):
        """
        Generates C++ header. generate_templates must be called first.
        """
        if not self._outer_template or not self._helper_template:
            raise RuntimeError("Templates have not been generated.")

        destination_path = self.get_destination_path(destination)
        header = destination_path.joinpath(self.header_filename)

        self._log(f"Generating header at {str(header)}", LOG_LEVEL_BASIC)
        generate_packet_file(
            header,
            [
                self._outer_template,
                self.header_template_path,  # static path to pkg resource.
                self._helper_template
            ],
            overwrite, self._logger
        )
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)

    def generate_source_code(self, destination, overwrite):
        """
        Generate C++ source code. generate_templates must be called first.
        """
        if not self._outer_template or not self._helper_template:
            raise RuntimeError("Missing generated templates.")

        destination_path = self.get_destination_path(destination)
        source = destination_path.joinpath(self.source_filename).resolve()

        self._log(f"Generating source at {str(source)}", LOG_LEVEL_BASIC)
        generate_packet_file(
            source,
            [
                self._outer_template,
                self.source_template_path,  # static path to pkg resource.
                self._helper_template
            ],
            overwrite, self._logger
        )
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)

    @property
    def language(self):
        return self._tf_spec.language.lower()

    @property
    def packet_class_name(self):
        """Class name of the data packet"""
        return self._tf_spec.data_item_class_name

    @property
    def ppd_name(self):
        """Preprocessor string for the data packet"""
        return f'{self.packet_class_name.upper()}_UNIQUE_IFNDEF_H_'

    @property
    def helper_template_name(self) -> str:
        return f"cg-tpl.helper_{self._tf_spec.data_item_class_name}.cpp"

    @property
    def outer_template_name(self) -> str:
        return f"cg-tpl.outer_{self._tf_spec.data_item_class_name}.cpp"

    @property
    def header_template_path(self) -> Path:
        template_path = resource_filename(
            __package__,
            'templates/cg-tpl.datapacket_header.cpp'
        )
        return Path(template_path).resolve()

    @property
    def source_template_path(self) -> Path:
        template_path = resource_filename(
            __package__,
            'templates/cg-tpl.datapacket.cpp'
        )
        return Path(template_path).resolve()

    @property
    def n_extra_streams(self) -> int:
        # data packet generator uses the number of *extra* streams for now.
        # The cpp2c and c2f layer generators also use the number of extra
        # streams.
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
        """
        Gets all external variables from the TaskFunction and formats them
        for convenience. Also inserts nTiles.
        """
        lang = self._tf_spec.language.lower()
        args = deepcopy(self._tf_spec.external_arguments)
        external = {
            'nTiles': {
                'source': INTERNAL_ARGUMENT,
                'name': 'nTiles',
                'type': 'int' if lang == "fortran" else 'std::size_t',
                'extents': "()"
            }
        }

        # insert arguments into separate dict for use later
        for item in args:
            external[item] = self._tf_spec.argument_specification(item)
            dtype = external[item]["type"]
            # Note: Since the data packet generator needs its own internal
            #       representations of each item in the packet,
            #       I convert any potential fortran names into C++ names.
            #       This is because the data packet generator is always
            #       generating C++ code, so it makes sense to have the
            #       internal representation use C++ type names, despite the
            #       tf spec being language agnostic.
            external[item]["type"] = F2C_TYPE_MAPPING.get(dtype, dtype)

        # Note: Python has stable sorting, so it's okay to do this to
        #       sort by multiple criteria.
        result = self._sort_dict(
            external.items(),
            lambda kat: kat[0],
            False
        )
        type_sort = self._sort_dict(
            result.items(),
            lambda key_and_type: (self._sizes.get(key_and_type[1]['type'], 0)),
            True
        )
        return type_sort

    @property
    @functools.lru_cache
    def tile_metadata_args(self) -> OrderedDict:
        """
        Gets all tile metadata arguments from the TaskFunction
        and formats it for easy use with this class.

        Note that internally, lbound arguments are treated as tile metadata.
        The two are combined in this function, with any non-lbound metadata
        sorted and appearing before any lbound metadata. This likely won't
        cause any issues in the near future, but we should consider fully
        separating the lbound section so there are no problems with how the
        data is inserted into the packet.

        Since lbound arguments are all the same type (int / IntVect),
        they are only sorted by name.
        """
        lang = self._tf_spec.language.lower()
        sort_func = None

        def cpp_sort(kv_pair):
            return self._sizes.get(
                SOURCE_DATATYPES[kv_pair[1]['source']], 0
            )

        def fortran_sort(x):
            dtype = SOURCE_DATATYPES[x[1]['source']]
            dtype = VECTOR_ARRAY_EQUIVALENT.get(dtype, dtype)
            return self._sizes.get(dtype, 0)

        if lang == 'c++':
            sort_func = cpp_sort
        elif lang == 'fortran':
            sort_func = fortran_sort
        else:
            raise RuntimeError("Language is not supported.")

        # we don't need to check for types here because tile metadata
        # has predetermined types based on milhoja classes.
        args = deepcopy(self._tf_spec.tile_metadata_arguments)
        mdata_names = []
        for names in args.values():
            mdata_names.extend(names)

        mdata = {}
        for key in mdata_names:
            spec = deepcopy(self._tf_spec.argument_specification(key))
            mdata[key] = spec
            mdata[key]['type'] = SOURCE_DATATYPES[mdata[key]["source"]]
            if lang == "fortran":
                dtype = mdata[key]['type']
                # Fortran has no equivalent of unsigned int.
                if dtype == 'unsigned int':
                    dtype = 'int'

                mdata[key]['type'] = \
                    VECTOR_ARRAY_EQUIVALENT.get(dtype, dtype)

        # append lbounds to tile metadata information.
        lbound_names = sorted(deepcopy(self._tf_spec.lbound_arguments))
        lbounds = OrderedDict()
        for key in lbound_names:
            spec = deepcopy(self._tf_spec.argument_specification(key))
            lbounds[key] = spec
            lbounds[key]['type'] = \
                SOURCE_DATATYPES[lbounds[key]["source"]]
            if lang == "fortran":
                dtype = lbounds[key]['type']
                lbounds[key]['type'] = \
                    VECTOR_ARRAY_EQUIVALENT.get(dtype, dtype)

        # NOTE: Sorting the lbounds dictionary is pointless because they are
        #       all sorted by type. And all lbound types are the same. It's
        #       possible to improve the sort function by allowing for a
        #       second key to sort by.
        mdata = self._sort_dict(mdata.items(), sort_func, True)
        mdata.update(lbounds)
        return mdata

    def __adjust_tile_data(self, args: dict) -> dict:
        """
        Adjust the tile data from the TaskFunction class for use with the
        DataPacketGenerator.

        :param dicts args: The dict of variables from the TF.
        """
        arg_dictionary = {}
        block_extents = self.block_extents
        nguard = self.n_guardcells
        for arg in args:
            arg_dictionary[arg] = deepcopy(
                self._tf_spec.argument_specification(arg)
            )
            struct_index = arg_dictionary[arg]['structure_index']

            # NOTE:
            #   We need to deep copy this otherwise we accidentally overwrite
            #   the strings in the dictionary... That was a fun debugging
            #   experience...
            extents = deepcopy(GRID_DATA_EXTENTS[struct_index[0].upper()])
            for idx, size in enumerate(block_extents):
                extents[idx] = extents[idx].format(size, nguard)
            arg_dictionary[arg]['extents'] = extents

            # adjust masking.
            # ..todo::
            #   * adjust masking based on given index space.
            if 'variables_in' in arg_dictionary[arg]:
                mask = arg_dictionary[arg]['variables_in']
                arg_dictionary[arg]['variables_in'] = [mask[0]-1, mask[1]-1]
            if 'variables_out' in arg_dictionary[arg]:
                mask = arg_dictionary[arg]['variables_out']
                arg_dictionary[arg]['variables_out'] = [mask[0]-1, mask[1]-1]

            # grid data has pre determined types as well.
            arg_dictionary[arg]['type'] = \
                SOURCE_DATATYPES[arg_dictionary[arg]['source']]
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
            lambda x: sizes.get(SOURCE_DATATYPES[x[1]["source"]], 0),
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
            lambda x: sizes.get(SOURCE_DATATYPES[x[1]["source"]], 0),
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
            lambda x: sizes.get(SOURCE_DATATYPES[x[1]["source"]], 0),
            True
        )

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
            arg_dictionary[arg] = deepcopy(
                self._tf_spec.argument_specification(arg)
            )
            arg_dictionary[arg]['extents'] = \
                parse_extents(arg_dictionary[arg]['extents'])
            arg_dictionary[arg]['lbound'] = arg_dictionary[arg]['lbound']
            # scratch does not have predetermined data types, so we need
            # to always convert to c++ types here as well.
            dtype = arg_dictionary[arg]['type']
            arg_dictionary[arg]['type'] = \
                F2C_TYPE_MAPPING.get(dtype, dtype)
        return self._sort_dict(
            arg_dictionary.items(),
            lambda x: (self._sizes.get(x[1]["type"], 0), x[0]),
            False
        )

    @property
    def block_extents(self):
        """Gets tf spec block interior shape"""
        return self._tf_spec.block_interior_shape

    @property
    def n_guardcells(self):
        """Get number of guard cells from tf spec"""
        return self._tf_spec.n_guardcells

    def _sort_dict(self, arguments, sort_key, reverse) -> OrderedDict:
        """
        Sorts a given dictionary using the sort key.

        :param dict section: The dictionary to sort.
        :param func sort_key: The function to sort with.
        :param func reverse: Reverse the order of the sort.
        """
        dict_items = [(k, v) for k, v in arguments]
        return OrderedDict(sorted(dict_items, key=sort_key, reverse=reverse))

    def get_destination_path(self, destination: str) -> Path:
        """Creates a new path using the passed in destination."""
        destination_path = Path(destination).resolve()
        if not destination_path.is_dir():
            raise RuntimeError(f"{destination_path} does not exist")
        return destination_path

    def warn(self, msg: str):
        self._warn(msg)

    def log_and_abort(self, msg: str, e: BaseException):
        """Print a message and raise error e."""
        self._error(msg)
        raise e
