import pathlib
import functools

from pkg_resources import resource_filename
from copy import deepcopy
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path

from .parse_helpers import parse_extents
from .parse_helpers import parse_lbound
from .generate_packet_file import generate_packet_file
from .Cpp2CLayerGenerator import Cpp2CLayerGenerator
from .C2FortranLayerGenerator import C2FortranLayerGenerator
from .DataPacketC2FModuleGenerator import DataPacketC2FModuleGenerator
from .TemplateUtility import TemplateUtility
from .FortranTemplateUtility import FortranTemplateUtility
from .CppTemplateUtility import CppTemplateUtility
from .AbcCodeGenerator import AbcCodeGenerator
from .TaskFunction import TaskFunction
from .BasicLogger import BasicLogger
from .LogicError import LogicError
from . import (
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG,
    LOG_LEVEL_MAX, INTERNAL_ARGUMENT, SOURCE_DATATYPE_MAPPING
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
    FORTRAN_EQUIVALENT = {
        "IntVect": "int",
        "RealVect": "real",
        "bool": "logical"
    }
    F_HOST_EQUIVALENT = FortranTemplateUtility.F_HOST_EQUIVALENT

    __DEFAULT_SOURCE_TREE_OPTS = {
        'codePath': pathlib.Path.cwd(),
        'indentSpace': ' '*4,
        'verbose': False,
        'verbosePre': '/* ',
        'verbosePost': ' */',
    }

    def __init__(
        self,
        tf_spec: TaskFunction,
        indent: int,
        logger: BasicLogger,
        sizes: dict
    ):
        if not isinstance(tf_spec, TaskFunction):
            raise TypeError(
                "TF Specification was not derived from task function."
            )

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
            tf_spec,
            header,
            source,
            indent,
            self._TOOL_NAME,
            logger
        )

        self._cpp2c_source = outputs[TaskFunction.CPP_TF_KEY]["source"]

        if self._tf_spec.language.lower() == "c++":
            self.template_utility = CppTemplateUtility
        elif self._tf_spec.language.lower() == "fortran":
            self.template_utility = FortranTemplateUtility
        else:
            self.log_and_abort(
                "No template utility for specifed language",
                LogicError()
            )

        self._cpp2c_extra_streams_tpl = "cg-tpl.cpp2c_no_extra_queue.cpp"
        if self.n_extra_streams > 0:
            self._cpp2c_extra_streams_tpl = "cg-tpl.cpp2c_extra_queue.cpp"

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
        """
        Generates templates from creating the data packet source code.
        """
        self._connectors[self.template_utility._CON_ARGS] = []
        self._connectors[self.template_utility._SET_MEMBERS] = []
        self._connectors[self.template_utility._SIZE_DET] = []
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
        """Generates the helper template with the provided JSON data."""
        with open(helper_template, 'w') as template:
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
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)
        # save templates for later use.
        self._outer_template = outer_template
        self._helper_template = helper_template

    def generate_header_code(self, destination, overwrite):
        """
        Generate C++ header
        """
        if not self._outer_template or not self._helper_template:
            raise RuntimeError(
                "Templates have not been generated. Call generate_templates "
                "first."
            )

        destination_path = self.get_destination_path(destination)
        header = destination_path.joinpath(self.header_filename)

        self._log(f"Generating header at {str(header)}", LOG_LEVEL_BASIC)
        generate_packet_file(
            header,
            self.__DEFAULT_SOURCE_TREE_OPTS,
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
        Generate C++ source code.
        """
        if not self._outer_template or not self._helper_template:
            raise RuntimeError(
                "Templates have not been generated. Call generate_templates "
                "first."
            )

        destination_path = self.get_destination_path(destination)
        source = destination_path.joinpath(self.source_filename).resolve()

        self._log(f"Generating source at {str(source)}", LOG_LEVEL_BASIC)
        generate_packet_file(
            source,
            self.__DEFAULT_SOURCE_TREE_OPTS,
            [
                self._outer_template,
                self.source_template_path,  # static path to pkg resource.
                self._helper_template
            ],
            overwrite, self._logger
        )
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)

        if self._tf_spec.language.lower() == "fortran":
            # generate cpp2c layer if necessary
            #   -> Cpp task function generator?
            outer_cpp2c = self.cpp2c_outer_template_name
            helper_cpp2c = self.cpp2c_helper_template_name
            data_item_c2f = destination_path.joinpath(self.module_file_name)

            cpp2c_layer = Cpp2CLayerGenerator(
                self._tf_spec, outer_cpp2c,
                helper_cpp2c, self._indent,
                self._logger.level,
                self.n_extra_streams, self.external_args
            )

            self._log(
                f"Generating Cpp2C helper at {str(helper_cpp2c)} and "
                f"Cpp2C outer at {str(outer_cpp2c)}",
                LOG_LEVEL_BASIC
            )
            cpp2c_layer.generate_source_code(destination, overwrite)
            cpp2c_destination = Path(destination, self.cpp2c_file_name)
            self._log(
                f"Generating Cpp2C Layer at {cpp2c_destination} using "
                f"{str(helper_cpp2c)} and {str(outer_cpp2c)}",
                LOG_LEVEL_BASIC
            )
            generate_packet_file(
                cpp2c_destination,
                self.__DEFAULT_SOURCE_TREE_OPTS,
                # dev note: ORDER MATTERS HERE!
                # If helpers is put before the base
                # template it will throw an error
                [
                    outer_cpp2c,
                    self.cpp2c_template_path,
                    helper_cpp2c,
                    self.cpp2c_streams_template_name
                ],
                overwrite, self._logger
            )
            self._log("Done", LOG_LEVEL_BASIC_DEBUG)
            # generate fortran to c layer if necessary
            c2f_layer = C2FortranLayerGenerator(
                self._tf_spec, self._indent,
                self._logger, self.n_extra_streams,
                self.external_args, self.tile_metadata_args,
                self.tile_in_args, self.tile_in_out_args,
                self.tile_out_args, self.scratch_args
            )
            # c2f layer does not use cgkit so no need
            # to call generate_packet_file
            c2f_layer.generate_source_code(destination, overwrite)

            dp_module = DataPacketC2FModuleGenerator(
                self._tf_spec, self._indent, self._logger, self.external_args
            )
            self._log(
                f"Generating mod file at {str(data_item_c2f)}",
                LOG_LEVEL_BASIC
            )
            dp_module.generate_source_code(destination, overwrite)

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
    def header_file_name(self) -> str:
        return super().header_filename

    @property
    def source_file_name(self) -> str:
        return super().source_filename

    @property
    def cpp2c_file_name(self) -> str:
        return self._tf_spec.output_filenames[
            TaskFunction.CPP_TF_KEY
        ]["source"]

    @property
    def cpp2c_outer_template_name(self) -> str:
        """
        Outer template for the cpp2c layer
        These are generated so we use the destination location.
        """
        return f"cg-tpl.outer_{self._tf_spec.data_item_class_name}_cpp2c.cpp"

    @property
    def cpp2c_helper_template_name(self) -> str:
        """
        Helper template path for the cpp2c layer
        These are generated so we use the destination location.
        """
        return f"cg-tpl.helper_{self._tf_spec.data_item_class_name}_cpp2c.cpp"

    @property
    def cpp2c_streams_template_name(self) -> Path:
        """Extra streams template for the cpp2c layer"""
        template_path = resource_filename(
            __package__, f'templates/{self._cpp2c_extra_streams_tpl}'
        )
        return Path(template_path).resolve()

    @property
    def cpp2c_template_path(self) -> Path:
        template_path = resource_filename(
            __package__, 'templates/cg-tpl.cpp2c.cpp'
        )
        return Path(template_path).resolve()

    @property
    def c2f_file_name(self) -> str:
        return self._tf_spec.output_filenames[
            TaskFunction.C2F_KEY
        ]["source"]

    @property
    def module_file_name(self) -> str:
        return self._tf_spec.output_filenames[
            TaskFunction.DATA_ITEM_KEY
        ]["module"]

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
        """
        Gets all external variables from the TaskFunction and formats them
        for convenience. Also inserts nTiles.
        """
        lang = self._tf_spec.language.lower()
        args = deepcopy(self._tf_spec.external_arguments)
        external = {
            'nTiles': {
                # placeholder source name
                'source': INTERNAL_ARGUMENT,
                'name': 'nTiles',
                'type': 'int' if lang == "fortran" else 'std::size_t',
                'extents': "()"
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
        """
        Gets all tile metadata arguments from the TaskFunction
        and formats it for easy use with this class.
        """
        lang = self._tf_spec.language.lower()
        sort_func = None

        def cpp_sort(kv_pair):
            return self._sizes.get(
                SOURCE_DATATYPE_MAPPING[kv_pair[1]['source']], 0
            )

        def fortran_sort(x):
            return self._sizes.get(
                self.F_HOST_EQUIVALENT[
                    SOURCE_DATATYPE_MAPPING[x[1]['source']]
                ],
                0
            )

        if lang == 'c++':
            sort_func = cpp_sort
        elif lang == 'fortran':
            sort_func = fortran_sort
        else:
            raise RuntimeError("Language is not supported.")

        args = deepcopy(self._tf_spec.tile_metadata_arguments)
        for key in args:
            args[key] = self._tf_spec.argument_specification(key)
            args[key]['type'] = SOURCE_DATATYPE_MAPPING[args[key]["source"]]
            if lang == "fortran":
                args[key]['type'] = self.FORTRAN_EQUIVALENT[args[key]['type']]

        return self._sort_dict(args.items(), sort_func, True)

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
            # ..todo::
            #   * adjust masking based on given index space.
            if 'variables_in' in arg_dictionary[arg]:
                mask = arg_dictionary[arg]['variables_in']
                arg_dictionary[arg]['variables_in'] = [mask[0]-1, mask[1]-1]
            if 'variables_out' in arg_dictionary[arg]:
                mask = arg_dictionary[arg]['variables_out']
                arg_dictionary[arg]['variables_out'] = [mask[0]-1, mask[1]-1]

            arg_dictionary[arg]['type'] = \
                SOURCE_DATATYPE_MAPPING[arg_dictionary[arg]['source']]
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
            lambda x: sizes.get(SOURCE_DATATYPE_MAPPING[x[1]["source"]], 0),
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
            lambda x: sizes.get(SOURCE_DATATYPE_MAPPING[x[1]["source"]], 0),
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
            lambda x: sizes.get(SOURCE_DATATYPE_MAPPING[x[1]["source"]], 0),
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

    def _sort_dict(self, arguments, sort_key, reverse) -> OrderedDict:
        """
        Sorts a given dictionary using the sort key.

        :param dict section: The dictionary to sort.
        :param func sort_key: The function to sort with.
        """
        dict_items = [(k, v) for k, v in arguments]
        return OrderedDict(sorted(dict_items, key=sort_key, reverse=reverse))

    def get_destination_path(self, destination: str) -> Path:
        destination_path = Path(destination).resolve()
        if not destination_path.is_dir():
            raise RuntimeError(f"{destination_path} does not exist")
        return destination_path

    def warn(self, msg: str):
        self._warn(msg)

    def log_and_abort(self, msg: str, e: BaseException):
        # print message and exit
        self._error(msg)
        raise e
