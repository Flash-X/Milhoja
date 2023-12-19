from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from pkg_resources import resource_filename

from .DataPacketMemberVars import DataPacketMemberVars
from .AbcCodeGenerator import AbcCodeGenerator
from .TaskFunction import TaskFunction
from .LogicError import LogicError
from .generate_packet_file import generate_packet_file
from . import (
    EXTERNAL_ARGUMENT, LOG_LEVEL_MAX, THREAD_INDEX_VAR_NAME, LOG_LEVEL_BASIC
)

_INSTANCE_ARGS = "instance_args"
_CONSTRUCT_ARGS = "host_members"
_DEVICE_MEMBERS = "get_device_members"
_C2F_ARGS = "c2f_arguments"


class Cpp2CLayerGenerator(AbcCodeGenerator):
    """
    C++ to C layer generator for Data Packets. Should only be used
    internally by the DataPacketGenerator.
    """

    def __init__(
        self,
        tf_spec: TaskFunction,
        indent,
        logger
    ):
        self.__TOOL_NAME = "Milhoja Cpp2C"
        source_name = \
            tf_spec.output_filenames[TaskFunction.CPP_TF_KEY]["source"]
        super().__init__(
            tf_spec, "", source_name, indent,
            self.__TOOL_NAME, logger
        )

        self._outer_template = self.cpp2c_outer_template_name
        self._helper_template = self.cpp2c_helper_template_name
        self._connectors = defaultdict(list)
        self._cpp2c_extra_streams_tpl = "cg-tpl.cpp2c_no_extra_queue.cpp"
        self._n_extra_streams = self._tf_spec.n_streams - 1
        if self._n_extra_streams > 0:
            self._cpp2c_extra_streams_tpl = "cg-tpl.cpp2c_extra_queue.cpp"

        self._log("Created Cpp2C layer generator", LOG_LEVEL_MAX)

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

    def generate_header_code(self, destination, overwrite):
        """No implementation for cpp2c header."""
        raise LogicError("No header file for C++ to C layer.")

    def generate_source_code(self, destination, overwrite):
        """
        Wrapper for generating the outer and helper templates.
        Note the destination gets passed into the constructor as the
        full file path for the output files, so destination gets unused in
        both functions.
        """
        outer = destination.joinpath(Path(self._outer_template))
        helper = destination.joinpath(Path(self._helper_template))
        cpp2c_destination = destination.joinpath(Path(self.source_filename))

        self._log(f"Generating outer file {outer}", LOG_LEVEL_BASIC)
        self._generate_cpp2c_outer(outer, overwrite)
        self._log(f"Generating helper file {helper}", LOG_LEVEL_BASIC)
        self._generate_cpp2c_helper(helper, overwrite)
        self._log(
            f"Generating Cpp2C Layer at {cpp2c_destination} using",
            LOG_LEVEL_BASIC
        )
        generate_packet_file(
            cpp2c_destination,
            # dev note: ORDER MATTERS HERE!
            # If helpers is put before the base
            # template it will throw an error
            [
                outer, self.cpp2c_template_path,
                helper, self.cpp2c_streams_template_name
            ],
            overwrite, self._logger
        )

    def _generate_cpp2c_outer(self, outer, overwrite):
        """
        Generates the outer template for the cpp2c layer.

        Note: The paths for each template file get passed into the
              constructor so destination is unused. This class is exclusively
              used by the data packet generator so it doesn't make
              sense to recreate the path instead of just passing it in at
              construction. The same could be argued for overwrite as well.

        :param generator: The DataPacketGenerator object that contains
                          the information to build the outer template.
        """
        if outer.is_file():
            self.warn(f"{str(outer)} already exists.")
            if not overwrite:
                self.log_and_abort(
                    f"Overwrite is {overwrite}",
                    FileExistsError()
                )

        file_name = self._tf_spec.output_filenames[TaskFunction.DATA_ITEM_KEY]
        packet_file_name = file_name['header']

        # ..todo::
        #   * replace delete / release / instantiate function names with
        #     properties in TaskFunction class
        with open(self._outer_template, 'w') as outer:
            outer.writelines([
                '/* _connector:cpp2c_outer */\n',
                f'/* _param:class_name = '
                f'{self._tf_spec.data_item_class_name} */\n\n',
                '/* _param:file_name = ',
                f'{packet_file_name} */\n',
                f'/* _param:taskfunctionname = '
                f'{self._tf_spec.name} */\n',
                f'/* _param:taskfunctionnametf = '
                f'{self._tf_spec.name}_cpp2c */\n',
                f'/* _param:taskfunctionnamec2f = '
                f'{self._tf_spec.name}_c2f */\n',
                f'/* _param:instantiate = instantiate_'
                f'{self._tf_spec.name}_packet_c */\n',
                f'/* _param:deletion = '
                f'delete_{self._tf_spec.name}_packet_c */\n',
                f'/* _param:release = release_'
                f'{self._tf_spec.name}_extra_queue_c */\n',
                f'/* _param:thread_id = {THREAD_INDEX_VAR_NAME} */\n\n',
                '/* _link:cpp2c */'
            ])

    def _generate_cpp2c_helper(self, helper, overwrite):
        """
        Generates the helper template for the cpp2c layer.

        Note: The paths for each template file get passed into the
              constructor so destination is unused. This class is exclusively
              used by the data packet generator so it doesn't make
              sense to recreate the path instead of just passing it in at
              construction. The same could be argued for overwrite as well.

        :param DataPacketGenerator data: The DataPacketGenerator calling this
                                         set of functions that contains a
                                         TaskFunction.
        :param destination: The destination of the file. This gets passed into
                            the constructor from the data packet generator so
                            this is unused.
        :param overwrite: The overwrite flag.
        """
        if helper.is_file():
            self.warn(f"{str(helper)} already exists.")
            if not overwrite:
                self.log_and_abort(
                    f"Overwrite is {overwrite}",
                    FileExistsError()
                )

        # generate DataPacketMemberVars instance for each item in TFAL.
        adjusted_args = deepcopy(self._tf_spec.dummy_arguments)
        adjusted_args = ["nTiles"] + adjusted_args

        # insert all connectors into helper template file
        with open(self._helper_template, 'w') as helper:
            spec_func = self._tf_spec.argument_specification
            n_ex_streams = self._n_extra_streams
            # write c2f dummy arg list:
            helper.writelines(
                ['\n/* _connector:c2f_argument_list */\n'] +
                ['void* packet_h,\n'] +
                ['const int queue1_h,\n'] +
                [
                    f'const int queue{i}_h,\n'
                    for i in range(2, n_ex_streams+2)
                ] +
                ['const int _nTiles_h,\n'] +
                [',\n'.join(
                    f'const void* _{item}_d' for item in adjusted_args
                )] +
                ['\n']
            )

            helper.writelines(
                ['/* _connector:get_host_members */\n'] +
                [
                    'const int queue1_h = packet_h->asynchronousQueue();\n',
                    'const int _nTiles_h = packet_h->_nTiles_h;\n'
                ] +
                [
                    f'const int queue{i}_h = '
                    f'packet_h->extraAsynchronousQueue({i});\n'
                    f'if (queue{i}_h < 0)\n'
                    f'\tthrow std::overflow_error('
                    f'"[{self._tf_spec.name}_cpp2c] '
                    'Potential overflow error when '
                    'accessing async queue id.");\n'
                    for i in range(2, n_ex_streams+2)
                ]
            )

            self._connectors[_INSTANCE_ARGS] = []
            self._connectors[_CONSTRUCT_ARGS] = []
            self._connectors[_DEVICE_MEMBERS] = []
            self._connectors[_C2F_ARGS] = []

            self._connectors[_C2F_ARGS].extend(
                ['packet_h', 'queue1_h'] +
                [f'queue{i}_h' for i in range(2, n_ex_streams+2)] +
                ['_nTiles_h']
            )

            for var in adjusted_args:
                spec = spec_func(var) if var != "nTiles" else None
                # We just need the naming scheme.
                var_info = DataPacketMemberVars(var, "", "", False)
                self._connectors[_DEVICE_MEMBERS].append(
                    f"void* {var_info.device} = static_cast<void*>"
                    f"( packet_h->{var_info.device} )"
                )

                self._connectors[_C2F_ARGS].append(var_info.device)

                if spec:
                    if spec["source"] == EXTERNAL_ARGUMENT:
                        self._connectors[_CONSTRUCT_ARGS].append(
                            f"{var}"
                        )
                        self._connectors[_INSTANCE_ARGS].append(
                            f"{spec['type']} {var}"
                        )

            self._connectors[_INSTANCE_ARGS].append("void** packet")

            # insert instantiation function arguments.
            helper.write(f'/* _connector:{_INSTANCE_ARGS} */\n')
            helper.write(',\n'.join(self._connectors[_INSTANCE_ARGS]) + '\n')

            # insert constructor arguments
            helper.write(f'/* _connector:{_CONSTRUCT_ARGS} */\n')
            helper.write(',\n'.join(self._connectors[_CONSTRUCT_ARGS]) + '\n')

            # write code to get device members
            helper.write(f'/* _connector:{_DEVICE_MEMBERS} */\n')
            helper.write(
                ';\n'.join(self._connectors[_DEVICE_MEMBERS]) + ';\n'
            )

            helper.write(f'/* _connector:{_C2F_ARGS} */\n')
            helper.write(',\n'.join(self._connectors[_C2F_ARGS]) + '\n')

    def warn(self, msg):
        self._warn(msg)

    def log_and_abort(self, msg, e: BaseException):
        self._error(msg)
        raise e
