from collections import defaultdict
from pathlib import Path

from .DataPacketMemberVars import DataPacketMemberVars
from .AbcCodeGenerator import AbcCodeGenerator
from .BasicLogger import BasicLogger
from milhoja import LOG_LEVEL_MAX
from milhoja import THREAD_INDEX_VAR_NAME
from milhoja import TaskFunction
from .LogicError import LogicError

_ARG_LIST_KEY = "c2f_argument_list"
_INST_ARGS_KEY = "instance_args"
_HOST_MEMBERS_KEY = "get_host_members"


class Cpp2CLayerGenerator(AbcCodeGenerator):
    """
    C++ to C layer generator for Data Packets. Should only be used
    internally by the DataPacketGenerator.
    """

    def __init__(
        self,
        tf_spec,
        outer,
        helper,
        indent,
        log_level,
        n_extra_streams,
        external_args
    ):
        self._outer_template = outer
        self._helper_template = helper
        self.__TOOL_NAME = "Milhoja Cpp2C"
        self._n_extra_streams = n_extra_streams
        self._connectors = defaultdict(list)
        self._externals = external_args

        logger = BasicLogger(log_level)
        super().__init__(
            tf_spec, outer,
            helper, indent,
            self.__TOOL_NAME, logger
        )
        self._log("Created Cpp2C layer generator", LOG_LEVEL_MAX)

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

        self._generate_cpp2c_outer(outer, overwrite)
        self._generate_cpp2c_helper(helper, overwrite)

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

    def _insert_connector_arguments(self, dpinfo_order: list):
        """
        Inserts various connector arguments into the connectors dictionary.

        :param dict data: The dictionary containing the data packet JSON data.
        :param dict connectors: The connectors dictionary to write to
                                containing all cgkit connectors.
        :param list dpinfo_order: A list of DataPacketMemberVars objects that
                                  use all of the items in the
                                  task_function_argument_list.
        """
        nTiles_type = self._externals["nTiles"]["type"]

        # initialize the boilerplate values for host members.
        # one queue always exists in the data packet
        # need to insert nTiles host manually
        self._connectors[_HOST_MEMBERS_KEY] = [
            'const int queue1_h = packet_h->asynchronousQueue();\n',
            f'const {nTiles_type} _nTiles_h = packet_h->_nTiles_h;\n'
        ]
        # insert the number of extra streams as a connector
        n_ex_streams = self._n_extra_streams
        self._connectors[_HOST_MEMBERS_KEY].extend([
            f'const int queue{i}_h = packet_h->extraAsynchronousQueue({i});\n'
            f'if (queue{i}_h < 0)\n'
            f'\tthrow std::overflow_error("[{self._tf_spec.name}_cpp2c] '
            'Potential overflow error when accessing async queue id.");\n'
            for i in range(2, n_ex_streams+2)
        ])

        # extend the argument list connector using dpinfo_order
        self._connectors[_ARG_LIST_KEY].extend(
            [('packet_h', 'void*')] +
            [('queue1_h', 'const int')] +
            [(f'queue{i}_h', 'const int') for i in range(2, n_ex_streams+2)] +
            [('_nTiles_h', f'const {nTiles_type}')] +
            [(item.device, 'const void*') for item in dpinfo_order]
        )

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
        dummy_args = ["nTiles"] + self._tf_spec.dummy_arguments
        dpinfo_order = ([
            DataPacketMemberVars(item, '', '', False)
            for item in dummy_args
        ])
        # insert connectors into dictionary
        self._insert_connector_arguments(dpinfo_order)
        # instance args only found in general section
        self._connectors[_INST_ARGS_KEY] = [
            (key, f'{data["type"]}')
            for key, data in self._externals.items() if key != "nTiles"
        ] + [('packet', 'void**')]

        # insert all connectors into helper template file
        with open(self._helper_template, 'w') as helper:
            helper.writelines(
                ['/* _connector:get_host_members */\n'] +
                self._connectors[_HOST_MEMBERS_KEY]
            )
            helper.writelines(
                ['\n/* _connector:c2f_argument_list */\n'] +
                [',\n'.join([f"{item[1]} {item[0]}"
                             for item in self._connectors[_ARG_LIST_KEY]])] +

                ['\n\n/* _connector:c2f_arguments */\n'] +
                [',\n'.join([f"{item[0]}"
                             for item in self._connectors[_ARG_LIST_KEY]])] +

                ['\n\n/* _connector:get_device_members */\n'] +
                [
                    ''.join([
                        f'void* {item.device} = static_cast<void*>( '
                        f'packet_h->{item.device} );\n'
                        for item in dpinfo_order
                    ])
                ] +

                ['\n/* _connector:instance_args */\n'] +
                [','.join([f'{item[1]} {item[0]}'
                           for item in self._connectors[_INST_ARGS_KEY]])] +

                ['\n\n/* _connector:host_members */\n'] +
                [
                    ','.join([
                        item for item in self._externals.keys()
                        if item != 'nTiles'
                    ])
                ]
            )

    def warn(self, msg):
        self._warn(msg)

    def log_and_abort(self, msg, e: BaseException):
        self._error(msg)
        raise e
