from milhoja import LOG_LEVEL_BASIC
from milhoja import LOG_LEVEL_BASIC_DEBUG
from pathlib import Path
from collections import defaultdict
from DataPacketMemberVars import DataPacketMemberVars
from milhoja import TaskFunction
from milhoja import AbcCodeGenerator
from milhoja import BasicLogger
from milhoja import LOG_LEVEL_MAX

_ARG_LIST_KEY = "c2f_argument_list"
_INST_ARGS_KEY = "instance_args"
_HOST_MEMBERS_KEY = "get_host_members"

class Cpp2CLayerGenerator(AbcCodeGenerator):

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
        self.__TOOL_NAME = self.__class__.__name__
        self._n_extra_streams = n_extra_streams
        self._connectors = defaultdict(list)
        self._externals = external_args

        logger = BasicLogger(log_level)
        super().__init__(tf_spec, outer, helper, indent, self.__TOOL_NAME, logger)
        
        self._log("Created Cpp2C layer generator", LOG_LEVEL_MAX)

    def generate_header_code(self):
        """No implementation for cpp2c header."""
        return
    
    def generate_source_code(self):
        self._generate_cpp2c_outer()
        self._generate_cpp2c_helper()

    def _generate_cpp2c_outer(self):
        """
        Generates the outer template for the cpp2c layer.

        :param generator: The DataPacketGenerator object that contains the information to build the outer template.
        """
        if self._outer_template.is_file():
            self._warn(f"{str(self._outer_template)} already exists. Overwriting.")

        # ..todo::
        #   * replace delete / release / instantiate function names with properties in TaskFunction class
        with open(self._outer_template, 'w') as outer:
            outer.writelines([
                '/* _connector:cpp2c_outer */\n',
                f'/* _param:class_name = {self._tf_spec.data_item_class_name} */\n\n',
                f'/* _param:taskfunctionname = {self._tf_spec.name} */\n',
                f'/* _param:taskfunctionnametf = {self._tf_spec.name}_Cpp2C */\n',
                f'/* _param:taskfunctionnamec2f = {self._tf_spec.name}_C2F */\n',
                f'/* _param:instantiate = instantiate_{self._tf_spec.data_item_class_name}_C */\n',
                f'/* _param:deletion = delete_{self._tf_spec.data_item_class_name}_C */\n',
                f'/* _param:release = release_{self._tf_spec.data_item_class_name}_extra_queue_C */\n\n',
                '/* _link:cpp2c */'
            ])

    def _insert_connector_arguments(self, dpinfo_order: list):
        """
        Inserts various connector arguments into the connectors dictionary.
        
        :param dict data: The dictionary containing the data packet JSON data.
        :param dict connectors: The connectors dictionary to write to containing all cgkit connectors.
        :param list dpinfo_order: A list of DataPacketMemberVars objects that use all of the items in the task_function_argument_list.
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
            f'const int queue{i}_h = packet_h->extraAsynchronousQueue({i});\n' + 
            f'if (queue{i}_h < 0)\n' + 
            f'\tthrow std::overflow_error("[{self._tf_spec.name}_Cpp2C] Potential overflow error when accessing async queue id.");\n'
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

    def _generate_cpp2c_helper(self):
        """
        Generates the helper template for the cpp2c layer.
        
        :param DataPacketGenerator data: The DataPacketGenerator calling this set of functions that contains a TaskFunction .
        """
        # generate DataPacketMemberVars instance for each item in TFAL.
        dummy_args = ["nTiles"] + self._tf_spec.dummy_arguments
        dpinfo_order = ([DataPacketMemberVars(item, '', '', False) for item in dummy_args])
        # insert connectors into dictionary
        self._insert_connector_arguments(dpinfo_order)
        # instance args only found in general section
        self._connectors[_INST_ARGS_KEY] = [
            (key, f'{data["type"]}') for key, data in self._externals.items() if key != "nTiles"
        ] + [('packet', 'void**')]
 
        # insert all connectors into helper template file
        with open(self._helper_template, 'w') as helper:
            helper.writelines(
                ['/* _connector:get_host_members */\n'] +
                self._connectors[_HOST_MEMBERS_KEY]
            )
            helper.writelines(
                ['\n/* _connector:c2f_argument_list */\n'] +
                [',\n'.join([f"{item[1]} {item[0]}" for item in self._connectors[_ARG_LIST_KEY]])] +

                ['\n\n/* _connector:c2f_arguments */\n'] +
                [',\n'.join([f"{item[0]}" for item in self._connectors[_ARG_LIST_KEY]])] +

                ['\n\n/* _connector:get_device_members */\n'] +
                [''.join([f'void* {item.device} = static_cast<void*>( packet_h->{item.device} );\n' 
                        for item in dpinfo_order])] +

                ['\n/* _connector:instance_args */\n'] +
                [','.join([f'{item[1]} {item[0]}' for item in self._connectors[_INST_ARGS_KEY]])] +

                ['\n\n/* _connector:host_members */\n'] +
                [','.join([item for item in self._externals.keys() if item != 'nTiles'])]
            )
